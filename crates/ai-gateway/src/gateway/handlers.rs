//! OpenAI API endpoint handlers for the OBEY-API gateway.
//!
//! Requirements: 2.1-2.12

use axum::{
    extract::{Json, State},
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::atomic::Ordering;

use crate::config::load_and_validate_config;
use crate::error::GatewayError;
use crate::gateway::apply_runtime_config_update;
use crate::logger::LogEntry;
use crate::models::openai::{Choice, OpenAIRequest, OpenAIResponse};
use crate::providers::Model;
use crate::router::trace_id::generate_trace_id;

#[derive(Debug, Clone)]
struct RequestLogContext {
    trace_id: String,
    status_code: u16,
    duration_ms: u64,
    provider: String,
    requested_model: String,
    responded_model: Option<String>,
    cost: f64,
}

impl RequestLogContext {
    fn from_response(request: &OpenAIRequest, trace_id: String, duration_ms: u64, response: &crate::models::openai::OpenAIResponse) -> Self {
        Self {
            trace_id,
            status_code: StatusCode::OK.as_u16(),
            duration_ms,
            provider: response.extra.get("gateway_provider").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
            requested_model: request.model.clone(),
            responded_model: response.extra.get("gateway_responded_model").and_then(|v| v.as_str()).map(|s| s.to_string()).or_else(|| if response.model.is_empty() { None } else { Some(response.model.clone()) }),
            cost: response.extra.get("gateway_cost").and_then(|v| v.as_f64()).unwrap_or(0.0),
        }
    }

    fn from_error(request: &OpenAIRequest, trace_id: String, duration_ms: u64, error: &GatewayError) -> Self {
        let provider = match error {
            GatewayError::Provider { provider, .. } => provider.clone(),
            GatewayError::AllProvidersFailed(agg) => agg.attempts.first().map(|attempt| attempt.provider.clone()).unwrap_or_default(),
            _ => String::new(),
        };
        Self {
            trace_id,
            status_code: error.status_code().as_u16(),
            duration_ms,
            provider,
            requested_model: request.model.clone(),
            responded_model: None,
            cost: 0.0,
        }
    }
}

/// Log a completed request to the SQLite database for the dashboard log viewer.
fn log_request(state: &super::AppState, request: &OpenAIRequest, context: &RequestLogContext) {
    let entry = LogEntry {
        trace_id: context.trace_id.clone(),
        timestamp: chrono::Utc::now(),
        method: "POST".to_string(),
        path: "/v1/chat/completions".to_string(),
        model: context.responded_model.clone().unwrap_or_else(|| context.requested_model.clone()),
        provider: context.provider.clone(),
        status_code: context.status_code,
        duration_ms: context.duration_ms,
        cost: context.cost,
        request_body: None,
        response_body: None,
        requested_model: Some(request.model.clone()),
        responded_model: context.responded_model.clone(),
    };
    if let Err(e) = state.logger.log(entry) {
        tracing::warn!(error = %e, trace_id = %context.trace_id, "Failed to write request log entry");
    }
}

fn trace_id_from_headers(headers: &HeaderMap) -> String {
    let request_id = headers
        .get("x-request-id")
        .or_else(|| headers.get("x-trace-id"))
        .and_then(|value| value.to_str().ok());
    generate_trace_id(request_id)
}

fn attach_trace_id_header(response: &mut Response, trace_id: &str) {
    let header_name = HeaderName::from_static("x-trace-id");
    if let Ok(header_value) = HeaderValue::from_str(trace_id) {
        response.headers_mut().insert(header_name, header_value);
    }
}

use super::AppState;

// ---------------------------------------------------------------------------
// Error → HTTP response mapping
// ---------------------------------------------------------------------------

impl IntoResponse for GatewayError {
    fn into_response(self) -> Response {
        let (status, body) = match &self {
            GatewayError::InvalidRequest(msg) => (
                StatusCode::BAD_REQUEST,
                serde_json::json!({ "error": { "message": msg, "type": "invalid_request_error" } }),
            ),
            GatewayError::Authentication(msg) => (
                StatusCode::UNAUTHORIZED,
                serde_json::json!({ "error": { "message": msg, "type": "authentication_error" } }),
            ),
            GatewayError::AllProvidersFailed(agg) => (
                StatusCode::BAD_GATEWAY,
                serde_json::json!({
                    "error": {
                        "message": "All providers failed to process the request",
                        "type": "all_providers_failed",
                        "attempts": agg.attempts,
                    }
                }),
            ),
            GatewayError::RateLimitExceeded(provider) => (
                StatusCode::TOO_MANY_REQUESTS,
                serde_json::json!({ "error": { "message": format!("Rate limit exceeded for provider: {}", provider), "type": "rate_limit_error" } }),
            ),
            GatewayError::Timeout(secs) => (
                StatusCode::GATEWAY_TIMEOUT,
                serde_json::json!({ "error": { "message": format!("Request timeout after {}s", secs), "type": "timeout_error" } }),
            ),
            GatewayError::Provider { provider: _, message: _, status_code } => {
                let sc = status_code
                    .and_then(|c| StatusCode::from_u16(c).ok())
                    .unwrap_or(StatusCode::BAD_GATEWAY);
                (
                    sc,
                    serde_json::json!({ "error": { "message": self.to_string(), "type": "provider_error" } }),
                )
            },
            _ => (
                StatusCode::INTERNAL_SERVER_ERROR,
                serde_json::json!({ "error": { "message": self.to_string(), "type": "server_error" } }),
            ),
        };

        (status, Json(body)).into_response()
    }
}

// ---------------------------------------------------------------------------
// GET /health  (Req 20.1-20.3)
// ---------------------------------------------------------------------------

/// Health check endpoint — returns 200 when operational, 503 when shutting down.
pub async fn health_check(State(state): State<AppState>) -> Response {
    if state.shutting_down.load(Ordering::Relaxed) {
        (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({ "status": "shutting_down" }))).into_response()
    } else {
        (StatusCode::OK, Json(serde_json::json!({ "status": "ok" }))).into_response()
    }
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions  (Req 2.1)
// ---------------------------------------------------------------------------

/// Chat completions handler — streaming and non-streaming.
pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<OpenAIRequest>,
) -> Response {
    tracing::info!(model = %request.model, stream = request.stream, "Received chat completion request");
    let trace_id = trace_id_from_headers(&headers);
    if request.stream {
        chat_completions_stream(state, request, trace_id).await
    } else {
        chat_completions_non_stream(state, request, trace_id).await
    }
}

async fn chat_completions_non_stream(state: AppState, request: OpenAIRequest, trace_id: String) -> Response {
    state.metrics.start_request();
    let start = std::time::Instant::now();
    tracing::debug!(model = %request.model, "Routing non-stream request");

    // Check semantic cache before routing to providers
    // Skip cache for requests with tools/tool_choice — these are stateful
    let skip_cache = request.extra.contains_key("tools") || request.extra.contains_key("tool_choice");
    if !skip_cache {
        if let Some(ref cache) = state.cache {
            match cache.get(&request).await {
                Ok(Some(cached_response)) => {
                    state.metrics.record_cache_hit();
                    state.metrics.complete_request(start.elapsed().as_millis() as u64);
                    // Parse the cached JSON response
                    match serde_json::from_str::<crate::models::openai::OpenAIResponse>(&cached_response) {
                        Ok(resp) => {
                            let mut response = Json(resp).into_response();
                            attach_trace_id_header(&mut response, &trace_id);
                            return response;
                        }
                        Err(_) => {
                            tracing::warn!("Failed to parse cached response, falling through to provider");
                        }
                    }
                }
                Ok(None) => {
                    state.metrics.record_cache_miss();
                }
                Err(e) => {
                    tracing::warn!("Cache lookup failed: {}, falling through to provider", e);
                    state.metrics.record_cache_miss();
                }
            }
        }
    }

    match state.router.route_request(&request).await {
        Ok(response) => {
            // Only cache responses that are safe to replay (no tool_calls, complete, etc.)
            if let Some(ref cache) = state.cache {
                if crate::router::router::Router::should_cache_response(&response) {
                    let response_json = serde_json::to_string(&response).unwrap_or_default();
                    if let Err(e) = cache.set(&request, &response_json, 0.0).await {
                        tracing::warn!("Failed to cache response: {}", e);
                    }
                }
            }
            let duration_ms = start.elapsed().as_millis() as u64;
            state.metrics.complete_request(duration_ms);
            let log_context = RequestLogContext::from_response(&request, trace_id.clone(), duration_ms, &response);
            log_request(&state, &request, &log_context);
            let mut http_response = Json(response).into_response();
            attach_trace_id_header(&mut http_response, &trace_id);
            http_response
        }
        Err(e) => {
            let duration_ms = start.elapsed().as_millis() as u64;
            state.metrics.complete_request(duration_ms);
            let log_context = RequestLogContext::from_error(&request, trace_id.clone(), duration_ms, &e);
            log_request(&state, &request, &log_context);
            let mut response = e.into_response();
            attach_trace_id_header(&mut response, &trace_id);
            response
        }
    }
}

async fn chat_completions_stream(state: AppState, request: OpenAIRequest, trace_id: String) -> Response {
    state.metrics.start_request();
    let start = std::time::Instant::now();
    tracing::debug!(
        trace_id = %trace_id,
        model = %request.model,
        "Client requested streaming response; gateway currently buffers the full upstream response before synthesizing SSE"
    );

    // Route the request first (provider always returns non-streaming JSON).
    // Errors here happen BEFORE any SSE chunks are sent, so we return a
    // normal JSON error response with the proper HTTP status code.
    let response = match state.router.route_request(&request).await {
        Ok(resp) => resp,
        Err(e) => {
            let duration_ms = start.elapsed().as_millis() as u64;
            state.metrics.complete_request(duration_ms);
            let log_context = RequestLogContext::from_error(&request, trace_id.clone(), duration_ms, &e);
            log_request(&state, &request, &log_context);
            let mut response = e.into_response();
            attach_trace_id_header(&mut response, &trace_id);
            return response;
        }
    };

    // Log the successful routed request before streaming begins
    let duration_ms = start.elapsed().as_millis() as u64;
    let log_context = RequestLogContext::from_response(&request, trace_id.clone(), duration_ms, &response);
    log_request(&state, &request, &log_context);

    // Success — convert the complete response into SSE chunk format for the client.
    //
    // The gateway always fetches a complete non-streaming response from the
    // provider, then re-chunks it as SSE for the client.  The chunk format
    // must exactly match the OpenAI streaming spec so that clients like
    // Roo Code and Kilo Code can parse tool_calls correctly.
    //
    // Reference (real OpenAI stream for tool_calls):
    //   Chunk 1: delta has role, content:null, tool_calls[0] with index/id/type/function.name/arguments:""
    //   Chunk 2..N: delta has tool_calls[0] with index + function.arguments fragment
    //   Final: delta:{}, finish_reason:"tool_calls", usage:{...}
    let stream_trace_id = trace_id.clone();
    let stream = async_stream::stream! {
        let choice = response.choices.first();

        // Extract tool_calls from message extra fields
        let tool_calls = choice
            .and_then(|c| c.message.extra.get("tool_calls"))
            .and_then(|v| v.as_array())
            .cloned();

        let has_tool_calls = tool_calls.as_ref().is_some_and(|tc| !tc.is_empty());
        let reasoning_text = choice
            .and_then(|c| {
                c.message
                    .extra
                    .get("reasoning")
                    .and_then(|v| v.as_str())
                    .or_else(|| {
                        c.message
                            .extra
                            .get("reasoning_content")
                            .and_then(|v| v.as_str())
                    })
            })
            .unwrap_or("");

        if !reasoning_text.is_empty() {
            tracing::warn!(
                trace_id = %stream_trace_id,
                model = %response.model,
                reasoning_len = reasoning_text.len(),
                has_tool_calls,
                finish_reason = ?choice.and_then(|c| c.finish_reason.as_deref()),
                "Buffered provider response contains reasoning content, but synthesized SSE currently emits only content/tool_calls chunks"
            );
        }

        for chunk in streaming_chunks_from_response(&response) {
            yield Ok::<_, Infallible>(Event::default().data(chunk.to_string()));
        }
        yield Ok(Event::default().data("[DONE]"));
    };

    state.metrics.complete_request(start.elapsed().as_millis() as u64);

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn streaming_chunks_from_response(response: &OpenAIResponse) -> Vec<serde_json::Value> {
    let choice = response.choices.first();

    let content = choice
        .map(|c| {
            match &c.message.content {
                serde_json::Value::String(s) => serde_json::Value::String(s.clone()),
                serde_json::Value::Null => serde_json::Value::Null,
                other => other.clone(),
            }
        })
        .unwrap_or(serde_json::Value::Null);

    let tool_calls = choice
        .and_then(|c| c.message.extra.get("tool_calls"))
        .and_then(|v| v.as_array())
        .cloned();

    let has_tool_calls = tool_calls.as_ref().is_some_and(|tc| !tc.is_empty());
    let reasoning_delta = reasoning_delta(choice);
    let mut chunks = Vec::new();

    if has_tool_calls {
        let tcs = tool_calls.as_ref().unwrap();
        let first_tc = &tcs[0];
        let tc_id = first_tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let tc_type = first_tc.get("type").and_then(|v| v.as_str()).unwrap_or("function");
        let fn_name = first_tc.get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or("");

        chunks.push(build_chunk_payload(
            response,
            serde_json::json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "index": 0,
                    "id": tc_id,
                    "type": tc_type,
                    "function": {
                        "name": fn_name,
                        "arguments": ""
                    }
                }]
            }),
            None,
        ));

        if let Some(delta) = reasoning_delta.clone() {
            chunks.push(build_chunk_payload(response, delta, None));
        }

        let fn_args = first_tc.get("function")
            .and_then(|f| f.get("arguments"))
            .and_then(|a| a.as_str())
            .unwrap_or("{}");
        chunks.push(build_chunk_payload(
            response,
            serde_json::json!({
                "tool_calls": [{
                    "index": 0,
                    "function": { "arguments": fn_args }
                }]
            }),
            None,
        ));

        for (i, tc) in tcs.iter().enumerate().skip(1) {
            let tc_id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let tc_type = tc.get("type").and_then(|v| v.as_str()).unwrap_or("function");
            let fn_name = tc.get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("");
            let fn_args = tc.get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|a| a.as_str())
                .unwrap_or("{}");

            chunks.push(build_chunk_payload(
                response,
                serde_json::json!({
                    "tool_calls": [{
                        "index": i,
                        "id": tc_id,
                        "type": tc_type,
                        "function": {
                            "name": fn_name,
                            "arguments": ""
                        }
                    }]
                }),
                None,
            ));

            chunks.push(build_chunk_payload(
                response,
                serde_json::json!({
                    "tool_calls": [{
                        "index": i,
                        "function": { "arguments": fn_args }
                    }]
                }),
                None,
            ));
        }
    } else {
        chunks.push(build_chunk_payload(
            response,
            serde_json::json!({ "role": "assistant", "content": "" }),
            None,
        ));

        if let Some(delta) = reasoning_delta {
            chunks.push(build_chunk_payload(response, delta, None));
        }

        if !content.is_null() && content.as_str().map(|s| !s.is_empty()).unwrap_or(true) {
            chunks.push(build_chunk_payload(
                response,
                serde_json::json!({ "content": content }),
                None,
            ));
        }
    }

    let finish_reason = if has_tool_calls {
        "tool_calls"
    } else {
        choice
            .and_then(|c| c.finish_reason.as_deref())
            .unwrap_or("stop")
    };
    chunks.push(serde_json::json!({
        "id": response.id,
        "object": "chat.completion.chunk",
        "created": response.created,
        "model": response.model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason
        }],
        "usage": response.usage
    }));

    chunks
}

fn build_chunk_payload(
    response: &OpenAIResponse,
    delta: serde_json::Value,
    finish_reason: Option<&str>,
) -> serde_json::Value {
    serde_json::json!({
        "id": response.id,
        "object": "chat.completion.chunk",
        "created": response.created,
        "model": response.model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason
        }]
    })
}

fn reasoning_delta(choice: Option<&Choice>) -> Option<serde_json::Value> {
    let choice = choice?;

    for field in ["reasoning", "reasoning_content"] {
        let Some(value) = choice.message.extra.get(field) else {
            continue;
        };
        if value.is_null() || value.as_str().is_some_and(|s| s.is_empty()) {
            continue;
        }

        let mut delta = serde_json::Map::new();
        delta.insert(field.to_string(), value.clone());
        return Some(serde_json::Value::Object(delta));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::streaming_chunks_from_response;
    use crate::models::openai::{Choice, Message, OpenAIResponse, Usage};

    fn base_response(message: Message) -> OpenAIResponse {
        OpenAIResponse {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion".to_string(),
            created: 123,
            model: "test-model".to_string(),
            choices: vec![Choice {
                index: 0,
                message,
                finish_reason: Some("stop".to_string()),
                extra: Default::default(),
            }],
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 2,
                total_tokens: 3,
                extra: Default::default(),
            },
            extra: Default::default(),
        }
    }

    #[test]
    fn streaming_chunks_include_reasoning_before_content() {
        let mut extra = serde_json::Map::new();
        extra.insert("reasoning".to_string(), serde_json::json!("thinking step"));

        let response = base_response(Message {
            role: "assistant".to_string(),
            content: serde_json::json!("final answer"),
            extra,
        });

        let chunks = streaming_chunks_from_response(&response);

        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(chunks[1]["choices"][0]["delta"]["reasoning"], "thinking step");
        assert_eq!(chunks[2]["choices"][0]["delta"]["content"], "final answer");
    }

    #[test]
    fn streaming_chunks_preserve_reasoning_content_field_name() {
        let mut extra = serde_json::Map::new();
        extra.insert("reasoning_content".to_string(), serde_json::json!("hidden chain"));

        let response = base_response(Message {
            role: "assistant".to_string(),
            content: serde_json::json!("visible answer"),
            extra,
        });

        let chunks = streaming_chunks_from_response(&response);

        assert_eq!(chunks[1]["choices"][0]["delta"]["reasoning_content"], "hidden chain");
        assert_eq!(chunks[2]["choices"][0]["delta"]["content"], "visible answer");
    }
}

// ---------------------------------------------------------------------------
// POST /v1/completions  (Req 2.2)
// ---------------------------------------------------------------------------

/// Legacy completions endpoint — pass-through proxy.
pub async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<OpenAIRequest>,
) -> Response {
    // Reuse chat completions routing; the OpenAI completions format is close enough
    // for provider pass-through. Full translation can be refined later.
    let trace_id = trace_id_from_headers(&headers);
    chat_completions_non_stream(state, request, trace_id).await
}

// ---------------------------------------------------------------------------
// POST /v1/embeddings  (Req 2.3)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
}

/// Embeddings endpoint — pass-through proxy to configured provider.
pub async fn embeddings(
    State(_state): State<AppState>,
    Json(_request): Json<EmbeddingRequest>,
) -> Response {
    // Pass-through proxy placeholder — will forward to the provider that owns the model.
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": { "message": "Embeddings endpoint: pass-through not yet wired to provider client", "type": "not_implemented" }
        })),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// POST /v1/images/generations  (Req 2.4)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ImageGenerationRequest {
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default = "default_image_count")]
    pub n: u32,
    pub size: Option<String>,
}

fn default_image_count() -> u32 {
    1
}

pub async fn image_generations(
    State(_state): State<AppState>,
    Json(_request): Json<ImageGenerationRequest>,
) -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": { "message": "Images endpoint: pass-through not yet wired to provider client", "type": "not_implemented" }
        })),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// POST /v1/audio/transcriptions  (Req 2.5)
// POST /v1/audio/translations    (Req 2.5)
// ---------------------------------------------------------------------------

pub async fn audio_transcriptions(headers: HeaderMap) -> Response {
    let _ = headers; // multipart handling deferred to provider pass-through
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": { "message": "Audio transcriptions endpoint: pass-through not yet wired", "type": "not_implemented" }
        })),
    )
        .into_response()
}

pub async fn audio_translations(headers: HeaderMap) -> Response {
    let _ = headers;
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": { "message": "Audio translations endpoint: pass-through not yet wired", "type": "not_implemented" }
        })),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// GET /v1/models  (Req 2.6, 2.12, 24.1-24.7)
// ---------------------------------------------------------------------------

/// Models list response in OpenAI format.
#[derive(Debug, Serialize)]
pub struct ModelsListResponse {
    pub object: String,
    pub data: Vec<Model>,
}

/// Aggregated models endpoint — queries all configured providers.
pub async fn list_models(State(state): State<AppState>) -> Response {
    let config = state.config.read().await;
    let mut all_models: Vec<Model> = Vec::new();
    let mut seen_ids = std::collections::HashSet::new();

    // List model group names first so clients can target groups directly
    for group in &config.model_groups {
        if seen_ids.insert(group.name.clone()) {
            all_models.push(Model {
                id: group.name.clone(),
                object: "model".to_string(),
                owned_by: "gateway".to_string(),
                created: None,
                context_window: None,
                max_completion_tokens: None,
            });
        }
    }

    // Also list individual model names for backward compatibility
    for group in &config.model_groups {
        for pm in &group.models {
            if seen_ids.insert(pm.model.clone()) {
                all_models.push(Model {
                    id: pm.model.clone(),
                    object: "model".to_string(),
                    owned_by: pm.provider.clone(),
                    created: None,
                    context_window: None,
                    max_completion_tokens: None,
                });
            }
        }
    }

    // Include manually specified models from provider configs
    for provider in &config.providers {
        for model_id in &provider.manual_models {
            if seen_ids.insert(model_id.clone()) {
                all_models.push(Model {
                    id: model_id.clone(),
                    object: "model".to_string(),
                    owned_by: provider.name.clone(),
                    created: None,
                    context_window: None,
                    max_completion_tokens: None,
                });
            }
        }
    }

    let response = ModelsListResponse {
        object: "list".to_string(),
        data: all_models,
    };

    Json(response).into_response()
}

// ---------------------------------------------------------------------------
// Assistants / Threads / Runs / Files / Fine-tuning  (Req 2.7-2.11)
// Pass-through stubs — these forward to the upstream provider once wired.
// ---------------------------------------------------------------------------

/// Generic pass-through stub returning 501 for unimplemented endpoints.
async fn not_implemented_stub(endpoint: &str) -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": {
                "message": format!("{} endpoint: pass-through not yet wired to provider client", endpoint),
                "type": "not_implemented"
            }
        })),
    )
        .into_response()
}

// --- Assistants (Req 2.7) ---
pub async fn create_assistant(State(_s): State<AppState>, body: String) -> Response {
    let _ = body;
    not_implemented_stub("Assistants").await
}
pub async fn list_assistants(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Assistants").await
}
pub async fn get_assistant(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Assistants").await
}
pub async fn modify_assistant(State(_s): State<AppState>, body: String) -> Response {
    let _ = body;
    not_implemented_stub("Assistants").await
}
pub async fn delete_assistant(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Assistants").await
}

// --- Threads (Req 2.8) ---
pub async fn create_thread(State(_s): State<AppState>, body: String) -> Response {
    let _ = body;
    not_implemented_stub("Threads").await
}
pub async fn get_thread(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Threads").await
}
pub async fn modify_thread(State(_s): State<AppState>, body: String) -> Response {
    let _ = body;
    not_implemented_stub("Threads").await
}
pub async fn delete_thread(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Threads").await
}

// --- Runs (Req 2.9) ---
pub async fn create_run(State(_s): State<AppState>, body: String) -> Response {
    let _ = body;
    not_implemented_stub("Runs").await
}
pub async fn list_runs(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Runs").await
}
pub async fn get_run(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Runs").await
}
pub async fn cancel_run(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Runs").await
}

// --- Messages on threads ---
pub async fn create_message(State(_s): State<AppState>, body: String) -> Response {
    let _ = body;
    not_implemented_stub("Messages").await
}
pub async fn list_messages(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Messages").await
}

// --- Files (Req 2.10) ---
pub async fn upload_file(headers: HeaderMap) -> Response {
    let _ = headers;
    not_implemented_stub("Files").await
}
pub async fn list_files(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Files").await
}
pub async fn get_file(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Files").await
}
pub async fn delete_file(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Files").await
}
pub async fn get_file_content(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Files").await
}

// --- Fine-tuning (Req 2.11) ---
pub async fn create_fine_tuning_job(State(_s): State<AppState>, body: String) -> Response {
    let _ = body;
    not_implemented_stub("Fine-tuning").await
}
pub async fn list_fine_tuning_jobs(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Fine-tuning").await
}
pub async fn get_fine_tuning_job(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Fine-tuning").await
}
pub async fn cancel_fine_tuning_job(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Fine-tuning").await
}
pub async fn list_fine_tuning_events(State(_s): State<AppState>) -> Response {
    not_implemented_stub("Fine-tuning").await
}

// ---------------------------------------------------------------------------
// GET /metrics  (Req 20.7-20.11) — Prometheus exposition format
// ---------------------------------------------------------------------------

/// Prometheus metrics endpoint — returns metrics in Prometheus text exposition format.
/// No external prometheus client library; we format the text directly from MetricsSnapshot.
pub async fn prometheus_metrics(State(state): State<AppState>) -> Response {
    let snap = state.metrics.snapshot();
    let mut out = String::with_capacity(2048);

    // Helper: append a metric block
    macro_rules! metric {
        (counter $name:expr, $help:expr, $val:expr) => {
            out.push_str(&format!(
                "# HELP {} {}\n# TYPE {} counter\n{} {}\n",
                $name, $help, $name, $name, $val
            ));
        };
        (gauge $name:expr, $help:expr, $val:expr) => {
            out.push_str(&format!(
                "# HELP {} {}\n# TYPE {} gauge\n{} {}\n",
                $name, $help, $name, $name, $val
            ));
        };
    }

    // Req 20.8: request count
    metric!(counter "obey_api_requests_total", "Total number of requests", snap.request_count);

    // Req 20.8: active requests
    metric!(gauge "obey_api_active_requests", "Current active requests", snap.active_requests);

    // Req 20.9: response time (avg as gauge — histogram buckets would need raw data)
    metric!(gauge "obey_api_response_time_avg_ms", "Average response time in milliseconds", snap.avg_response_time_ms);

    // Request rate
    metric!(gauge "obey_api_request_rate_per_min", "Requests per minute", snap.request_rate_per_min);

    // Cumulative cost
    metric!(gauge "obey_api_cumulative_cost_dollars", "Cumulative cost in dollars", snap.cumulative_cost);

    // Req 20.8: per-provider request counts
    if !snap.provider_health.is_empty() {
        out.push_str("# HELP obey_api_provider_requests_total Total requests by provider\n");
        out.push_str("# TYPE obey_api_provider_requests_total counter\n");
        for ph in &snap.provider_health {
            out.push_str(&format!(
                "obey_api_provider_requests_total{{provider=\"{}\"}} {}\n",
                ph.provider, ph.total_requests
            ));
        }

        out.push_str("# HELP obey_api_provider_success_total Successful requests by provider\n");
        out.push_str("# TYPE obey_api_provider_success_total counter\n");
        for ph in &snap.provider_health {
            out.push_str(&format!(
                "obey_api_provider_success_total{{provider=\"{}\"}} {}\n",
                ph.provider, ph.successful_requests
            ));
        }

        out.push_str("# HELP obey_api_provider_failures_total Failed requests by provider\n");
        out.push_str("# TYPE obey_api_provider_failures_total counter\n");
        for ph in &snap.provider_health {
            out.push_str(&format!(
                "obey_api_provider_failures_total{{provider=\"{}\"}} {}\n",
                ph.provider, ph.failed_requests
            ));
        }

        // Req 20.9: per-provider avg response time (histogram proxy)
        out.push_str("# HELP obey_api_provider_response_time_avg_ms Average response time by provider in milliseconds\n");
        out.push_str("# TYPE obey_api_provider_response_time_avg_ms gauge\n");
        for ph in &snap.provider_health {
            out.push_str(&format!(
                "obey_api_provider_response_time_avg_ms{{provider=\"{}\"}} {}\n",
                ph.provider, ph.avg_response_time_ms
            ));
        }
    }

    // Req 20.10: circuit breaker state gauges
    let cb_states = state.router.get_circuit_breaker_states().await;
    if !cb_states.is_empty() {
        out.push_str("# HELP obey_api_circuit_breaker_state Circuit breaker state (0=closed, 1=open, 2=half_open)\n");
        out.push_str("# TYPE obey_api_circuit_breaker_state gauge\n");
        for (provider, state_label) in &cb_states {
            let val = match state_label.as_str() {
                "closed" => 0,
                "open" => 1,
                "half_open" => 2,
                _ => 0,
            };
            out.push_str(&format!(
                "obey_api_circuit_breaker_state{{provider=\"{}\",state=\"{}\"}} {}\n",
                provider, state_label, val
            ));
        }
    }

    // Req 20.11: cache hit rate gauge
    if let Some(rate) = snap.cache_hit_rate {
        metric!(gauge "obey_api_cache_hit_rate", "Cache hit rate (0.0 to 1.0)", rate);
    }

    // Cost by provider
    if !snap.cost_by_provider.is_empty() {
        out.push_str("# HELP obey_api_cost_by_provider_dollars Cumulative cost by provider in dollars\n");
        out.push_str("# TYPE obey_api_cost_by_provider_dollars gauge\n");
        for (provider, cost) in &snap.cost_by_provider {
            out.push_str(&format!(
                "obey_api_cost_by_provider_dollars{{provider=\"{}\"}} {}\n",
                provider, cost
            ));
        }
    }

    if !snap.retry_count_by_provider.is_empty() {
        out.push_str("# HELP obey_api_provider_retries_total Total retry attempts by provider\n");
        out.push_str("# TYPE obey_api_provider_retries_total counter\n");
        for (provider, retry_count) in &snap.retry_count_by_provider {
            out.push_str(&format!(
                "obey_api_provider_retries_total{{provider=\"{}\"}} {}\n",
                provider, retry_count
            ));
        }
    }

    if !snap.retry_delay_ms_by_provider.is_empty() {
        out.push_str("# HELP obey_api_provider_retry_delay_ms_total Total retry delay applied by provider in milliseconds\n");
        out.push_str("# TYPE obey_api_provider_retry_delay_ms_total counter\n");
        for (provider, retry_delay_ms) in &snap.retry_delay_ms_by_provider {
            out.push_str(&format!(
                "obey_api_provider_retry_delay_ms_total{{provider=\"{}\"}} {}\n",
                provider, retry_delay_ms
            ));
        }
    }

    if !snap.budget_limit_by_provider.is_empty() {
        out.push_str("# HELP obey_api_provider_budget_limit_dollars Configured budget limit by provider in dollars\n");
        out.push_str("# TYPE obey_api_provider_budget_limit_dollars gauge\n");
        for (provider, budget_limit) in &snap.budget_limit_by_provider {
            out.push_str(&format!(
                "obey_api_provider_budget_limit_dollars{{provider=\"{}\"}} {}\n",
                provider, budget_limit
            ));
        }
    }

    if !snap.budget_exhaustions_by_provider.is_empty() {
        out.push_str("# HELP obey_api_provider_budget_exhaustions_total Total provider budget exhaustion events\n");
        out.push_str("# TYPE obey_api_provider_budget_exhaustions_total counter\n");
        for (provider, budget_exhaustions) in &snap.budget_exhaustions_by_provider {
            out.push_str(&format!(
                "obey_api_provider_budget_exhaustions_total{{provider=\"{}\"}} {}\n",
                provider, budget_exhaustions
            ));
        }
    }

    if !snap.unknown_cost_by_provider.is_empty() {
        out.push_str("# HELP obey_api_provider_unknown_cost_total Total successful responses without usable usage data by provider\n");
        out.push_str("# TYPE obey_api_provider_unknown_cost_total counter\n");
        for (provider, unknown_cost) in &snap.unknown_cost_by_provider {
            out.push_str(&format!(
                "obey_api_provider_unknown_cost_total{{provider=\"{}\"}} {}\n",
                provider, unknown_cost
            ));
        }
    }

    if !snap.rate_limit_exhaustions_by_provider.is_empty() {
        out.push_str("# HELP obey_api_provider_rate_limit_exhaustions_total Total provider skips caused by local rate-limit exhaustion\n");
        out.push_str("# TYPE obey_api_provider_rate_limit_exhaustions_total counter\n");
        for (provider, rate_limit_exhaustions) in &snap.rate_limit_exhaustions_by_provider {
            out.push_str(&format!(
                "obey_api_provider_rate_limit_exhaustions_total{{provider=\"{}\"}} {}\n",
                provider, rate_limit_exhaustions
            ));
        }
    }

    (
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        out,
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// POST /admin/config/reload  (Req 26.1-26.7)
// ---------------------------------------------------------------------------

/// Reload configuration from disk without restarting the gateway.
///
/// On success the new config is applied to future requests, circuit breaker
/// states are reset, and the models list cache is invalidated.
/// On validation failure the existing config is kept and an error is returned.
pub async fn reload_config(State(state): State<AppState>) -> Response {
    let config_path = state.config_path.as_ref();

    // Read & validate new config from disk (Req 26.1, 26.2)
    let new_config = match load_and_validate_config(config_path) {
        Ok(cfg) => cfg,
        Err(e) => {
            // Req 26.3: keep existing config, return error
            tracing::warn!("Config reload validation failed: {}", e);
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Configuration validation failed: {}", e),
                        "type": "configuration_error"
                    }
                })),
            )
                .into_response();
        }
    };

    // Apply new config (Req 26.4)
    apply_runtime_config_update(&state, new_config).await;

    // Req 26.6: models list cache is implicitly cleared because list_models
    // reads from the config on every call.

    tracing::info!("Configuration reloaded successfully from {}", config_path.display());

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "ok",
            "message": "Configuration reloaded successfully"
        })),
    )
        .into_response()
}
