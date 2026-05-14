//! Codex backend translation pipeline.
//!
//! This module implements the Chat Completions ↔ Responses API translation
//! layer that allows OAuth-authenticated OpenAI providers to dispatch through
//! `https://chatgpt.com/backend-api/codex/responses`.
//!
//! Submodules are populated incrementally by subsequent tasks in the
//! `codex-backend-translation` spec. Public re-exports are added to this file
//! as each submodule's contents become available.

pub mod client;
pub mod effort_map;
pub mod errors;
pub mod instructions;
pub mod jwt;
pub mod model_map;
pub mod models_discovery;
pub mod sse;
pub mod translate_request;
pub mod translate_response;

pub use crate::codex::errors::CodexError;
pub use crate::codex::instructions::InstructionsStore;
pub use crate::codex::client::CodexProviderClient;
pub use crate::codex::jwt::extract_chatgpt_account_id;
pub use crate::codex::model_map::{
    is_reasoning, is_xhigh, resolve_model, REASONING_MODEL_PATTERNS, XHIGH_MODEL_PATTERNS,
};
pub use crate::codex::effort_map::map_effort;
pub use crate::codex::models_discovery::ModelsDiscovery;
pub use crate::codex::sse::{
    CompletedPayload, ErrorPayload, IncompleteDetails, OutputItem, OutputItemContent,
    ResponsesEvent, SseLineParser, Usage,
};
pub use crate::codex::translate_request::ChatToResponsesTranslator;
pub use crate::codex::translate_response::ResponseAccumulator;
