use async_trait::async_trait;
use futures::Stream;
use std::collections::HashMap;
use std::pin::Pin;

use crate::error::GatewayError;
use crate::models::openai::OpenAIRequest;
use crate::providers::{Model, ProviderClient, ProviderResponse, SSEEvent};
use super::openai_compatible::OpenAICompatibleProvider;

/// NVIDIA NIM provider client
/// Uses OpenAI-compatible API format
pub struct NvidiaNIMProvider {
    inner: OpenAICompatibleProvider,
}

impl NvidiaNIMProvider {
    /// Create a new NVIDIA NIM provider client
    /// NVIDIA NIM API endpoint: https://integrate.api.nvidia.com/v1
    pub fn new(name: String, api_key: String, max_connections: Option<u32>, timeout_seconds: Option<u64>, custom_headers: HashMap<String, String>) -> Result<Self, GatewayError> {
        let inner = OpenAICompatibleProvider::new(
            name,
            "https://integrate.api.nvidia.com/v1".to_string(),
            api_key,
            max_connections,
            timeout_seconds,
            custom_headers,
        )?;

        Ok(Self { inner })
    }

    /// Create a new NVIDIA NIM provider with custom base URL
    pub fn new_with_base_url(
        name: String,
        base_url: String,
        api_key: String,
        max_connections: Option<u32>,
        timeout_seconds: Option<u64>,
        custom_headers: HashMap<String, String>,
    ) -> Result<Self, GatewayError> {
        let inner = OpenAICompatibleProvider::new(name, base_url, api_key, max_connections, timeout_seconds, custom_headers)?;
        Ok(Self { inner })
    }
}

#[async_trait]
impl ProviderClient for NvidiaNIMProvider {
    async fn chat_completion(
        &self,
        request: OpenAIRequest,
    ) -> Result<ProviderResponse, GatewayError> {
        self.inner.chat_completion(request).await
    }

    async fn chat_completion_stream(
        &self,
        request: OpenAIRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<SSEEvent, GatewayError>> + Send>>, GatewayError> {
        self.inner.chat_completion_stream(request).await
    }

    async fn list_models(&self) -> Result<Vec<Model>, GatewayError> {
        self.inner.list_models().await
    }

    fn provider_name(&self) -> &str {
        self.inner.provider_name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_nvidia_nim_provider_creation() {
        let provider = NvidiaNIMProvider::new(
            "nvidia-nim".to_string(),
            "test-api-key".to_string(),
            None,
            None,
            HashMap::new(),
        );

        assert!(provider.is_ok());
        let provider = provider.unwrap();
        assert_eq!(provider.provider_name(), "nvidia-nim");
    }

    #[test]
    fn test_nvidia_nim_provider_with_custom_base_url() {
        let provider = NvidiaNIMProvider::new_with_base_url(
            "nvidia-custom".to_string(),
            "https://custom.nvidia.com/v1".to_string(),
            "test-key".to_string(),
            None,
            None,
            HashMap::new(),
        );

        assert!(provider.is_ok());
        assert_eq!(provider.unwrap().provider_name(), "nvidia-custom");
    }
}
