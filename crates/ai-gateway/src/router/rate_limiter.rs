use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

#[cfg(test)]
use std::time::Duration;

#[cfg(test)]
use proptest::prelude::*;

/// Internal state for the rate limiter, protected by a single mutex
/// to avoid potential deadlocks from acquiring multiple locks
#[derive(Debug)]
struct RateLimiterState {
    /// Current number of available tokens
    tokens: f64,
    /// Last refill timestamp
    last_refill: Instant,
}

/// Token bucket rate limiter for per-provider rate limiting
#[derive(Debug)]
pub struct RateLimiter {
    /// Maximum number of tokens (requests per minute)
    capacity: u32,
    /// Requests per minute limit
    requests_per_minute: u32,
    /// Combined state protected by a single mutex to prevent deadlocks
    state: Arc<Mutex<RateLimiterState>>,
}

impl RateLimiter {
    /// Create a new rate limiter with specified requests per minute
    /// 
    /// # Arguments
    /// * `requests_per_minute` - Maximum requests allowed per minute (0 = unlimited)
    pub fn new(requests_per_minute: u32) -> Self {
        Self {
            capacity: requests_per_minute,
            requests_per_minute,
            state: Arc::new(Mutex::new(RateLimiterState {
                tokens: requests_per_minute as f64,
                last_refill: Instant::now(),
            })),
        }
    }

    /// Check if a request can be made without consuming a token
    /// 
    /// Returns true if tokens are available, false otherwise
    pub async fn check_available(&self) -> bool {
        // Unlimited rate limit
        if self.requests_per_minute == 0 {
            return true;
        }

        let mut state = self.state.lock().await;
        self.refill_tokens_internal(&mut state);
        state.tokens >= 1.0
    }

    /// Consume a token for a request
    /// 
    /// Returns true if token was consumed, false if no tokens available
    pub async fn consume(&self) -> bool {
        // Unlimited rate limit
        if self.requests_per_minute == 0 {
            return true;
        }

        let mut state = self.state.lock().await;
        self.refill_tokens_internal(&mut state);
        
        if state.tokens >= 1.0 {
            state.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Refill tokens based on elapsed time since last refill
    /// Internal method that operates on already-locked state
    fn refill_tokens_internal(&self, state: &mut RateLimiterState) {
        let now = Instant::now();
        let elapsed = now.duration_since(state.last_refill);
        
        // Calculate tokens to add based on elapsed time
        // tokens_per_second = requests_per_minute / 60
        let tokens_per_second = self.requests_per_minute as f64 / 60.0;
        let tokens_to_add = elapsed.as_secs_f64() * tokens_per_second;
        
        if tokens_to_add > 0.0 {
            state.tokens = (state.tokens + tokens_to_add).min(self.capacity as f64);
            state.last_refill = now;
        }
    }

    /// Get current token count (for testing/monitoring)
    pub async fn get_tokens(&self) -> f64 {
        let mut state = self.state.lock().await;
        self.refill_tokens_internal(&mut state);
        state.tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_unlimited_rate_limit() {
        let limiter = RateLimiter::new(0);
        
        // Should always allow requests
        for _ in 0..100 {
            assert!(limiter.check_available().await);
            assert!(limiter.consume().await);
        }
    }

    #[tokio::test]
    async fn test_rate_limit_enforcement() {
        let limiter = RateLimiter::new(60); // 60 requests per minute
        
        // Should allow up to capacity
        for _ in 0..60 {
            assert!(limiter.consume().await);
        }
        
        // Should reject when exhausted
        assert!(!limiter.check_available().await);
        assert!(!limiter.consume().await);
    }

    #[tokio::test]
    async fn test_token_refill() {
        let limiter = RateLimiter::new(60); // 60 requests per minute = 1 per second
        
        // Consume all tokens
        for _ in 0..60 {
            assert!(limiter.consume().await);
        }
        
        assert!(!limiter.check_available().await);
        
        // Wait for 1 second to refill 1 token
        sleep(Duration::from_millis(1100)).await;
        
        assert!(limiter.check_available().await);
        assert!(limiter.consume().await);
        
        // Should be exhausted again
        assert!(!limiter.check_available().await);
    }

    #[tokio::test]
    async fn test_check_available_does_not_consume() {
        let limiter = RateLimiter::new(60);
        
        // Consume all but one token
        for _ in 0..59 {
            assert!(limiter.consume().await);
        }
        
        // Check multiple times without consuming
        assert!(limiter.check_available().await);
        assert!(limiter.check_available().await);
        assert!(limiter.check_available().await);
        
        // Should still have 1 token available
        assert!(limiter.consume().await);
        
        // Now should be exhausted
        assert!(!limiter.check_available().await);
    }

    #[tokio::test]
    async fn test_token_refill_caps_at_capacity() {
        let limiter = RateLimiter::new(10);
        
        // Consume 5 tokens
        for _ in 0..5 {
            assert!(limiter.consume().await);
        }
        
        // Wait long enough to refill more than capacity
        sleep(Duration::from_secs(2)).await;
        
        // Should have capacity tokens, not more
        let tokens = limiter.get_tokens().await;
        assert!(tokens <= 10.0, "Tokens should be capped at capacity: {}", tokens);
    }

    #[tokio::test]
    async fn test_fractional_token_accumulation() {
        let limiter = RateLimiter::new(60); // 1 token per second
        
        // Consume all tokens
        for _ in 0..60 {
            assert!(limiter.consume().await);
        }
        
        // Wait for 0.5 seconds (should accumulate 0.5 tokens)
        sleep(Duration::from_millis(500)).await;
        assert!(!limiter.check_available().await); // Not enough for 1 request
        
        // Wait another 0.6 seconds (total 1.1 tokens)
        sleep(Duration::from_millis(600)).await;
        assert!(limiter.check_available().await); // Now have >= 1 token
        assert!(limiter.consume().await);
    }

    // Feature: ai-gateway, Property 40: Rate Limit Enforcement
    // **Validates: Requirements 44.2, 44.3**
    #[cfg(test)]
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig {
                cases: 20,
                max_shrink_iters: 100,
                .. ProptestConfig::default()
            })]

            #[test]
            fn prop_rate_limit_enforced_in_60s_window(
                rate_limit in 10u32..100u32,
                burst_size in 1usize..20usize,
            ) {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    let limiter = RateLimiter::new(rate_limit);
                    let start = std::time::Instant::now();
                    let mut successful_requests = 0u32;

                    // Attempt burst_size requests immediately
                    for _ in 0..burst_size.min(rate_limit as usize) {
                        if limiter.consume().await {
                            successful_requests += 1;
                        }
                    }

                    // Verify we don't exceed rate limit in initial burst
                    prop_assert!(successful_requests <= rate_limit);

                    // Wait 1 second and try more requests
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    
                    let expected_refill = (rate_limit as f64 / 60.0).ceil() as u32;
                    let mut second_batch = 0u32;
                    
                    for _ in 0..expected_refill + 5 {
                        if limiter.consume().await {
                            second_batch += 1;
                        }
                    }

                    // Total requests in ~1 second should not exceed rate_limit + expected_refill
                    let total = successful_requests + second_batch;
                    let elapsed = start.elapsed().as_secs_f64();
                    let max_allowed = (rate_limit as f64 * elapsed / 60.0).ceil() as u32 + rate_limit;
                    
                    prop_assert!(total <= max_allowed, 
                        "Rate limit violated: {} requests in {:.2}s (limit: {} req/min, max_allowed: {})",
                        total, elapsed, rate_limit, max_allowed);

                    Ok(())
                })?;
            }
        }
    }
}
