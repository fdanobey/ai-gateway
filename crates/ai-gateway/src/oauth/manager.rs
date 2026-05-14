//! OAuth session manager — wraps flow + token store + background refresh task.
//!
//! Owns the runtime-shared [`OAuthSessionState`] and the on-disk
//! [`OAuthTokenStore`], and optionally runs a tokio background task that
//! keeps the access token fresh.
//!
//! # Responsibilities (tasks 5.6 – 5.8)
//!
//! * **5.6** — `OAuthManager` wrapping the [`OAuthFlow`] orchestrator, the
//!   [`OAuthTokenStore`], and the shared session state.
//! * **5.7** — Background token-refresh task: ticks every 60 s and refreshes
//!   any token that is within 5 min of expiry (Req 5.2).
//! * **5.8** — Exponential backoff on transient refresh failures (Req 5.5):
//!   `2 s, 4 s, 8 s` (three attempts) before marking the session as
//!   [`OAuthSessionState::Expired`]. `401/403` responses short-circuit the
//!   retry loop and expire the session immediately (Req 5.4).
//!
//! # Security invariants
//!
//! * Token material is never logged (Req 8.4). All refresh-token and
//!   access-token handling flows through `#[tracing::instrument(skip(...))]`
//!   annotations or occurs inside fields of types whose `Debug` impls
//!   redact them ([`StoredTokens`], [`crate::oauth::token::TokenResponse`]).
//! * `session_state` is mirrored into an `Arc<RwLock<_>>` so admin endpoints
//!   (task 7.x) can render a read-only snapshot without ever touching the
//!   on-disk ciphertext or the in-memory plaintext tokens.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::sync::{Mutex, RwLock};
use tokio::task::JoinHandle;

use super::error::OAuthError;
use super::flow::{OAuthFlow, OAuthSessionState};
use super::store::{OAuthTokenStore, StoredTokens};
use super::token::{refresh_access_token, TokenResponse};

/// Background-refresh tick cadence (Req 5.2: "check at a regular interval").
const REFRESH_TICK_INTERVAL: Duration = Duration::from_secs(60);

/// Pre-expiry refresh window (Req 5.2: "within 5 minutes of expiry").
const REFRESH_WINDOW_SECS: u64 = 300;

/// Number of refresh attempts before marking the session as expired
/// (Req 5.5: "up to 3 attempts").
const BACKOFF_MAX_ATTEMPTS: u32 = 3;

/// Exponential-backoff delay before refresh attempt `attempt`
/// (Req 5.5 / Property 8: `2^(attempt + 1)` seconds).
///
/// `attempt = 0 → 2 s`, `1 → 4 s`, `2 → 8 s`. Exposed `pub(crate)` so the
/// property test in task 5.9 can pin the formula without re-deriving it.
pub(crate) fn backoff_delay_secs(attempt: u32) -> u64 {
    2u64.pow(attempt + 1)
}

/// Top-level OAuth session manager.
///
/// Shared across the router via `AppState` (task 7.1). Cheap to clone as an
/// `Arc<OAuthManager>`; all mutable state is behind `RwLock` / `Mutex`.
pub struct OAuthManager {
    store: Arc<OAuthTokenStore>,
    session_state: Arc<RwLock<OAuthSessionState>>,
    http_client: reqwest::Client,
    #[allow(dead_code)] // consumed by admin login handler (task 7.2)
    flow: Arc<Mutex<OAuthFlow>>,
    refresh_task: Mutex<Option<JoinHandle<()>>>,
}

impl OAuthManager {
    /// Construct a manager around an existing store and HTTP client.
    ///
    /// The initial [`OAuthSessionState`] is [`OAuthSessionState::Unauthenticated`];
    /// call [`Self::load_existing_session`] after construction to reconcile it
    /// with whatever is already on disk.
    pub fn new(store: OAuthTokenStore, http_client: reqwest::Client) -> Self {
        let flow = OAuthFlow::new(http_client.clone());
        Self {
            store: Arc::new(store),
            session_state: Arc::new(RwLock::new(OAuthSessionState::Unauthenticated)),
            http_client,
            flow: Arc::new(Mutex::new(flow)),
            refresh_task: Mutex::new(None),
        }
    }

    /// Read any tokens already on disk and reconcile [`Self::session_state`].
    ///
    /// Mapping:
    /// * No file / unreadable blob → `Unauthenticated` (Req 4.4).
    /// * Present but already expired → `Expired`.
    /// * Present and still valid → `Authenticated { expires_at, scopes }`.
    pub async fn load_existing_session(&self) -> Result<(), OAuthError> {
        let tokens = self.store.load()?;
        let mut state = self.session_state.write().await;
        *state = match tokens {
            None => OAuthSessionState::Unauthenticated,
            Some(t) if t.is_expired(now_unix_secs()) => OAuthSessionState::Expired,
            Some(t) => OAuthSessionState::Authenticated {
                expires_at: t.expires_at,
                scopes: t.scopes,
            },
        };
        Ok(())
    }

    /// Snapshot the current session state for read-only consumers (admin
    /// status endpoint, router key resolution).
    pub async fn session_state(&self) -> OAuthSessionState {
        self.session_state.read().await.clone()
    }

    /// Persist a fresh [`StoredTokens`] payload and update
    /// [`Self::session_state`] to `Authenticated`.
    ///
    /// Called from the token-exchange completion path (task 5.3+) and from
    /// successful refreshes inside the background task.
    #[tracing::instrument(skip(self, tokens), fields(expires_at = tokens.expires_at))]
    pub(crate) async fn store_tokens(&self, tokens: StoredTokens) -> Result<(), OAuthError> {
        self.store.save(&tokens)?;
        let mut state = self.session_state.write().await;
        *state = OAuthSessionState::Authenticated {
            expires_at: tokens.expires_at,
            scopes: tokens.scopes,
        };
        Ok(())
    }

    /// Return the current access token when the session is
    /// [`OAuthSessionState::Authenticated`] and the on-disk blob is still
    /// decryptable; otherwise `None`.
    ///
    /// Used by the router's per-request key resolution (task 6.x) to inject
    /// the Bearer token into upstream OpenAI requests.
    pub async fn get_access_token(&self) -> Option<String> {
        // Skip the disk round-trip when the in-memory state already says the
        // session is not usable — avoids unnecessary crypto work on the hot
        // request path.
        match *self.session_state.read().await {
            OAuthSessionState::Authenticated { .. } | OAuthSessionState::Refreshing => {}
            _ => return None,
        }

        match self.store.load() {
            Ok(Some(t)) if !t.is_expired(now_unix_secs()) => Some(t.access_token),
            _ => None,
        }
    }

    /// Force an immediate token refresh (Req 6.4).
    ///
    /// Called by the router when an upstream OAuth provider returns HTTP 401,
    /// indicating the access token has expired mid-request. This triggers a
    /// refresh *before* the circuit breaker's retry path consumes the attempt,
    /// so the next retry uses a fresh token.
    ///
    /// Returns `Ok(new_access_token)` on success, or an error if the refresh
    /// fails (session will be marked `Expired` in that case).
    #[tracing::instrument(skip(self))]
    pub async fn force_refresh(&self) -> Result<String, OAuthError> {
        let tokens = self.store.load()?.ok_or(OAuthError::TokenRefreshHttp {
            status: 0,
            body: "no stored tokens to refresh".to_string(),
        })?;

        // Transition to Refreshing while we attempt the refresh.
        {
            let mut state = self.session_state.write().await;
            *state = OAuthSessionState::Refreshing;
        }

        match self.attempt_refresh_with_backoff(&tokens.refresh_token).await {
            Ok(response) => {
                let new_access_token = response.access_token.clone();
                self.apply_refreshed_tokens(&tokens, response).await?;
                Ok(new_access_token)
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "force_refresh: refresh failed; marking session expired"
                );
                let mut state = self.session_state.write().await;
                *state = OAuthSessionState::Expired;
                Err(e)
            }
        }
    }

    /// Initiate an OAuth login flow (Req 1.1, 1.6).
    ///
    /// Delegates to [`OAuthFlow::initiate`] which generates PKCE material,
    /// opens the browser, and starts the callback server. The returned
    /// [`InitiationOutcome`] tells the caller whether the browser was
    /// successfully opened or whether the auth URL must be surfaced for
    /// manual navigation.
    ///
    /// On success the flow proceeds asynchronously: the callback server
    /// awaits the redirect, exchanges the code for tokens, and persists them
    /// via [`Self::store_tokens`]. The caller does not need to poll — the
    /// session state transitions to `Authenticated` once the exchange
    /// completes.
    #[tracing::instrument(skip(self))]
    pub async fn initiate_login(&self) -> Result<super::flow::InitiationOutcome, OAuthError> {
        use super::token::exchange_code;

        let mut flow = self.flow.lock().await;

        // OpenAI's OAuth app (client_id app_EMoamEEZ73f0CkXaXp7hrann) only
        // accepts http://localhost:1455/auth/callback as a registered
        // redirect URI. Using any other port causes the auth request to
        // fail with "unknown_error".
        let callback_port = 1455u16;
        let scope = "openid profile email offline_access";
        let redirect_uri_placeholder = format!("http://localhost:{}/auth/callback", callback_port);
        let callback_timeout = Duration::from_secs(120);

        let (outcome, server_handle) = flow
            .initiate(&redirect_uri_placeholder, scope, callback_port, callback_timeout)
            .await?;

        // Capture what we need from the flow before releasing the lock.
        let code_verifier = flow
            .pkce_verifier()
            .expect("pkce_verifier set after initiate()")
            .to_string();
        let redirect_uri = flow
            .redirect_uri()
            .expect("redirect_uri set after initiate()")
            .to_string();

        drop(flow); // release the Mutex so the callback server can proceed

        // Spawn a task that awaits the callback, exchanges the code, and
        // persists the tokens. This runs in the background so the admin
        // endpoint can return immediately.
        let store = self.store.clone();
        let session_state = self.session_state.clone();
        let http_client = self.http_client.clone();

        tokio::spawn(async move {
            let auth_code = match server_handle.await {
                Ok(Ok(code)) => code,
                Ok(Err(e)) => {
                    tracing::warn!(error = %e, "OAuth callback server returned error");
                    return;
                }
                Err(e) => {
                    tracing::warn!(error = %e, "OAuth callback server task panicked");
                    return;
                }
            };

            // Exchange the authorization code for tokens.
            match exchange_code(
                &http_client,
                &auth_code.0,
                &code_verifier,
                &redirect_uri,
            )
            .await
            {
                Ok(token_response) => {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    let expires_at = now.saturating_add(token_response.expires_in);
                    let scopes = token_response
                        .scope
                        .unwrap_or_else(|| scope.to_string());
                    let tokens = StoredTokens {
                        access_token: token_response.access_token,
                        refresh_token: token_response
                            .refresh_token
                            .unwrap_or_default(),
                        expires_at,
                        scopes: scopes.clone(),
                    };
                    if let Err(e) = store.save(&tokens) {
                        tracing::error!(error = %e, "Failed to persist OAuth tokens");
                        return;
                    }
                    let mut state = session_state.write().await;
                    *state = OAuthSessionState::Authenticated {
                        expires_at,
                        scopes,
                    };
                    tracing::info!("OAuth login completed successfully");
                }
                Err(e) => {
                    tracing::warn!(error = %e, "OAuth token exchange failed");
                }
            }
        });

        Ok(outcome)
    }

    /// Log the user out: delete the on-disk token blob and mark the session
    /// [`OAuthSessionState::Unauthenticated`] (Req 7.3).
    ///
    /// Idempotent: callable when no tokens are present.
    pub async fn logout(&self) -> Result<(), OAuthError> {
        self.store.delete()?;
        let mut state = self.session_state.write().await;
        *state = OAuthSessionState::Unauthenticated;
        Ok(())
    }

    /// Spawn the background refresh task (Req 5.1 – 5.5).
    ///
    /// Must be called on an `Arc<Self>` so the spawned task can share
    /// ownership of the manager for the duration of its runtime. Calling
    /// this more than once replaces the previous handle; the previous task
    /// is **not** aborted automatically — call [`Self::stop_refresh_loop`]
    /// first if you need deterministic shutdown.
    pub fn start_refresh_loop(self: Arc<Self>) {
        let manager = self.clone();
        let handle = tokio::spawn(async move {
            manager.run_refresh_loop().await;
        });

        // Record the handle so `stop_refresh_loop` can abort it later. We
        // acquire the lock in a separate spawn to avoid blocking the caller.
        let this = self.clone();
        tokio::spawn(async move {
            let mut slot = this.refresh_task.lock().await;
            *slot = Some(handle);
        });
    }

    /// Abort the background refresh task if one is running.
    pub async fn stop_refresh_loop(&self) {
        let mut slot = self.refresh_task.lock().await;
        if let Some(handle) = slot.take() {
            handle.abort();
        }
    }

    /// Body of the background refresh task (task 5.7).
    async fn run_refresh_loop(self: Arc<Self>) {
        let mut interval = tokio::time::interval(REFRESH_TICK_INTERVAL);
        // Skip the immediate first tick; we want to wait a full cadence
        // before the first check so startup I/O doesn't race with disk init.
        interval.tick().await;

        loop {
            interval.tick().await;

            let tokens = match self.store.load() {
                Ok(Some(t)) => t,
                Ok(None) => continue,
                Err(e) => {
                    tracing::warn!(error = %e, "oauth refresh loop: failed to load tokens");
                    continue;
                }
            };

            // Nothing to do unless we are inside the pre-expiry window.
            if !tokens.needs_refresh(now_unix_secs(), REFRESH_WINDOW_SECS) {
                continue;
            }

            // Transition to Refreshing; the previous state is restored on
            // success (Authenticated) or replaced with Expired on failure.
            {
                let mut state = self.session_state.write().await;
                *state = OAuthSessionState::Refreshing;
            }

            match self.attempt_refresh_with_backoff(&tokens.refresh_token).await {
                Ok(response) => {
                    if let Err(e) = self.apply_refreshed_tokens(&tokens, response).await {
                        tracing::warn!(
                            error = %e,
                            "oauth refresh loop: failed to persist refreshed tokens"
                        );
                        let mut state = self.session_state.write().await;
                        *state = OAuthSessionState::Expired;
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "oauth refresh loop: refresh failed; marking session expired"
                    );
                    let mut state = self.session_state.write().await;
                    *state = OAuthSessionState::Expired;
                }
            }
        }
    }

    /// Refresh the access token with the (2 s, 4 s, 8 s) backoff schedule
    /// (Req 5.5 / Property 8).
    ///
    /// * Network errors trigger a retry after
    ///   [`backoff_delay_secs(attempt)`] seconds.
    /// * `401/403` ([`OAuthError::SessionExpired`]) short-circuits the loop
    ///   — the refresh token itself is dead and retrying cannot help.
    /// * Any other error (HTTP 4xx/5xx, parse failures, etc.) also
    ///   short-circuits, since none are transport-level and so are unlikely
    ///   to succeed on immediate retry.
    ///
    /// Returns the last transport error once the attempt budget is spent.
    #[tracing::instrument(skip(self, refresh_token))]
    async fn attempt_refresh_with_backoff(
        &self,
        refresh_token: &str,
    ) -> Result<TokenResponse, OAuthError> {
        let mut last_err: Option<OAuthError> = None;
        for attempt in 0..BACKOFF_MAX_ATTEMPTS {
            match refresh_access_token(&self.http_client, refresh_token).await {
                Ok(tokens) => return Ok(tokens),
                Err(e @ OAuthError::SessionExpired { .. }) => return Err(e),
                Err(e @ OAuthError::TokenRefreshNetwork(_)) => {
                    last_err = Some(e);
                    tokio::time::sleep(Duration::from_secs(backoff_delay_secs(attempt))).await;
                    continue;
                }
                Err(other) => return Err(other),
            }
        }
        Err(last_err.unwrap_or(OAuthError::TokenRefreshHttp {
            status: 0,
            body: "backoff exhausted".to_string(),
        }))
    }

    /// Merge a successful refresh response into a fresh [`StoredTokens`]
    /// payload, persist it, and transition the session back to
    /// `Authenticated`.
    ///
    /// If the authorization server rotates the refresh token (Req 3.2
    /// supports it being optional), the new value is adopted; otherwise the
    /// previous refresh token is retained.
    #[tracing::instrument(skip(self, previous, response))]
    async fn apply_refreshed_tokens(
        &self,
        previous: &StoredTokens,
        response: TokenResponse,
    ) -> Result<(), OAuthError> {
        let expires_at = now_unix_secs().saturating_add(response.expires_in);
        let scopes = response.scope.unwrap_or_else(|| previous.scopes.clone());
        let refresh_token = response
            .refresh_token
            .unwrap_or_else(|| previous.refresh_token.clone());
        let tokens = StoredTokens {
            access_token: response.access_token,
            refresh_token,
            expires_at,
            scopes,
        };
        self.store_tokens(tokens).await
    }
}

/// Current wall-clock time as seconds since the Unix epoch. Clamped at 0 on
/// systems with a clock earlier than the epoch so refresh arithmetic never
/// underflows.
fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::backoff_delay_secs;
    use proptest::prelude::*;

    // Feature: openai-oauth-login, Property 8: Exponential Backoff Schedule
    //
    // **Validates: Requirement 5.5, Property 8**
    //
    // Req 5.5 specifies the refresh retry schedule as three attempts with
    // delays of 2 s, 4 s, and 8 s before marking the session expired. The
    // closed-form statement of that schedule (design.md, Property 8) is:
    //
    //     ∀ attempt ∈ {0, 1, 2}. backoff_delay_secs(attempt) == 2^(attempt + 1)
    //
    // The valid domain is exactly `{0, 1, 2}` because `BACKOFF_MAX_ATTEMPTS`
    // is 3, so a tight proptest range is sufficient — 16 cases let proptest
    // exercise each point in the domain with room to spare while keeping the
    // test cheap. We also pin the three literal schedule values (2, 4, 8) so
    // a regression that silently changes the formula but keeps the exponential
    // shape would still be caught.
    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 16,
            .. ProptestConfig::default()
        })]

        #[test]
        fn prop_backoff_schedule(attempt in 0u32..=2u32) {
            let expected = 2u64.pow(attempt + 1);
            let actual = backoff_delay_secs(attempt);
            prop_assert_eq!(actual, expected);

            match attempt {
                0 => prop_assert_eq!(actual, 2),
                1 => prop_assert_eq!(actual, 4),
                2 => prop_assert_eq!(actual, 8),
                _ => unreachable!(),
            }
        }
    }
}
