//! JWT decoder for extracting `chatgpt_account_id` from OpenAI OAuth access tokens.
//!
//! The Codex backend at `https://chatgpt.com/backend-api/codex/responses`
//! requires a `chatgpt-account-id` header sourced from the OAuth access
//! token's payload. The token is a standard three-segment JWT
//! (`header.payload.signature`) whose middle segment is a base64url-no-pad
//! JSON object. The `chatgpt_account_id` claim appears under the nested
//! key `"https://api.openai.com/auth"."chatgpt_account_id"` in the Codex
//! CLI convention, with a flat top-level `"chatgpt_account_id"` accepted
//! as a fallback.
//!
//! # Security
//!
//! Per Requirement 1.7 of the `codex-backend-translation` spec, the access
//! token value, the raw payload segment, and the decoded JSON object must
//! never appear in logs at any level. The returned [`OAuthError`] variants
//! are discriminator-only; they do not carry payload bytes.
//!
//! The `#[tracing::instrument(skip(access_token))]` attribute on
//! [`extract_chatgpt_account_id`] ensures that even when the function is
//! invoked inside a traced span, the token is not recorded as a field.

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;

use crate::oauth::OAuthError;

/// Extract the `chatgpt_account_id` claim from a JWT access token.
///
/// Accepts both the nested Codex CLI convention
/// (`"https://api.openai.com/auth": { "chatgpt_account_id": "..." }`) and a
/// flat variant (top-level `"chatgpt_account_id"`). Nested lookup is tried
/// first; the flat variant is consulted only when the nested key is absent
/// or the nested value is not a non-empty string.
///
/// # Errors
///
/// * [`OAuthError::JwtMalformed`] — the token does not have exactly three
///   `.`-separated segments.
/// * [`OAuthError::JwtDecodeFailed`] — the middle segment is not
///   base64url-no-pad decodable, or the decoded bytes are not valid JSON.
/// * [`OAuthError::JwtMissingAccountId`] — the JSON parsed successfully but
///   the `chatgpt_account_id` claim is absent, non-string, or an empty
///   string under both the nested and flat lookup strategies.
///
/// # Security
///
/// The token, payload bytes, and decoded JSON are never logged. Error
/// variants carry only discriminators, no payload data.
#[tracing::instrument(skip(access_token))]
pub fn extract_chatgpt_account_id(access_token: &str) -> Result<String, OAuthError> {
    // Segment count must be exactly 3 (header.payload.signature).
    // `split('.')` includes empty segments, so `"a.b"` → 2, `"a.b.c.d"` → 4.
    let segments: Vec<&str> = access_token.split('.').collect();
    if segments.len() != 3 {
        return Err(OAuthError::JwtMalformed);
    }
    let payload_b64 = segments[1];

    let raw = URL_SAFE_NO_PAD
        .decode(payload_b64.as_bytes())
        .map_err(|_| OAuthError::JwtDecodeFailed)?;
    let json: serde_json::Value =
        serde_json::from_slice(&raw).map_err(|_| OAuthError::JwtDecodeFailed)?;

    // Nested variant first (Codex CLI convention).
    if let Some(s) = json
        .get("https://api.openai.com/auth")
        .and_then(|v| v.get("chatgpt_account_id"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
    {
        return Ok(s.to_owned());
    }

    // Flat variant fallback.
    if let Some(s) = json
        .get("chatgpt_account_id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
    {
        return Ok(s.to_owned());
    }

    Err(OAuthError::JwtMissingAccountId)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;
    use serde_json::json;

    /// Build a three-segment JWT with the given payload JSON. Header and
    /// signature are dummy strings — the decoder never inspects them.
    fn make_jwt(payload: &serde_json::Value) -> String {
        let payload_bytes = serde_json::to_vec(payload).expect("serialize payload");
        let payload_b64 = URL_SAFE_NO_PAD.encode(payload_bytes);
        format!("hhh.{payload_b64}.sss")
    }

    #[test]
    fn nested_claim_is_extracted() {
        let token = make_jwt(&json!({
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct_123"
            }
        }));
        assert_eq!(
            extract_chatgpt_account_id(&token).expect("should extract"),
            "acct_123"
        );
    }

    #[test]
    fn flat_claim_is_extracted_when_nested_absent() {
        let token = make_jwt(&json!({
            "chatgpt_account_id": "acct_flat"
        }));
        assert_eq!(
            extract_chatgpt_account_id(&token).expect("should extract"),
            "acct_flat"
        );
    }

    #[test]
    fn segment_count_too_few_is_malformed() {
        let err = extract_chatgpt_account_id("only.two").expect_err("should fail");
        assert!(matches!(err, OAuthError::JwtMalformed));
    }

    #[test]
    fn segment_count_too_many_is_malformed() {
        let err = extract_chatgpt_account_id("a.b.c.d").expect_err("should fail");
        assert!(matches!(err, OAuthError::JwtMalformed));
    }

    #[test]
    fn undecodable_payload_is_decode_failed() {
        // `!` is not a valid base64url character.
        let err = extract_chatgpt_account_id("h.!!!not_base64!!!.s").expect_err("should fail");
        assert!(matches!(err, OAuthError::JwtDecodeFailed));
    }

    #[test]
    fn non_json_payload_is_decode_failed() {
        // Valid base64url, but decodes to "not json at all".
        let payload_b64 = URL_SAFE_NO_PAD.encode(b"not json at all");
        let token = format!("h.{payload_b64}.s");
        let err = extract_chatgpt_account_id(&token).expect_err("should fail");
        assert!(matches!(err, OAuthError::JwtDecodeFailed));
    }

    #[test]
    fn missing_claim_is_missing_account_id() {
        let token = make_jwt(&json!({ "sub": "user_xyz", "exp": 1_700_000_000u64 }));
        let err = extract_chatgpt_account_id(&token).expect_err("should fail");
        assert!(matches!(err, OAuthError::JwtMissingAccountId));
    }

    #[test]
    fn empty_string_claim_is_missing_account_id() {
        let token = make_jwt(&json!({ "chatgpt_account_id": "" }));
        let err = extract_chatgpt_account_id(&token).expect_err("should fail");
        assert!(matches!(err, OAuthError::JwtMissingAccountId));
    }

    #[test]
    fn non_string_claim_is_missing_account_id() {
        let token = make_jwt(&json!({ "chatgpt_account_id": 42 }));
        let err = extract_chatgpt_account_id(&token).expect_err("should fail");
        assert!(matches!(err, OAuthError::JwtMissingAccountId));
    }

    #[test]
    fn nested_empty_string_falls_back_to_flat() {
        // Nested present but empty → fallback to flat wins.
        let token = make_jwt(&json!({
            "https://api.openai.com/auth": { "chatgpt_account_id": "" },
            "chatgpt_account_id": "acct_flat_wins"
        }));
        assert_eq!(
            extract_chatgpt_account_id(&token).expect("should extract"),
            "acct_flat_wins"
        );
    }
}

#[cfg(test)]
mod property_tests {
    //! Property-based tests for [`extract_chatgpt_account_id`].
    //!
    //! Validates: Requirements 13.1

    use super::*;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;
    use proptest::prelude::*;
    use serde_json::json;

    proptest! {
        /// **Property 1 — JWT Round-Trip**
        ///
        /// For all `account_id ∈ [A-Za-z0-9_-]{1,64}`, synthesizing a JWT
        /// whose payload carries the nested
        /// `"https://api.openai.com/auth"."chatgpt_account_id"` claim and
        /// wrapping it with dummy header/signature segments must round-trip:
        /// `extract_chatgpt_account_id(token) == Ok(account_id)`.
        ///
        /// Validates: Requirements 13.1
        #[test]
        fn jwt_round_trip(account_id in "[A-Za-z0-9_-]{1,64}") {
            let payload = json!({
                "https://api.openai.com/auth": {
                    "chatgpt_account_id": account_id,
                }
            });
            let payload_bytes = serde_json::to_vec(&payload)
                .expect("serialize payload");
            let payload_b64 = URL_SAFE_NO_PAD.encode(payload_bytes);
            let token = format!("hhh.{payload_b64}.sss");

            let extracted = extract_chatgpt_account_id(&token)
                .expect("round-trip must succeed");
            prop_assert_eq!(extracted, account_id);
        }
    }
}
