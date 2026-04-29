# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Build & Run

```bash
cargo build --release -p ai-gateway    # Release binary at target/release/ai-gateway
cargo run -p ai-gateway -- --config ./config.yaml
```

## Test

```bash
cargo test -p ai-gateway               # All tests
cargo test -p ai-gateway <test_name>   # Single test
cargo test -p ai-gateway -- --nocapture  # With output
```

## Non-Obvious Patterns

- **API key resolution**: `api_key_env` in config is tried as env var name first, falls back to literal value if env var not found ([`router.rs:286-291`](crates/ai-gateway/src/router/router.rs:286))
- **Base URL normalization**: Provider URLs are stripped of trailing `/` and `/v1` is appended if missing ([`router.rs:278-283`](crates/ai-gateway/src/router/router.rs:278))
- **Config path resolution**: CLI `--config` → `CONFIG_PATH` env → `./config.yaml` ([`validation.rs`](crates/ai-gateway/src/config/validation.rs))
- **Circuit breaker reset**: All circuit breakers clear on config hot-reload via `/admin/config/reload`
- **Tests use `tower::ServiceExt::oneshot()`**: Integration tests don't bind ports; they call router directly
- **Property tests with proptest**: Many tests use `proptest!` macro for randomized input validation
