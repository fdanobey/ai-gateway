<p align="center">
  <img src="Assets/logo.jpg" alt="OBEY API Gateway" width="200" />
</p>

<h1 align="center">OBEY API Gateway</h1>

<p align="center">
  OpenAI-compatible AI gateway with intelligent routing, automatic failover, and multi-provider support.<br/>
  Single Rust binary. No runtime dependencies. Just download and run.
</p>

<p align="center">
  <a href="https://github.com/fdanobey/ai-gateway/releases/latest"><img src="https://img.shields.io/github/v/release/fdanobey/ai-gateway?style=flat-square" alt="Release" /></a>
  <a href="https://github.com/fdanobey/ai-gateway/actions"><img src="https://img.shields.io/github/actions/workflow/status/fdanobey/ai-gateway/release.yml?style=flat-square&label=build" alt="Build" /></a>
</p>

<p align="center">
  <strong><a href="https://github.com/fdanobey/ai-gateway/releases/latest">Download Latest Release</a></strong> · Windows installer and portable zip
</p>

---

## Why?

AI providers go down. Rate limits hit. Models get deprecated. When that happens, you're stuck manually switching providers, changing model names, and hoping the next one works.

OBEY API Gateway sits between your application and your AI providers. Point your existing OpenAI SDK at it instead of `api.openai.com`, and you get automatic failover, circuit breakers, and multi-provider routing — without changing your application code.

## Key Features

- **Drop-in OpenAI replacement** — full `/v1/*` API compatibility (chat, completions, embeddings, images, audio, assistants)
- **Multi-provider routing** — OpenAI, Ollama, AWS Bedrock, Groq, Together AI, NVIDIA NIM, vLLM, LM Studio
- **Automatic failover** — circuit breakers + retry with exponential backoff across providers
- **Priority & cost-aware routing** — configure model groups with priority, cost, and latency-based selection
- **Context window management** — automatic truncation when requests exceed model limits
- **Semantic caching** — optional Qdrant-based response caching to reduce cost and latency
- **Encrypted API key storage** — provider keys encrypted at rest with a machine-local master key
- **Admin panel & dashboard** — embedded web UIs for configuration, metrics, and log viewing
- **Prometheus metrics** — `/metrics` endpoint for existing monitoring infrastructure
- **Request logging** — SQLite-based structured logging with configurable retention
- **TLS support** — optional HTTPS with certificate configuration
- **Windows system tray** — double-click desktop app with splash screen and tray menu
- **Hot config reload** — change settings through the admin UI without restarting

## Quick Start

### Option 1: Download (Windows)

Grab the [latest release](https://github.com/fdanobey/ai-gateway/releases/latest) — either the installer (`.exe`) or portable zip. Double-click to run. The gateway starts on `http://localhost:8080` and opens the dashboard automatically on first launch.

### Option 2: Build from Source

```bash
# Clone
git clone https://github.com/fdanobey/ai-gateway.git
cd ai-gateway

# Build (headless)
cargo build --release -p ai-gateway

# Build with Windows tray support
cargo build --release -p ai-gateway --features tray

# Run
./target/release/ai-gateway --config ./config.yaml
```

### Point Your App at the Gateway

```bash
# Any OpenAI-compatible SDK or tool
export OPENAI_API_BASE=http://localhost:8080/v1
```

```python
# Python example
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
response = client.chat.completions.create(
    model="gpt-4-group",  # Use your model group name
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Configuration

Config file is resolved in this order:

1. `--config` CLI flag
2. `CONFIG_PATH` environment variable
3. `./config.yaml` in working directory

If no config exists on first run, a default is created automatically. See [`config.example.yaml`](crates/ai-gateway/config.example.yaml) for the full reference.

### Minimal Example

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  request_timeout_seconds: 30

providers:
  - name: "openai"
    type: "openai"
    base_url: "https://api.openai.com/v1"
    api_key_env: "OPENAI_API_KEY"       # Env var name, not the key itself
    timeout_seconds: 30

  - name: "ollama"
    type: "ollama"
    base_url: "http://localhost:11434"
    timeout_seconds: 120

model_groups:
  - name: "gpt-4-group"
    models:
      - provider: "openai"
        model: "gpt-4"
        priority: 1                     # Lower = higher priority
      - provider: "ollama"
        model: "llama3"
        priority: 2                     # Fallback
```

### Provider Types

| Type | Provider | Notes |
|------|----------|-------|
| `openai` | OpenAI, Nano-GPT, any OpenAI-compatible API | Generic OpenAI protocol |
| `ollama` | Ollama | Local models, no API key needed |
| `bedrock` | AWS Bedrock | API key or AWS SDK auth ([details below](#bedrock-authentication)) |
| `groq` | Groq | |
| `together` | Together AI | |
| `nvidia_nim` | NVIDIA NIM | |
| `vllm` | vLLM | Self-hosted inference |
| `lmstudio` | LM Studio | Local models |

### API Key Management

Provider keys can be configured three ways:

1. **Environment variable reference** — set `api_key_env: "OPENAI_API_KEY"` and export the env var
2. **Admin UI** — enter keys through the web interface; they're encrypted automatically
3. **Encrypted in config** — stored as `api_key_encrypted: "enc-v1:<nonce>:<ciphertext>"`

The master encryption key is stored outside the config file in your platform's secure directory (e.g. `%APPDATA%\ai-gateway\master.key` on Windows).

### Bedrock Authentication

AWS Bedrock supports two modes:

```yaml
# Mode 1: API key (Bedrock Mantle endpoint)
- name: "bedrock-api-key"
  type: "bedrock"
  region: "us-east-1"
  api_key_env: "AWS_BEARER_TOKEN_BEDROCK"

# Mode 2: AWS SDK credentials (env vars, shared credentials, IAM role)
- name: "bedrock-sdk"
  type: "bedrock"
  region: "us-east-1"
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `CONFIG_PATH` | Override config file location |
| `OPENAI_API_KEY` | Provider API key (name matches `api_key_env` in config) |
| `ADMIN_USERNAME` | Admin panel username |
| `ADMIN_PASSWORD` | Admin panel password |
| `RUST_LOG` | Tracing filter (`info`, `debug`, `ai_gateway=trace`) |

## API Endpoints

All `/v1/*` endpoints are OpenAI-compatible. Requests include an `x-trace-id` response header for correlation.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Prometheus metrics |
| `POST` | `/v1/chat/completions` | Chat completions (streaming + non-streaming) |
| `POST` | `/v1/completions` | Legacy completions |
| `POST` | `/v1/embeddings` | Embeddings |
| `GET` | `/v1/models` | List available models |
| `POST` | `/v1/images/generations` | Image generation |
| `POST` | `/v1/audio/transcriptions` | Audio transcription |
| `POST` | `/v1/audio/translations` | Audio translation |
| `*` | `/v1/assistants/**` | Assistants API passthrough |
| `*` | `/v1/threads/**` | Threads & messages passthrough |
| `*` | `/v1/files/**` | Files API passthrough |
| `*` | `/v1/fine_tuning/**` | Fine-tuning API passthrough |

## How Routing Works

1. Your app requests model `"gpt-4-group"`
2. The gateway finds the matching model group
3. Providers are sorted by priority → cost → latency
4. Providers with open circuit breakers or exhausted rate limits are skipped
5. The request goes to the highest-priority available provider
6. On failure, the gateway retries with the next provider in the list
7. If a context-length error is detected, the gateway truncates and retries automatically

### Circuit Breaker

Each provider has an independent circuit breaker. After `failure_threshold` consecutive failures, the circuit opens and the provider is temporarily removed from rotation. Backoff follows a configurable sequence (e.g. 5s → 10s → 20s → 40s → 300s). Circuit breakers reset on config hot-reload.

### Context Management

When a provider returns a context-length error, the gateway can automatically truncate the conversation and retry:

- **`remove_oldest`** — removes oldest messages, preserving system messages
- **`sliding_window`** — keeps only the N most recent messages

```yaml
context:
  enabled: true
  truncation_strategy: "remove_oldest"
  sliding_window_size: 10
  max_truncation_retries: 3
```

## Admin Panel & Dashboard

Both are embedded SPAs compiled into the binary — no external dependencies.

- **Admin** (`/admin`) — provider configuration, API key management, circuit breaker status, config hot-reload
- **Dashboard** (`/dashboard`) — real-time metrics via WebSocket, provider health, error logs, request log viewer

```yaml
admin:
  enabled: true
  path: "/admin"
  auth:
    enabled: true
    username_env: "ADMIN_USERNAME"
    password_env: "ADMIN_PASSWORD"

dashboard:
  enabled: true
  path: "/dashboard"
```

## Desktop / System Tray Mode

When built with `--features tray` on Windows, the binary runs as a desktop application:

- First launch shows a splash screen, starts the gateway, and opens the dashboard
- Subsequent launches start silently in the system tray
- Tray menu provides quick access to Dashboard, Admin, server status, and Quit
- Single-instance guard prevents duplicate processes
- Optional Windows login startup entry

## Project Structure

```
.
├── Cargo.toml                        # Workspace manifest
├── crates/
│   └── ai-gateway/
│       ├── Cargo.toml                # Crate manifest & dependencies
│       ├── build.rs                  # Windows resource embedding
│       ├── config.example.yaml       # Reference configuration
│       └── src/
│           ├── main.rs               # Entry point, CLI, tray bootstrap
│           ├── lib.rs                # Public module exports
│           ├── config/               # Config structs & validation
│           ├── gateway/              # HTTP server, middleware, route handlers
│           ├── router/               # Provider selection, circuit breaker, rate limiter
│           ├── providers/            # Provider implementations (8 providers)
│           ├── context/              # Context window management & truncation
│           ├── cache/                # Semantic caching (Qdrant)
│           ├── admin/                # Admin panel routes & embedded UI
│           ├── dashboard/            # Dashboard routes & WebSocket metrics
│           ├── logger/               # SQLite request logging
│           ├── metrics/              # Prometheus metrics
│           ├── secrets.rs            # API key encryption/decryption
│           ├── error/                # Error types & HTTP status mapping
│           ├── models/               # OpenAI-compatible data models
│           └── tray/                 # Windows system tray (feature-gated)
├── scripts/
│   ├── build-release.ps1             # Release packaging script
│   ├── build-installer.ps1           # Inno Setup installer build
│   └── installer.iss                 # Inno Setup configuration
├── Assets/                           # Icons and logos
└── .github/workflows/release.yml     # CI/CD: build + GitHub Release on tag
```

## Technologies

| Category | Technology |
|----------|-----------|
| Language | Rust (2021 edition) |
| Async runtime | Tokio |
| Web framework | Axum + Tower middleware |
| HTTP client | Reqwest |
| Database | SQLite (rusqlite, bundled) |
| Vector DB | Qdrant (optional, for semantic cache) |
| Crypto | ring + base64 |
| TLS | rustls via axum-server |
| AWS | aws-sdk-bedrockruntime + aws-config |
| CLI | clap |
| Logging | tracing + tracing-subscriber |
| Asset embedding | rust-embed |
| Testing | proptest, wiremock, tempfile |
| CI/CD | GitHub Actions |
| Installer | Inno Setup 6 |

## Building for Release

```powershell
# Full release package (binary + assets + zip)
powershell -ExecutionPolicy Bypass -File ./scripts/build-release.ps1

# Release package + Windows installer
powershell -ExecutionPolicy Bypass -File ./scripts/build-installer.ps1
```

The release profile is optimized for size and performance:

```toml
[profile.release]
opt-level = 3
lto = true
strip = true
codegen-units = 1
panic = "abort"
```

## Testing

```bash
cargo test -p ai-gateway               # All tests
cargo test -p ai-gateway <test_name>   # Single test
cargo test -p ai-gateway -- --nocapture # With output
```

Tests use `tower::ServiceExt::oneshot()` for integration testing (no port binding) and `proptest` for property-based validation of config parsing and input handling.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes — match the existing code style and patterns
4. Run `cargo test -p ai-gateway` and ensure all tests pass
5. Run `cargo clippy -p ai-gateway` for lint checks
6. Submit a pull request

### Guidelines

- Keep patches focused and minimal
- Pin dependency versions; justify new dependencies
- Use environment variables for secrets — never hardcode API keys
- Add tests for new routing logic or provider implementations
- Property-based tests (`proptest`) are preferred for input validation

## License

Copyright © 2026. All rights reserved.
