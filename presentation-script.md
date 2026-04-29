
# OBEY API Router - 10 Minute Presentation Script

## Slide 1: Title & Introduction (30 seconds)

**Speaker Notes:**
> Good [morning/afternoon]. Today I'm going to introduce you to the OBEY API Router — an intelligent AI gateway designed to ensure your AI applications stay running, even when your providers don't.
>
> My name is [Your Name], and I built this because I was tired of manually switching between AI providers every time one became unreliable.

---

## Slide 2: The Problem (2 minutes)

**Speaker Notes:**
> Let me start with why this exists. Like many of you, I rely heavily on AI APIs for my work. I was using a particular provider that, frankly, became increasingly unreliable.
>
> The culprit? An influx of users — likely due to OpenClaw and similar tools — overwhelmed the provider's infrastructure. Suddenly, I was dealing with:
>
> - **Rate limiting errors** — 429s coming back constantly
> - **Timeouts** — requests hanging for minutes
> - **Degraded responses** — partial outputs, malformed JSON, or just garbage
> - **Complete outages** — the service just... disappearing
>
> And here's the frustrating part: I was doing this manually. Every time a provider failed, I'd:
> 1. Notice the error
> 2. Switch my code to use a different provider
> 3. Maybe change the model name
> 4. Test it
> 5. Hope the new provider was stable
>
> This is not sustainable. I needed my AI applications to have **high availability** and **fault tolerance**. I needed to get work done, even if I wasn't using the latest and greatest model.
>
> So I built OBEY API Router.

---

## Slide 3: What is OBEY API Router? (1.5 minutes)

**Speaker Notes:**
> OBEY API Router is an OpenAI-compatible API gateway written in Rust. It sits between your application and your AI providers, acting as an intelligent proxy.
>
> **Key characteristics:**
>
> - **Single binary** — no dependencies to install, just download and run
> - **OpenAI-compatible** — drop-in replacement for OpenAI's API
> - **Multi-provider support** — works with OpenAI, Anthropic via proxy, Groq, Ollama, LM Studio, NVIDIA NIM, vLLM, Together AI, AWS Bedrock, and more
> - **Self-hosted** — you control your data, your keys, your infrastructure
>
> The beauty is that your application code doesn't change. You point your existing OpenAI SDK or application at OBEY API Router instead of `api.openai.com`, and everything just works.
>
> But now you get superpowers.

---

## Slide 4: Core Features — Intelligent Routing (2 minutes)

**Speaker Notes:**
> Let's talk about the core features, starting with **Intelligent Routing**.
>
> **Model Groups with Priority Fallback:**
>
> You define "model groups" — logical names for what model you want. For example, I have a "GLM" model group that includes:
> - GLM-5:thinking (priority 1)
> - GLM-4.7:thinking (priority 2)
> - GLM-4.6:thinking (priority 3)
> - And so on...
>
> When you request "GLM", the router tries the highest priority model first. If that fails, it automatically falls back to the next one. No manual intervention.
>
> **Cross-Provider Support:**
>
> The same model can be available from multiple providers. If Nano-GPT is down but Nvidia NIM has the same model, the router handles that transparently.
>
> **Cost Awareness:**
>
> You can configure cost per million tokens for each provider/model combination. The router can optimize for cost while maintaining availability.
>
> **Context Management:**
>
> Ever hit a "context length exceeded" error? The router automatically handles context window limits. It can truncate your conversation using strategies like "remove oldest" or "sliding window" and retry automatically.

---

## Slide 5: Core Features — Fault Tolerance (2 minutes)

**Speaker Notes:**
> Now let's talk about **Fault Tolerance** — this is where the magic happens.
>
> **Circuit Breaker Pattern:**
>
> The router implements a circuit breaker for each provider. When a provider starts failing:
> - After N failures (configurable), the circuit "opens"
> - The router stops sending requests to that provider
> - It uses exponential backoff before trying again
> - Once it sees success, the circuit "closes" and traffic resumes
>
> This prevents cascading failures and gives struggling providers time to recover.
>
> **Automatic Retry with Backoff:**
>
> When a request fails, the router can retry it with configurable backoff sequences. But here's the smart part — it doesn't retry 4xx errors (except 429 rate limits and 408 timeouts). Those are client errors, retrying won't help.
>
> **Rate Limit Handling:**
>
> When a provider returns 429 (rate limited), the router:
> 1. Respects the `Retry-After` header if present
> 2. Falls back to another provider if available
> 3. Queues the request if no fallback exists
>
> **Latency Tracking:**
>
> The router tracks response latency per provider and model. This data feeds into routing decisions and is visible in the dashboard.

---

## Slide 6: Core Features — Observability & Admin (1.5 minutes)

**Speaker Notes:**
> You can't manage what you can't see. OBEY API Router includes built-in observability.
>
> **Admin Dashboard:**
>
> A web-based admin panel (embedded in the binary, no external dependencies) where you can:
> - Configure providers and model groups
> - View circuit breaker status for each provider
> - See which providers are healthy, degraded, or offline
> - Manage API keys with encrypted storage
>
> **Real-Time Metrics:**
>
> The dashboard shows real-time metrics via WebSocket:
> - Requests per second
> - Token usage
> - Error rates
> - Latency percentiles
>
> **Request Logging:**
>
> All requests are logged to a local SQLite database with:
> - Timestamp, provider, model, status code
> - Trace IDs for debugging
> - Configurable retention (default 30 days)
>
> **Prometheus Integration:**
>
> For those with existing monitoring infrastructure, metrics are exposed at `/metrics` in Prometheus format.

---

## Slide 7: Advanced Features (1.5 minutes)

**Speaker Notes:**
> Let me quickly cover some advanced features for power users.
>
> **Semantic Caching:**
>
> If you enable it, the router can cache similar requests using vector embeddings. Ask "What is Python?" and later ask "Tell me about Python" — the cached response is returned without hitting the provider. This saves tokens, money, and latency.
>
> **Encrypted Key Storage:**
>
> API keys can be stored encrypted in the config file. The master key lives outside the config, in your platform's secure directory. This makes it safe to share config files without exposing secrets.
>
> **Desktop / System Tray Mode:**
>
> On Windows, you can run the gateway as a tray application. Double-click the executable, it starts in the background, shows a splash screen, and opens the dashboard. No command-line needed.
>
> **Hot Configuration Reload:**
>
> Change your config through the admin UI, and it takes effect immediately. No restart required.

---

## Slide 8: Architecture & Technology (1 minute)

**Speaker Notes:**
> A quick word on the technology stack.
>
> **Written in Rust** — for performance, safety, and reliability. No garbage collection pauses, memory-safe by design.
>
> **Async Architecture** — built on Tokio, handles thousands of concurrent connections efficiently.
>
> **Embedded Assets** — the admin UI and dashboard are compiled into the binary. Single file deployment.
>
> **SQLite for Logs** — lightweight, no external database required.
>
> **OpenAPI Compatible** — supports chat completions, embeddings, image generation, audio transcription, and passthrough for Assistants API.

---

## Slide 9: Demo / Use Case Walkthrough (1.5 minutes)

**Speaker Notes:**
> Let me walk you through a real-world scenario.
>
> **Scenario:** You're building an AI assistant that uses GLM models. You want high availability.
>
> **Step 1:** Configure multiple providers in `config.yaml`:
> ```yaml
> providers:
>   - name: Nano-GPT
>     type: openai
>     base_url: https://nano-gpt.com/api/
>   - name: Nvidia NIM
>     type: nvidia_nim
>     base_url: https://integrate.api.nvidia.com/
> ```
>
> **Step 2:** Define your model group with fallbacks:
> ```yaml
> model_groups:
>   - name: GLM
>     models:
>       - provider: Nano-GPT
>         model: glm-5:thinking
>         priority: 1
>       - provider: Nvidia NIM
>         model: glm5
>         priority: 2
> ```
>
> **Step 3:** Point your application at the gateway:
> ```bash
> export OPENAI_API_BASE=http://localhost:8080/v1
> ```
>
> **Result:** Your application calls "GLM". If Nano-GPT fails, the router automatically tries Nvidia NIM. You didn't change any code. Your users didn't notice anything. You kept working.

---

## Slide 10: Conclusion & Q&A (1 minute)

**Speaker Notes:**
> To summarize:
>
> **OBEY API Router solves the problem of unreliable AI providers by:**
> - Automatically failing over between providers and models
> - Implementing circuit breakers to prevent cascading failures
> - Providing visibility through dashboards and metrics
> - Managing context windows automatically
> - Caching responses to reduce costs
>
> **Why I built it:**
> - My provider became unreliable due to user influx
> - I was manually switching providers — not scalable
> - I needed high availability for my AI applications
> - I wanted to get work done, even without the "best" model
>
> **The result:**
> - A single binary that just works
> - OpenAI-compatible, drop-in replacement
> - Self-hosted, you control your data
> - Written in Rust for performance and reliability
>
> The code is available at [your repository link]. I'd love to hear your feedback and answer any questions.
>
> Thank you.

---

## Appendix: Key Talking Points for Q&A

**Q: Why Rust?**
> Performance, safety, and single-binary deployment. No runtime dependencies, no GC pauses, memory-safe by design.

**Q: How does it compare to other gateways like LiteLLM?**
> LiteLLM is Python-based and feature-rich. OBEY is a native binary focused on simplicity and performance. Choose based on your infrastructure preferences.

**Q: Can I use this in production?**
> Yes. It's designed for production use with proper error handling, logging, and observability.

**Q: What about streaming?**
> Full support for streaming responses via SSE (Server-Sent Events). The router streams from the provider to your client with minimal latency overhead.

**Q: How do I handle authentication?**
> The gateway can pass through your provider API keys, or store them encrypted locally. Admin UI supports basic auth for the dashboard.

**Q: What's the performance overhead?**
> Minimal. The router adds typically <10ms latency for routing decisions. It's designed to be a thin, fast proxy.
