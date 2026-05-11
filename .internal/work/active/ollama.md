# Informity AI — Ollama Provider Integration Plan

**Status:** 🚧 IN PROGRESS (Phase 0, Phase 1, Phase 2, and Phase 3 completed)  
**Version:** 1.1  
**Last updated:** 2026-05-09  
**Scope:** Introduce provider pre-abstraction first, then add Ollama as an optional LLM runtime while preserving the current tuned GGUF path as default.

---

## Goal

Prepare Informity AI for multi-provider LLM runtime by first extracting the current `xllamacpp` path behind a provider abstraction with no behavior change, then adding Ollama as an optional provider for models outside our tuned in-process profiles.

---

## Document Role

- This document is a backlog implementation roadmap for Ollama provider support.
- It does not redefine existing model-profile contracts; it extends runtime/provider selection.
- Existing tuned model behavior remains owned by `src/informity/llm/model_adapter.py` and current settings routes.

---

## Implementation Summary

| Phase | Scope | Status |
|---|---|---|
| Phase 0 | Provider pre-abstraction (xllamacpp only, parity refactor) | ✅ COMPLETED |
| Phase 1 | Provider contract and settings surface | ✅ COMPLETED |
| Phase 2 | Ollama engine implementation and fallback behavior | ✅ COMPLETED |
| Phase 3 | Router/handler integration and model capability mapping | ✅ COMPLETED |
| Phase 4 | UX, diagnostics, and release hardening | ⏳ NOT STARTED |

---

## Rollout Strategy (Least Destructive)

Adopt an additive, runtime-first rollout to protect current stable behavior:

1. Implement provider runtime behavior first (Phase 2), with strict stream/event normalization to existing internal contracts.
2. Apply model-safety/capability defaults next (Phase 3), without changing the existing chat orchestration path.
3. Only after runtime parity is validated, wire provider-aware setup/UX and diagnostics hardening (Phase 4).

This sequencing intentionally avoids early changes to setup gates and chat flows before Ollama runtime behavior is proven.

### Phase 0 — Provider Pre-Abstraction (Parity Refactor) ✅ COMPLETED

- Goal: Refactor current LLM runtime internals to a provider interface without changing functionality, quality, or performance.
- Scope:
- Introduce provider contract used by `llm_engine` (streaming + sync completion + token count + lifecycle).
- Move existing `xllamacpp` implementation behind `XllamaCppProvider` with near-identical logic.
- Keep all current defaults, model pathing, timeouts, prompt/template handling, stop behavior, and diagnostics semantics.
- Preserve existing public engine API used by routes/handlers.
- Add parity-focused regression tests and before/after benchmark checks (TTFT, tokens/sec, end-to-end latency, memory).
- Implementation targets:
- `src/informity/llm/engine.py`
- `src/informity/llm/streaming.py`
- `tests/test_engine.py`
- Exit criteria:
- No API contract changes for current callers.
- No intentional behavior changes in local GGUF mode.
- No measurable regression in baseline performance/quality.

Completed notes:
- Added provider facade in `llm_engine` while preserving legacy engine API and private compatibility hooks.
- Current runtime moved behind `XllamaCppProvider`.
- Added placeholder `OllamaProvider` with explicit "not implemented yet" behavior.
- Parity regression tests passed for engine/config/settings slices.

### Phase 1 — Provider Contract and Settings Surface ✅ COMPLETED

- Goal: Expose explicit provider configuration once Phase 0 abstraction is complete.
- Scope:
- Add provider enum/settings: `local_gguf` (default), `ollama`.
- Add Ollama connection settings: base URL, model ID/tag, timeout, optional keep-alive.
- Extend settings API/schema and persisted config handling.
- Keep all existing defaults unchanged (current users should remain on `local_gguf`).
- Implementation targets:
- `src/informity/config.py`
- `src/informity/api/schemas.py`
- `src/informity/api/routes_settings.py`
- `src/frontend/src/pages/SettingsPage.tsx`
- Exit criteria:
- New provider settings are persisted and returned by `/api/settings`.
- Existing installs upgrade without config migration errors.
- Default behavior remains current in-process GGUF path.

Completed notes:
- Added `llm_provider` setting with allowed values `local_gguf` (default) and `ollama`.
- Extended settings schemas/routes/env-var metadata to include provider setting and validation.
- Kept default behavior unchanged for existing users (`local_gguf` path).

### Phase 2 — Ollama Engine Implementation and Fallback Behavior ✅ COMPLETED

- Goal: Add a production-safe Ollama-backed chat completion/streaming engine behind the Phase 0 provider interface.
- Scope:
- Extract provider-agnostic LLM engine interface (stream + sync completion where needed).
- Implement Ollama adapter for streaming and non-streaming completions.
- Normalize finish reasons and streaming chunk shapes to current internal format.
- Add clear error mapping for unreachable daemon, missing model, timeout, malformed stream.
- Implementation targets:
- `src/informity/llm/engine.py`
- `src/informity/llm/streaming.py`
- `src/informity/exceptions.py`
- Exit criteria:
- Researcher and assistant responses stream correctly through Ollama.
- Classifier/simple/non-stream paths work without contract regressions.
- Connection and model-not-found failures return actionable user-safe errors.

Completed notes:
- Implemented `OllamaProvider` runtime with `/api/chat` streaming and non-streaming support.
- Added stream normalization to existing internal token/finish-reason handling.
- Added provider-side error mapping for HTTP, connection, timeout, and malformed JSON responses.
- Added minimal Ollama runtime settings (`ollama_base_url`, `ollama_timeout_seconds`) to config + settings API + env-var metadata.
- Added focused engine tests for Ollama sync and streaming behavior.

### Phase 3 — Router/Handler Integration and Capability Mapping ✅ COMPLETED

- Goal: Preserve current quality safeguards when running untuned Ollama models.
- Scope:
- Define conservative default capability profile for Ollama models.
- Add optional per-model capability overrides (internal mapping first; user JSON deferred).
- Keep existing tuned `ModelProfile` logic for `local_gguf` models.
- For Ollama models, apply safe defaults for reasoning, stops, sampling, and retrieval budgets.
- Implementation targets:
- `src/informity/llm/model_adapter.py`
- `src/informity/llm/handlers/rag.py`
- `src/informity/llm/handlers/simple.py`
- `src/informity/llm/prompt_builder.py`
- Exit criteria:
- Untuned Ollama models do not break baseline chat or RAG flow.
- Default safeguards prevent common failure modes (reasoning leakage, premature stop, empty answer loops).
- Local GGUF tuned profiles behave exactly as before.

Completed notes:
- Added provider-aware profile selection (`get_profile`) that branches by `llm_provider`.
- Added conservative `OLLAMA_DEFAULT_PROFILE` for unknown Ollama models:
- reasoning disabled (`NEVER`), no `/no_think` token injection, conservative retrieval/time budget defaults.
- Added model-id alias matching for known Ollama IDs (`qwen-9b`, `qwen-14b`, `qwen-35b-a3b`) to reuse tuned profiles when applicable.
- Added profile-selection tests for Ollama known/unknown model IDs.

### Phase 4 — UX, Diagnostics, and Release Hardening ⏳ NOT STARTED

- Goal: Ship Ollama support with observability, docs, and rollback safety.
- Scope:
- Settings UI controls and validation messaging for Ollama setup.
- Provider-aware diagnostics metrics (provider used, model ID, error class).
- Add tests for provider selection, settings validation, stream contract, and fallback behavior.
- Add docs for installation, model pull expectations, privacy/offline implications.
- Provider-aware setup gating and readiness checks:
- `local_gguf` provider keeps current GGUF setup/download requirements.
- `ollama` provider bypasses GGUF setup gating and instead requires:
- Ollama daemon reachable at configured URL.
- Configured Ollama model available (pulled locally).
- Provider-aware operator guidance/error copy (e.g., daemon not running, model missing, timeout).
- Implementation targets:
- `src/frontend/src/pages/SettingsPage.tsx`
- `src/informity/api/routes_settings.py`
- `src/informity/api/routes_system.py`
- `src/informity/chat_trace.py`
- `tests/`
- `README.md`
- Exit criteria:
- End-to-end provider switch works and is reversible without manual config edits.
- Diagnostics can distinguish local GGUF vs Ollama regressions.
- Release notes/docs include operational setup and troubleshooting paths.
- Setup screen/readiness behavior is provider-specific and does not force GGUF downloads when `llm_provider=ollama`.
- Existing `local_gguf` users see unchanged setup path by default.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Refactor introduces regressions before Ollama work starts | Require Phase 0 parity tests and baseline benchmark comparison before enabling provider selection changes. |
| Ollama model behavior diverges from tuned assumptions (reasoning tags, stops, formatting) | Start with conservative defaults; disable advanced reasoning controls unless explicitly validated per model. |
| Streaming contract mismatch causes UI regressions | Normalize Ollama stream payloads to existing internal token/event format and add adapter-level tests. |
| Settings complexity/confusion for current users | Keep `local_gguf` as default and gate Ollama controls behind explicit provider selection. |
| Privacy-mode expectation mismatch | Document provider network/localhost behavior and enforce clear warnings when provider is external or unavailable. |
| Regression risk to current tuned path | Strict provider branching, regression tests on current profiles, and no behavior changes when provider is `local_gguf`. |

---

## Non-Goals

- Replacing current tuned local GGUF runtime.
- Full parity tuning for every Ollama model at launch.
- User-editable arbitrary model profile JSON in this phase.
- LM Studio integration in this document (tracked separately).

---

## Success Criteria

- Phase 0 parity refactor lands with no behavior/performance/quality regression in current local GGUF mode.
- Ollama is selectable as an alternate provider with stable researcher and assistant chat behavior.
- Existing local GGUF users see no behavior change by default.
- Unsupported/untuned model risks are contained via conservative defaults and explicit diagnostics.
