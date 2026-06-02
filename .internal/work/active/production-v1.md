# Informity AI — Production Readiness v1

**Status:** 🔄 IN PROGRESS  
**Version:** 1.0  
**Last updated:** 2026-06-02  
**Scope:** Production readiness assessment and remediation for the Translation feature and cross-cutting concerns (logging, LLM co-dependency, stop actions)

---

## Goal

The translation feature (feature/translate-v3, merged to develop) is functionally complete but requires several targeted fixes and verifications before it can be called production-ready. This document tracks the remaining work across six dimensions: pipeline robustness, multi-model support, logging standardization, LLM co-dependency, stop/cancel correctness, and explicit go/no-go criteria.

---

## Implementation Summary

| Phase | Scope | Status |
|-------|-------|--------|
| Phase 1 | Fix retry-on-truncation (pipeline correctness) | ⏳ NOT STARTED |
| Phase 2 | Chat activity log — application-level structlog event | ⏳ NOT STARTED |
| Phase 3 | Cancel path — activity log event for user-initiated stops | ⏳ NOT STARTED |
| Phase 4 | Output validation for translate | ⏸️ DEFERRED |
| Phase 5 | LLM co-dependency — backend mutual exclusion | ⏸️ DEFERRED |
| Phase 6 | Scan deferral during translation | ⏸️ DEFERRED |

---

## Current State Assessment

### Pipeline Robustness ✅ Mostly done, one critical fix remaining

**Completed:**
- Section-aware chunking from Docling section boundaries
- Glossary extraction with immediate cancel support (`gen.aclose()`)
- Progressive SSE streaming with full reconnect replay (`glossary_done`, `sections_ready`, all `section_done`)
- Page reload recovery via sessionStorage + SSE re-connect
- TOCTOU lock protection with fast-fail in `_run_translate_job`
- Truncation budget fix: 1.35× → 1.6× output multiplier (1450 → 1700 tokens)
- `finish_reason` captured per section (`stop`, `length`, `timeout`, `cancelled`)
- LaTeX rendering via `remark-math` + `rehype-katex`
- Table group-header row detection and full-width rendering
- Markdown artifact pre-processing (orphaned separators, HTML normalisation)
- AST-based plaintext export replacing regex stripping
- Activity log: `translate_job_completed/failed/stalled` events

**Critical gap — retry-on-truncation is counterproductive (Phase 1):**
When `finish_reason=length`, the retry path reduces input to `TRANSLATE_RETRY_TOKEN_CAP=400` tokens, capping output at only 740 tokens — worse than the original attempt. A truncated section gets a shorter truncated result on retry. The fix is to detect truncation and either skip retry or handle separately.

---

### Multi-Model Support (9B / 14B / 35B) ✅ Unified parameters work

**Completed:**
- Unified parameters are correct for all three models. Timeouts (360s section, 150s glossary) are very generous relative to measured speeds (37–52 tok/s actual).
- `finish_reason` captured per section → truncation rates observable per model via structlog
- Activity log shows per-job timing regardless of model
- No model-specific code paths — clean, maintainable

**Accepted limitations:**
- 9B translation accuracy is inherently lower; no automated detection or fallback. Accepted tradeoff for low-RAM hardware. Advisory warning removed at user request.
- Glossary quality from 9B not validated. A bad 9B glossary infects all sections. No alert mechanism. Accepted for v1.
- Truncation rates per model captured in `translate_job_completed.truncation_rate` but not surfaced in UI.

---

### Logging & Metrics Standardization ✅ Mostly done, one gap on chat side

**Completed — shared LLM engine tier (both pipelines):**
- `llm_stream_completed`: tokens, duration_ms, first_token_ms, finish_reason, output_length, provider — identical shape for chat and translate

**Completed — translate application tier:**
- `translate_section_completed`: job_id, section_index, section_title, status, finish_reason, truncated, chars_output, duration_ms, attempt_count
- `translate_job_completed`: job_id, language, tone, status, sections_total/completed/failed/truncated, truncation_rate, glossary_terms, total_elapsed_ms
- Activity log (DB-backed, user-facing): `translate_job_completed`, `translate_job_failed`, `translate_job_stalled`

**Completed — chat application tier:**
- `chat_response_completed` structlog event (already existed): chat_id, chat_mode, duration_ms, tokens_streamed, sources_count
- Activity log: `chat_message_generated` (always emitted, not gated by `chat_trace_logging`)

**Gap — chat side missing application-level structlog event (Phase 2):**
Chat has no structured `chat_generation_completed` event tagging `chat_id`, `mode`, and `retrieval_count` together at the application level. The only shared event is the LLM engine's `llm_stream_completed` which lacks pipeline context. This makes log queries like "how many Researcher mode responses took > 30s this week" impossible without joining HTTP request IDs. Not a blocker for v1 but a known gap.

**No retrieval metrics for chat:** chunks retrieved, relevance scores, retrieval latency are not logged. Deferred.

---

### LLM Co-Dependency (Chat ↔ Translate) ✅ UI-enforced, backend partial

**Completed:**
- `_translate_lock` asyncio.Lock prevents concurrent translate jobs
- 409 response when lock held; frontend retries with 400ms × attempt backoff (5 attempts)
- Cancel sets `_job_cancel_events[job_id]` → `gen.aclose()` → C++ thread released immediately
- UI: chat send button morphs to "Stop translation" when `isTranslating`; vice versa on translate
- `handleSend` returns early when `isTranslating` — Enter key blocked during translation
- Cancel endpoint guards against overwriting `done`/`failed` status (race condition fix)

**Known gap — no backend mutual exclusion between chat and translate (Phase 5 — deferred):**
`_translate_lock` gates translate jobs against each other, but not against concurrent chat. The C++ LLM engine serializes internally so this won't crash, but output quality degrades if both fire simultaneously. This is a UI-only guard. For a desktop app with a single user this is acceptable. Implement if the app ever exposes the API to multiple concurrent clients.

---

### Stop / Cancel Correctness ✅ Mostly done, one minor gap

**Completed:**
- Cancel endpoint: sets `status='stalled'`, sets cancel_event, emits SENTINEL to SSE queue
- `_translate_section`: polls cancel_event in generate_stream loop → `gen.aclose()` releases C++ thread
- `_extract_glossary`: also cancellable via cancel_event
- Frontend: aborts SSE stream (`AbortController`) + calls `DELETE /api/translate/jobs/{job_id}`
- `clearActiveJob()` clears sessionStorage on cancel
- Stop buttons use `ri-stop-circle-line` consistently across chat (streaming) and translate (own + cross-service)

**Minor gap — cancel doesn't emit activity log event (Phase 3):**
When `DELETE /api/translate/jobs/{job_id}` is called, `status='stalled'` is written to DB but no `translate_job_stalled` activity log event is emitted from the cancel path. The event only fires from the timeout stall check inside `_run_translate_job`. A user who cancels a translation sees nothing in the activity log. Low priority.

---

### Output Validation (Phase 4 — deferred)

Chat has `outputLint.ts` that detects source leakage, heading-level jumps, and table anomalies. Translate has no equivalent. If the model outputs English instead of Spanish, or structurally malformed markdown, it is displayed without warning. This is deferred to v2 — adding it for v1 would require defining what "valid translated output" looks like, which is model- and language-dependent.

---

## Phase 1 — Fix Retry on Truncation ⏳ NOT STARTED

- **Goal:** Prevent truncated sections from getting worse on retry. Currently `finish_reason=length` triggers a retry with a smaller input (400 tokens → 740 token output cap) which produces a shorter truncated result.
- **Scope:**
  - In `_run_translate_job`, capture `finish_reason` from each `_translate_section` call (already captured since our logging work)
  - If `finish_reason == 'length'` on the first attempt: skip retry entirely and accept the truncated output as the section result, logging a warning
  - The section still goes to `section_done` — the user gets what was generated rather than an even shorter version
  - Do NOT reduce `TRANSLATE_RETRY_TOKEN_CAP` — leave retry as-is for genuine error recovery
- **Exit criteria:**
  - [ ] Section with `finish_reason=length` is accepted as-is, no retry attempted
  - [ ] `translate_section_completed` log event reflects `truncated=True`, `attempt_count=1`
  - [ ] Section still delivered to frontend via `section_done` SSE event
  - [ ] Retry still fires for `finish_reason=stop` with no output (empty result)

---

## Phase 2 — Chat Application-Level Structlog Event ⏳ NOT STARTED

- **Goal:** Add a single structured `chat_generation_completed` event to `routes_chat.py` with pipeline context so chat performance can be analysed at the application level, not just the LLM engine level.
- **Scope:**
  - Emit after `chat_response_completed` log event at response completion
  - Fields: `chat_id`, `chat_mode` (assistant/researcher), `generation_seconds`, `tokens_output`, `sources_count`, `retrieval_latency_ms` (if available), `finish_reason`
  - Does NOT replace `chat_response_completed` (which is more detailed) — supplements it
  - Makes log queries like "Researcher mode responses > 30s" possible
- **Exit criteria:**
  - [ ] `chat_generation_completed` structlog event emitted for every completed AI reply
  - [ ] Fields match translate pipeline's event shape where applicable (duration_ms, finish_reason)
  - [ ] No performance impact on response streaming

---

## Phase 3 — Cancel Emits Activity Log Event ⏳ NOT STARTED

- **Goal:** User-initiated translation cancels should appear in the activity log, not just backend timeout stalls.
- **Scope:**
  - In `cancel_translate_job` endpoint (after status update + cancel_event set): call `emit_log_event` for `translate_job_stalled` with message indicating user cancellation
  - Need filename: fetch from DB (`get_file_by_id`) before emitting
  - Message format: `'Translation cancelled: \'filename\' → Language · N sections completed'`
- **Exit criteria:**
  - [ ] Activity log shows cancel event when user clicks Stop
  - [ ] Event does not fire if job was already done/failed (guard already exists)
  - [ ] Message is distinguishable from timeout stall (include "cancelled" in message)

---

## Phase 4 — Translation Output Validation ⏸️ DEFERRED

Deferred to v2. Requires defining language-specific validation rules and is out of scope for the initial production release.

**What it would cover:**
- Detect output in wrong language (model hallucination)
- Detect structurally malformed markdown that renders poorly
- Detect suspiciously short output relative to input length
- Surface warnings to user without blocking the translation display

**Decision gate:** Implement after user feedback indicates this is causing real-world problems.

---

## Phase 5 — Backend Chat↔Translate Mutual Exclusion ⏸️ DEFERRED

Deferred to v2. The desktop app's single-user model makes this a low-risk gap. The UI-level guard (`isTranslating` check in `handleSend`) covers normal usage.

**What it would cover:**
- Extend `_translate_lock` (or add a second shared lock) to cover chat requests
- Translate endpoint returns 409 if chat is streaming; chat endpoint returns 409 if translate is running
- Frontend already handles 409 gracefully with backoff retry

**Decision gate:** Implement if API is opened to multiple concurrent clients, or if user reports concurrent-use issues.

---

## Phase 6 — Scan Deferral During Translation ⏸️ DEFERRED

Deferred to v2. The current behaviour (scan runs concurrently with translation) works correctly but may cause resource contention on lower-end hardware.

**Why translation, not chat:**
Translation holds the LLM exclusively for 5–30+ minutes. Docling PDF extraction (the most CPU-intensive scan task) competes with LLM inference for RAM bandwidth and thermal headroom on Apple Silicon. Chat responses are 10–60 seconds — too short and too frequent to make scan pausing practical.

**What it would cover:**
- Before starting a new scan job, check if a translation job is active (query for `status IN ('queued', 'running')` in `translate_jobs`, or check `_translate_lock.locked()`)
- If active: defer the scan start; queue a rescan trigger for when the translation completes
- Do NOT interrupt a scan already in progress — only gate *new* scan starts
- Resume normal scan scheduling when translation reaches a terminal state (`done`, `failed`, `stalled`, `cancelled`)
- No change for chat streaming — scan proceeds as normal during chat

**Implementation targets:**
- `src/informity/scanner/` — scan scheduler / trigger point
- `src/informity/api/routes_translate.py` — emit a signal or update a shared flag on job completion
- Alternatively: scan scheduler polls `translate_jobs` table directly (no coupling to translate routes)

**Decision gate:** Implement if users on 16GB or lower RAM machines report thermal throttling or significantly degraded translation speed when a scan runs concurrently.

---

## Go / No-Go Criteria for Production v1

### Must fix before shipping (blocking):
- [ ] **Phase 1** (retry-on-truncation) — truncated sections must not be made worse by retry

### Should fix before shipping (recommended):
- [ ] **Phase 3** (cancel activity log event) — user cancels should appear in activity log

### Acceptable to ship with (known limitations):
- [x] 9B translation accuracy lower than 14B/35B — documented, accepted tradeoff
- [x] Chat↔translate mutual exclusion is UI-only — acceptable for single-user desktop
- [x] No translation output validation — deferred to v2
- [x] Startup-only cleanup of stale translate.local files — low impact for desktop
- [x] No translation history view — past jobs not browsable
- [x] Language support limited to 5 languages — expandable
- [x] Phase 15 (steering prompt) — deferred, documented in translate-v3.md

### Deferred (v2):
- [ ] Phase 2 (chat structlog event)
- [ ] Phase 4 (output validation)
- [ ] Phase 5 (backend mutual exclusion)
- [ ] Phase 6 (scan deferral during translation)

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Truncated sections visible to user with no warning | Phase 1 fix prevents retry making it worse; truncation rate logged in `translate_job_completed` for monitoring |
| 9B produces wrong-language output | No automated detection in v1; user sees the output immediately and can re-run with a better model |
| Chat and translate fire concurrently at API level | C++ engine serialises internally; UI prevents it in normal use; deferred to Phase 5 |
| Stale translate.local files accumulate | Cleanup runs on startup; for a desktop app this is acceptable — app restarts periodically |
| Scan runs concurrently with translation, causing thermal throttling on low-RAM hardware | Phase 6 deferred; decision gate is user-reported throttling. Most users on 32GB+ will not see this. |
| SSE connection lost mid-translation | Full replay on reconnect: `glossary_done`, `sections_ready`, all completed `section_done` events |

---

## Implementation Targets

| File | Phase | Change |
|------|-------|--------|
| `src/informity/api/routes_translate.py` | 1 | Skip retry when `finish_reason=length` |
| `src/informity/api/routes_translate.py` | 3 | Emit `translate_job_stalled` from cancel endpoint |
| `src/informity/api/routes_chat.py` | 2 | Add `chat_generation_completed` structlog event |
