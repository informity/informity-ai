# Informity AI — Document Translation (v3)

**Status:** 🔄 IN PROGRESS  
**Version:** 1.1  
**Last updated:** 2026-05-31  
**Scope:** Clean-slate translation implementation — glossary-first, structure-aware chunking, shared composer UI pattern

---

## Document Role

- This document describes the third implementation attempt of the translation feature.
- Prior attempts are preserved in `.internal/work/active/translate-v2.md` (architecture) and `feature/translate` branch (implementation). Do not reference them during implementation — start clean.
- This document is the sole implementation roadmap for `feature/translate-v3`.
- App compliance rules are governed by `contracts/app-compliance-contract.md`.

---

## What Failed Before (and Why)

**V1** (feature/translate, SSE streaming): Sequential per-chunk LLM calls with no retries, no state persistence, heuristic stitching. Fragile — one failure killed the entire job.

**V2** (feature/translate, job-based): Correct failure model but wrong token budget (1K input instead of possible 6K), two implementations coexisting behind a feature flag (`TRANSLATE_V2_ENABLED`), terminology drift never solved, UI grew to 900+ lines managing a state machine instead of showing a document, no progressive rendering (user saw nothing until job fully complete).

**The root constraint nobody named explicitly:** Qwen3.6 35B Q4_K_M on Apple Silicon generates ~10–15 tokens/second. This is a physics ceiling — no architecture changes it. At 1K input tokens per chunk, a 50-page document required ~50 LLM calls. At 6K it requires ~8. The 6× throughput improvement was left on the table across both attempts.

---

## Honest Performance Expectations

These figures are based on measured eval results (2026-05-31) with `TRANSLATE_BATCH_TARGET_TOKENS = 1000`. Observed throughput ~45 tok/s (significantly faster than the model profile's conservative 5 tok/s estimate); actual ~17s/section average.

| Document size | Est. sections | Measured / est. time |
|---|---|---|
| 1–5 pages | 3–5 | ~1 min |
| 10 pages | 7 | ~2 min |
| 30 pages | 20 | ~6 min |
| 50 pages | 33 | ~9 min |
| 100 pages | 65 | ~18 min |
| Frankenstein (200K words) | 138 | 38 min — measured ✓ |
| 1000 pages | 650+ | ~3 hours — feasible but slow |

Time to first section: 22–41s (glossary + first LLM call). The UI must show a realistic estimate before the user starts. The soft-limit warning must work correctly — see Phase 8.

---

## Implementation Summary

| Phase | Scope | Status |
|-------|-------|--------|
| Phase 1 | Branch setup, `translate_policy.py`, DB schema, upload endpoint | ✅ COMPLETED (2026-05-31) |
| Phase 2 | Glossary extraction — pre-translation terminology call | ✅ COMPLETED (2026-05-31) |
| Phase 3 | Section-aware chunking — Docling section boundaries, not token windows | ✅ COMPLETED (2026-05-31) |
| Phase 4 | SSE job pipeline — section-level events, progressive emission | ✅ COMPLETED (2026-05-31) |
| Phase 5 | Frontend — combobox picker, progressive section rendering, no panel component | ✅ COMPLETED (2026-05-31) |
| Phase 6 | Time estimation, page count gate, promote out of dev flag | ✅ COMPLETED (2026-05-31) |
| Phase 7 | Glossary fix — restore `chat_complete` with calibrated timeout (fix now) | ✅ COMPLETED (2026-06-01) |
| Phase 8 | Page count investigation — why `page_count` is NULL, fix estimate + soft limit (fix now) | ✅ COMPLETED (2026-06-01) |
| Phase 9  | Shared composer infrastructure — `composer.css`, `composerSizing.ts`, `flag-icons` | ⏳ NOT STARTED |
| Phase 10 | Chat page rebuild — migrate ChatView to shared composer classes, verify zero regression | ⏳ NOT STARTED |
| Phase 11 | Translate page rebuild — full composer-based UI, streaming result, run header, footer | ⏳ NOT STARTED |
| Phase 12 | Files page translate integration — row icon, chip routing, `onTranslate` prop | ⏳ NOT STARTED |
| Phase 13 | Section titles for headerless documents — PDF/EPUB without Docling-detected headers | ⏳ NOT STARTED |

---

## Phase 1 — Branch Setup and Foundation ✅ COMPLETED (2026-05-31)

- **Goal:** Create `feature/translate-v3` from `main` with a clean policy module, DB schema, and upload/delete endpoints. No feature flags. No legacy code.
- **Scope:**
  - `git checkout main && git checkout -b feature/translate-v3`
  - `src/informity/translate_policy.py` — new file, all translation constants live here (none in route handlers inline):
    ```python
    TRANSLATE_PROVIDER              = 'translate.local'
    TRANSLATE_BATCH_TARGET_TOKENS   = 6000   # input tokens per section call
    TRANSLATE_CALL_MAX_TOKENS       = 9000   # output reserve (handles 40% expansion)
    TRANSLATE_GLOSSARY_INPUT_TOKENS = 2000   # first N tokens used for glossary extraction
    TRANSLATE_GLOSSARY_TERM_COUNT   = 25     # max terms to extract
    TRANSLATE_SECTION_RETRY_MAX     = 2      # retries per section before marking failed
    TRANSLATE_SECTION_TIMEOUT_S     = 120    # per-section LLM call timeout (seconds)
    TRANSLATE_JOB_STALL_S           = 90     # no progress for N seconds → stalled state
    TRANSLATE_JOB_MAX_RUNTIME_S     = 1800   # hard ceiling for any job (30 min)
    TRANSLATE_CLEANUP_AGE_HOURS     = 24     # sweep deletes translate.local files older than this
    TRANSLATE_SOFT_PAGE_LIMIT       = 50     # warn user; do not block
    TONE_INSTRUCTIONS: dict[str, str] = {
        'natural':  'Use natural, fluent {language}.',
        'literal':  'Translate as literally as possible, preserving sentence structure.',
        'formal':   'Use formal, professional register.',
    }
    ```
  - DB schema — two new tables (fresh, no legacy fields from feature/translate):
    - `translate_jobs`: `job_id` (UUID), `file_id`, `target_language`, `tone`, `output_mode`, `status` (queued/running/done/failed/stalled), `glossary_json`, `section_count`, `completed_sections`, `failed_sections`, `created_at`, `updated_at`, `error`
    - `translate_job_sections`: `section_id` (UUID), `job_id`, `section_index`, `section_title`, `status` (pending/running/done/failed), `result_text`, `attempt_count`, `created_at`, `updated_at`, `error`
  - `TRANSLATE_PROVIDER` added to all exclusion clauses that currently list `UPLOAD_PROVIDER` — retrieval, MCP tools — so `translate.local` files never leak into RAG context
  - `POST /api/translate/upload` — carries from v2: accepts file, runs Docling extract → chunk → embed → store with `source_provider='translate.local'`, returns `{file_id, filename, page_count, size_bytes}`
  - `DELETE /api/translate/upload/{file_id}` — removes file, chunks, and vectors for the given `translate.local` file
  - `src/informity/api/routes_translate.py` — single file, Phase 1 contains only upload/delete; translation job endpoints added in Phases 3–4
  - `main.py` — import + `include_router(routes_translate.router)`
- **Exit criteria:**
  - [ ] Branch exists from clean `main`
  - [ ] `translate_policy.py` has all constants; no magic numbers in route handler
  - [ ] DB migration creates both tables; existing tables unaffected
  - [ ] `TRANSLATE_PROVIDER` excluded from retrieval and MCP queries
  - [ ] Upload endpoint indexes a PDF under `translate.local`; confirmed via DB query
  - [ ] Delete endpoint removes file, chunks, and vectors cleanly
  - [ ] No feature flags anywhere in the codebase

---

## Phase 2 — Glossary Extraction ✅ COMPLETED (2026-05-31)

- **Goal:** Before translating any content, make one fast LLM call that extracts key terminology from the document. Store the glossary in the job record and inject it into every subsequent section prompt to eliminate cross-section terminology drift.
- **Scope:**
  - Called at job start, before the first section translation
  - Input: first `TRANSLATE_GLOSSARY_INPUT_TOKENS` tokens of the document's stored chunk text (fetched from DB, ordered by position)
  - Prompt (system): `"You are a professional translator preparing to translate a document to {target_language}. Extract up to {term_count} key terms, proper nouns, technical phrases, or domain-specific words that must be translated consistently. Return only valid JSON: [{\"source\": \"...\", \"translation\": \"...\"}]. No commentary."`
  - LLM call: `response_format={"type": "json_object"}` to enforce JSON output; timeout 30s; no retry (failure → proceed without glossary, log warning)
  - Result stored as `translate_jobs.glossary_json` (JSON string)
  - SSE event emitted: `{"event": "glossary_done", "data": {"term_count": N}}`
  - Glossary injected into every section prompt as:
    ```
    Terminology (use these translations consistently):
    - Annual Revenue → Jahresumsatz
    - Board of Directors → Vorstand
    ...
    ```
  - If glossary extraction fails or returns empty: proceed without it, job continues normally
- **Exit criteria:**
  - [ ] Glossary extracted and stored in `translate_jobs.glossary_json` before first section starts
  - [ ] SSE `glossary_done` event fires within 30s of job start
  - [ ] Glossary JSON is valid (parseable); invalid JSON → treated as empty, job continues
  - [ ] Every section prompt includes the glossary block when `glossary_json` is non-empty
  - [ ] Glossary failure does not fail the job

---

## Phase 3 — Section-Aware Chunking ✅ COMPLETED (2026-05-31)

- **Goal:** Replace arbitrary token-boundary chunking with chunking at Docling section boundaries. This eliminates stitching artifacts at section transitions and produces output that is structurally a valid translated document — not concatenated fragments.
- **Scope:**
  - Section detection uses `char_to_header_level_ranges` stored in the `chunks` table (set during indexing by the Docling extractor). This data already exists for all indexed files — no re-extraction needed.
  - Algorithm:
    1. Fetch all chunks for the file ordered by `page_number`, `position`
    2. Group consecutive chunks into sections: a new section starts when a chunk's `header_level` is 1 or 2, or when no header level data exists (treat as one section)
    3. Merge chunks within each section into a single text block (preserve `\n\n` paragraph boundaries)
    4. If a merged section exceeds `TRANSLATE_BATCH_TARGET_TOKENS` (6K): split at the nearest paragraph boundary, creating sub-sections; each sub-section is translated independently and their outputs concatenated without any LCS dedup (paragraph boundary is a clean split point)
    5. If no section header data exists (e.g., plain text file with no headers): fall back to token-window chunking at `TRANSLATE_BATCH_TARGET_TOKENS` with `\n\n` split preference — this is the only fallback path and is explicitly logged
  - Result: `translate_job_sections` rows created with `section_index`, `section_title` (header text if available, else `"Part N"`), and source text
  - `translate_jobs.section_count` set to total section count immediately after this step
  - SSE event: `{"event": "sections_ready", "data": {"section_count": N}}`
- **Technical notes:**
  - No LLM call in this phase — pure DB query + text grouping
  - Section count is what the user sees in the progress indicator ("Section 3 of 8"), not chunk count
  - The fallback token-window path must not silently activate — `log.info('translate_section_fallback', reason='no_header_data', file_id=..., section_count=N)`
- **Exit criteria:**
  - [ ] A document with Docling-detected headers produces one section per H1/H2 boundary
  - [ ] A section exceeding 6K tokens is split at paragraph boundary into sub-sections
  - [ ] A plain text file with no headers uses token-window fallback with a logged warning
  - [ ] `section_count` stored in job record before any translation begins
  - [ ] `sections_ready` SSE event fires with correct count

---

## Phase 4 — SSE Job Pipeline ✅ COMPLETED (2026-05-31)

- **Goal:** Implement the translation job worker that translates sections in order, emits each completed section immediately via SSE, and handles retries, timeouts, and terminal states deterministically.
- **Scope:**
  - **Endpoints (all in `routes_translate.py`):**
    - `POST /api/translate/jobs` — request: `{file_id, target_language, tone, output_mode}`; creates job record, kicks off background worker, returns `{job_id}`
    - `GET /api/translate/jobs/{job_id}` — returns current job status (polling fallback)
    - `GET /api/translate/jobs/{job_id}/events` — SSE stream of job events
    - `GET /api/translate/jobs/{job_id}/result?format=markdown|text` — export completed or partial result
    - `DELETE /api/translate/jobs/{job_id}` — cancel in-flight job, clean up
  - **SSE event sequence:**
    ```
    glossary_done    → {term_count}
    sections_ready   → {section_count}
    section_started  → {section_index, section_title}
    section_done     → {section_index, section_title, text}   ← includes translated text
    section_failed   → {section_index, attempt, error}
    job_done         → {section_count, completed_sections}
    job_failed       → {error}
    job_stalled      → {}
    ```
  - **`section_done` carries the translated text** in the SSE payload — the frontend renders it immediately without a separate fetch. This is the key difference from V2 where the frontend had to poll for results.
  - **Worker loop (per job):**
    1. Phase 2: glossary extraction
    2. Phase 3: section detection, create section rows
    3. For each section in order:
       - Set section `status = running`, emit `section_started`
       - Build prompt: system + glossary block + source text
       - Call LLM with `timeout=TRANSLATE_SECTION_TIMEOUT_S`
       - On success: store `result_text`, set `status = done`, increment `completed_sections`, emit `section_done` with text
       - On failure/timeout: increment `attempt_count`; if `attempt_count <= TRANSLATE_SECTION_RETRY_MAX`, retry with reduced token budget (cap source text at 2K tokens on retry); else set `status = failed`, increment `failed_sections`, emit `section_failed`, continue to next section
    4. After all sections: if any completed → emit `job_done`, set `status = done`; if all failed → set `status = failed`
    5. Stall watchdog: if no `section_done` event for `TRANSLATE_JOB_STALL_S` seconds, emit `job_stalled`, set `status = stalled`
    6. Hard ceiling: if total runtime exceeds `TRANSLATE_JOB_MAX_RUNTIME_S`, terminate worker, set `status = stalled`
  - **LLM concurrency:** `POST /api/translate/jobs` returns `409 Conflict` with `{"error": "llm_busy", "message": "..."}` if chat is actively streaming. The LLM engine runs one generation at a time — no exceptions. No concurrent sections within a job (V2's `TRANSLATE_V2_MAX_CONCURRENT_CHUNKS = 3` was misleading — calls queued sequentially anyway).
  - **Result export:** `GET /api/translate/jobs/{job_id}/result` assembles all `status = done` sections ordered by `section_index`, joins with `\n\n`, and returns as Markdown (default) or plain text (strips `#`/`**`/`_` markers via deterministic regex — not LLM post-processing)
  - **Translation prompt template:**
    ```
    System:
    You are a professional translator. Translate the following text to {target_language}.
    {tone_instruction}
    Preserve all Markdown formatting: headers (#, ##), bold (**), lists, tables.
    Output only the translated text. No commentary, no explanations, no preamble.

    [Terminology block if glossary non-empty]

    User:
    {section_source_text}
    ```
  - **Disable reasoning:** set `ReasoningMode.NEVER` for all translation LLM calls — no `<think>` tokens
- **Exit criteria:**
  - [ ] `POST /api/translate/jobs` creates job, returns `job_id` within 1s
  - [ ] SSE stream emits `glossary_done` → `sections_ready` → `section_done` × N → `job_done` in order
  - [ ] Each `section_done` event carries the translated text in the payload
  - [ ] A section that times out is retried once with reduced budget, then marked failed; job continues
  - [ ] All failed sections: job still emits `job_done` with `completed_sections < section_count`
  - [ ] `409` returned when chat is streaming
  - [ ] `GET .../result` returns Markdown joining all completed sections
  - [ ] A 10-page PDF translates end-to-end in under 4 minutes

---

## Phase 5 — Frontend ✅ COMPLETED (2026-05-31)

- **Goal:** A single focused page, under 300 lines, with no separate panel component. File selection via combobox, not a file table. Translated sections appear progressively as SSE events arrive.
- **Scope:**
  - **Files created:** `TranslatePage.tsx`, `TranslatePage.css`
  - **Files deleted:** `TranslatePanel.tsx`, `TranslatePanel.css` (logic merged into page)
  - **`TranslatePage` layout — three zones, no scroll required before interacting:**

    **Zone 1 — File picker (top):**
    - Searchable combobox: user types, fires `GET /api/files?search=...&limit=15`, shows results as dropdown options. Combobox label: "Search library files..."
    - Below combobox: drag-and-drop zone (`POST /api/translate/upload`). Shows "or drop a file here" when no file selected. Accepts same extensions as the translate upload endpoint.
    - Selected file shown as a dismissible chip: `[filename.pdf  ×]`. Dismissing calls `DELETE /api/translate/upload/{file_id}` if it was an upload, or just clears selection if it was a library file.

    **Zone 2 — Controls (one row, below file picker):**
    - `[Language ▾]` — dropdown, 20 common languages, default from `translate_default_language` setting
    - `[Tone ▾]` — Natural / Formal / Literal
    - `[Translate →]` — `settings-btn--primary`; disabled when no file selected or LLM busy
    - When translating: button becomes `[Cancel]` (`settings-btn--secondary`); language and tone selectors disabled

    **Zone 3 — Result area (fills remaining height):**
    - **Sticky toolbar at top of result area** (always visible regardless of scroll):
      - Left: `"Section 3 of 8  ·  ~12 min remaining"` in `var(--color-text-muted)` (or `"Translation complete"`)
      - Right: `[Copy]` `[Save .md]` `[Save .txt]` — Copy always active once any section exists; Save actions active when job is done; `[Cancel]` replaces Save while translating
    - **Result body:** each `section_done` SSE event appends a rendered section. Sections separated by a subtle `<hr>` line. Markdown rendered via `react-markdown` (same library used in chat messages). Section header (`##`) rendered as a styled heading, not raw `##`.
    - **Empty state** (no job yet): `<CenteredState icon="ri-translate-2" title="Select a document to translate" description="..." />`
    - **LLM busy state:** `<CenteredState>` with message `"LLM is in use by chat."` and a `[Stop Chat]` button that calls `stopStreaming()` from `ChatProvider`
    - **Failed section indicator:** between sections, a subtle inline notice: `"Section 4 could not be translated — skipped."` Does not interrupt rendering of surrounding sections.

  - **`TranslateProvider` state (carry from v2, simplify):**
    - `jobId: string | null`
    - `status: TranslateJobStatus | null`
    - `sections: Array<{index, title, text, status}>` — populated as `section_done` events arrive
    - `sectionCount: number | null`
    - `completedSections: number`
    - `glossaryTermCount: number | null`
    - `estimatedMinutes: number | null`
    - `targetLanguage: string`
    - `tone: string`
    - `fileId: number | null`
    - `fileName: string | null`
  - **State survival:** `TranslateProvider` wraps the app — navigating to Chat and back restores the result view completely from provider state. If the job is still running, SSE reconnects automatically.
  - **Copy behaviour:** copies all completed sections joined with `\n\n` as plain text
  - **Sidebar spinner:** when `status === 'running'`, the Translate nav item shows the spinner (same pattern as Chat and Dashboard)
  - **Design system compliance:**
    - Buttons: `settings-btn settings-btn--primary` / `settings-btn--secondary` only
    - Spacing: `var(--space-*)` exclusively
    - Colors: `var(--color-*)` exclusively — no hardcoded hex
    - Result area uses `react-markdown` for rendering (consistent with chat, no new deps)
    - Combobox follows `var(--color-border)`, `var(--border-radius-md)` patterns from existing inputs

- **Exit criteria:**
  - [ ] `TranslatePage.tsx` under 300 lines; `TranslatePage.css` under 80 lines
  - [ ] `TranslatePanel.tsx` and `TranslatePanel.css` deleted
  - [ ] Combobox searches library files as user types; results appear as dropdown
  - [ ] Drag-drop zone uploads a file and creates the chip; chip dismissal calls delete endpoint
  - [ ] All controls fit in one visible row without scrolling
  - [ ] `section_done` SSE events render sections progressively in the result area
  - [ ] Sticky toolbar shows correct section count and Copy/Save/Cancel at all times
  - [ ] Failed section shows inline notice; surrounding sections render normally
  - [ ] Navigating to Chat and back preserves in-progress or completed result
  - [ ] Sidebar spinner active when job is running
  - [ ] No custom hex values, raw pixel values, or non-token spacing in CSS

---

## Phase 6 — Time Estimation, Page Gate, and Promote ✅ COMPLETED (2026-05-31)

- **Goal:** Show the user a realistic time estimate before they commit to a translation; warn (not block) on large documents; remove the dev-only flag.
- **Scope:**
  - **Time estimation:**
    - When file is selected, `POST /api/translate/jobs/estimate` (lightweight, no LLM call): returns `{page_count, section_count_estimate, estimated_minutes, exceeds_soft_limit: bool}`
    - Backend calculates: `section_count ≈ ceil(page_count * avg_tokens_per_page / TRANSLATE_BATCH_TARGET_TOKENS)`, `estimated_minutes ≈ section_count * avg_section_seconds / 60`
    - `avg_tokens_per_page = 650` (standard estimate), `avg_section_seconds = 90` (conservative)
    - Displayed below the controls row: `"~12 min for 50 pages"` in `var(--color-text-muted)`; hidden when no file selected
  - **Soft page limit warning:**
    - When `exceeds_soft_limit` is true (> `TRANSLATE_SOFT_PAGE_LIMIT = 50` pages): show inline warning below estimate: `"Large document (~120 pages). Translation may take ~36 minutes. You can still proceed."`
    - Translate button remains enabled — user decides
    - Warning styled with `var(--color-warning-text)` and `var(--color-warning-border)` — not an error state
  - **Promote out of dev flag:**
    - Remove `import.meta.env.DEV` guard from sidebar nav item and lazy import in `App.tsx`
    - Translate becomes a permanent nav entry
  - **24-hour cleanup sweep:**
    - Background task (runs at startup and every 6 hours, alongside existing maintenance) deletes `translate.local` files with `indexed_at < now - TRANSLATE_CLEANUP_AGE_HOURS`
    - Logs count of deleted files/chunks for observability
    - Also deletes associated `translate_jobs` and `translate_job_sections` rows
  - **Default language setting:**
    - New `translate_default_language: str = "Spanish"` in `config.py`
    - Language picker pre-selects this value on page load
    - Exposed in Settings UI alongside other preferences
- **Exit criteria:**
  - [ ] Estimate appears within 500ms of file selection (no LLM call — pure math)
  - [ ] Warning appears for files > 50 pages; Translate button remains enabled
  - [ ] Translate nav item visible in production build (`npm run build` confirmed)
  - [ ] Cleanup sweep deletes records older than 24h; count logged
  - [ ] Default language from Settings pre-selects the picker

---

## Phase 7 — Glossary Fix ✅ COMPLETED (2026-06-01)

- **Goal:** Restore working glossary extraction. Switching to `generate_stream` in the pipeline fix broke the glossary because `generate_stream` does not support `response_format={"type":"json_object"}`, so Qwen3 never produces reliable JSON. Eval confirmed: 0 glossary terms extracted for all 5 files.
- **Root cause:** The original glossary used `asyncio.to_thread(chat_complete)` with a 30s timeout. At 5 tok/s with 512 max tokens, the model needed 102s to finish — timeout fired at 30s, leaving a zombie thread that blocked the first section call. The fix at the time was to switch everything to `generate_stream`, which broke JSON mode.
- **Correct fix:** Switch the glossary back to `chat_complete`. It is now safe because `TRANSLATE_GLOSSARY_TIMEOUT_S = 150s` and `TRANSLATE_GLOSSARY_MAX_TOKENS = 512` — at 45 tok/s observed speed, generation completes in ~11s; even at the conservative 5 tok/s profile estimate, 512/5 = 102s < 150s. No zombie thread occurs when the model finishes before the timeout.
- **Scope:**
  - In `_extract_glossary()`: switch back to `asyncio.to_thread(llm_engine.chat_complete, messages, TRANSLATE_GLOSSARY_MAX_TOKENS, TRANSLATE_GLOSSARY_TEMPERATURE, None, {"type":"json_object"})` with `asyncio.wait_for(..., timeout=TRANSLATE_GLOSSARY_TIMEOUT_S)`
  - Keep section translation on `generate_stream` — it has real cancel semantics and is correct for long outputs
  - File: `src/informity/api/routes_translate.py` (`_extract_glossary` function only)
- **Exit criteria:**
  - [ ] Glossary extraction returns ≥ 10 terms for the CISA DOCX (a well-structured English document)
  - [ ] Glossary block appears in section translation prompts (loggable at debug level)
  - [ ] No zombie thread: next section starts within 5s of glossary completion
  - [ ] Job still completes if glossary returns 0 terms (best-effort, not required)

---

## Phase 8 — Page Count Investigation ✅ COMPLETED (2026-06-01)

- **Goal:** Determine why `page_count` is NULL for all files in the eval, and fix the estimate endpoint and soft-limit warning to work correctly. Without page count, users get no time estimate and no large-document warning.
- **Scope:**
  - **Investigate:** Query DB directly to confirm whether `page_count` is stored in the `files` table for indexed PDFs and DOCXs. If stored in DB but not in API response, fix `GET /api/files` serialization. If not stored, fix the Docling extractor to write it.
  - **Check:** Does the `GET /api/files` response include `page_count` in the JSON? Check the `IndexedFile` serialization in `routes_scan.py` or wherever files are serialized.
  - **Check:** Is `page_count` being written to the DB during indexing? Look at `db/sqlite.py`'s file upsert and the Docling extractor's `page_count` extraction (lines 357–370 of `docling.py`).
  - **Fix:** Wherever the gap is, ensure `page_count` is populated for PDF, DOCX, and PPTX files (formats where Docling reliably detects page count). EPUB and HTML are acceptable as NULL.
  - **Estimate accuracy:** Once `page_count` is populated, verify the estimate endpoint returns sensible `estimated_minutes` for the 5 eval files. Recalibrate `TRANSLATE_AVG_SECTION_SECONDS` in `translate_policy.py` if needed (measured actual is ~17s/section, current config value is 240s — wildly pessimistic).
- **Exit criteria:**
  - [ ] `page_count` is non-NULL in the API response for all PDF and DOCX files that were previously showing `— ` in the eval
  - [ ] Estimate endpoint returns a time within 50% of actual for the 5 eval files
  - [ ] Soft-limit warning fires correctly for a document with > 50 pages
  - [ ] `TRANSLATE_AVG_SECTION_SECONDS` updated to reflect measured performance (~20s is a reasonable conservative value vs the current 240s)

---

## Phase 9 — Shared Composer Infrastructure ⏳ NOT STARTED

- **Goal:** Build the shared CSS and JavaScript foundation that both Chat and Translate pages will use. Nothing is shipped visually yet — this phase creates the building blocks that Phases 10 and 11 consume.
- **Scope:**
  - **`src/frontend/src/styles/shared/composer.css`** — shared structural classes used verbatim by both pages:
    - `.composer` — base: max-width constraint (`--page-content-width`), centered vertically and horizontally in the page when in default state
    - `.composer--docked` — modifier added on first interaction: transitions the composer to the bottom of the viewport; content area builds above
    - `.composer__textarea` — identical sizing, padding, border-radius, font-size, shadow, and resize behavior in both pages; placeholder color via `var(--color-text-muted)`
    - `.composer__chip-row` — flex row above the textarea for attached file chips; hidden when empty
    - `.composer__chip` — individual chip: file icon + truncated name + × dismiss button; same height and padding in both pages
    - `.composer__controls` — bottom flex row: layout only (flex, gap, align); does not prescribe what controls live inside (Chat puts toggles here; Translate puts selects here)
    - `.composer__send` — primary action button: same size, border-radius, accent color, disabled state, and hover state in both pages
    - `.composer__warning` — soft inline warning (e.g. "Large document — ~38 min estimated"): `var(--color-warning-text)` color, same in both pages
    - All spacing via `var(--space-*)`, all colors via `var(--color-*)`, all radii via `var(--border-radius-*)` — zero raw values
  - **`src/frontend/src/utils/composerSizing.ts`** — shared JavaScript utility:
    - `computeScopedTopPadding(wrapperEl, chipRowEl, baseOffset)` — measures chip row height in real DOM, returns the px offset the textarea needs to avoid overlap; called from both Chat and Translate on chip add/remove
    - `applyComposerDocked(composerEl)` — adds `.composer--docked` and handles the transition cleanly, called once per page on first send/translate
    - Both functions are pure DOM utilities — no React, no page-specific logic
  - **`flag-icons@^7.5.0`** — add to `package.json`; renders as `<span className="fi fi-{code}" />` in the language selector; no emoji fallback needed
  - **ChatView.css improvements** cherry-picked from `feature/translate`:
    - Control background contrast: `color-mix(... 70%, white)` → `84%`
    - Foreground text: from `color-text-muted 60%` → `color-text 76%` (higher readability)
    - New `--chat-view-control-border` token (explicit border on control buttons)
    - Send button hover state (currently missing — real bug fix)
    - Dropdown shadow: `var(--shadow-md)` → `var(--shadow-menu)` (semantically correct)
  - **`useOptionalTranslateContext()`** — non-throwing variant of `useTranslateContext` that returns `null` when called outside the provider; needed by Sidebar so it can render safely before provider initializes
- **Exit criteria:**
  - [ ] `composer.css` exists; `TranslatePage.css` and `ChatView.css` import it
  - [ ] All classes use only `var(--*)` tokens — no raw values, no hex
  - [ ] `composerSizing.ts` exports both functions; unit-testable without DOM (pure calculations)
  - [ ] `flag-icons` installed; a test render of `<span className="fi fi-es" />` shows the Spanish flag
  - [ ] ChatView.css contrast and hover improvements applied; Chat UI visually unchanged (no regression)
  - [ ] `useOptionalTranslateContext()` exported from `useTranslateContext.ts`

---

## Phase 10 — Chat Page Rebuild with Shared Classes ⏳ NOT STARTED

- **Goal:** Migrate the Chat page to use the shared `.composer` classes from Phase 9. The chat composer's behavior must be identical after migration — this is a refactor with zero intended behavior change. It also validates that `composer.css` is complete enough to support a real page before Translate is built on top of it.
- **Scope:**
  - Replace ChatView's bespoke textarea, chip, controls, and send-button CSS with the shared `.composer` classes
  - Retain all ChatView-specific CSS that has no equivalent in Translate (message rendering, source cards, streaming indicators, etc.) — only the composer area is migrated
  - Call `computeScopedTopPadding()` from `composerSizing.ts` wherever ChatView currently computes chip offset dynamically
  - Call `applyComposerDocked()` when the first message is sent (replacing any equivalent inline logic)
  - Sidebar Translate nav item uses `useOptionalTranslateContext()` to show spinner without throwing
  - No changes to ChatProvider, ChatPage routing, or any non-composer Chat component
- **Exit criteria:**
  - [ ] Chat page passes visual spot-check: empty composer centered, grows upward with file chips, slides to bottom on first message — identical to pre-migration behavior
  - [ ] No new CSS variables or hardcoded values introduced in ChatView.css
  - [ ] `npm run typecheck` and `npm run lint` pass
  - [ ] `npm run test` passes (existing Chat tests unaffected)
  - [ ] Side-by-side comparison of Chat before and after migration shows no visual difference

---

## Phase 11 — Translate Page Rebuild ⏳ NOT STARTED

- **Goal:** Replace the current `TranslatePage.tsx` (Phase 5, never visually tested, combobox-based layout) with a full implementation using the shared `.composer` classes. The result should be visually and behaviourally consistent with Chat: same centered-to-docked transition, same chip pattern, same streaming display above the composer.
- **Scope:**

  **Composer (bottom):**
  - `.composer` centered on mount; transitions to `.composer--docked` after first Translate
  - `.composer__chip-row` / `.composer__chip` — selected file chip (filename + × dismiss button); appears above textarea when a file is selected; `computeScopedTopPadding()` adjusts textarea to not overlap
  - `.composer__textarea` — optional steering prompt field; placeholder: `"Steer translation (optional) — e.g. focus on methodology, skip references…"`; thin and secondary by default (shorter min-height than Chat textarea), same border and font
  - `.composer__controls` — language `<select>` (with `flag-icons` flag + language name), tone `<select>` (Natural / Formal / Literal), Translate / Cancel `.composer__send` button
  - File source: two entry points, both produce a chip:
    - **Library search** — a search-as-you-type input (`GET /api/files?search=...`) showing file results in a dropdown; selecting one creates the chip
    - **Drag-and-drop / click-to-upload** — same `POST /api/translate/upload` endpoint as before; creates chip on success
  - Large-document estimate warning: `.composer__warning` appears below controls when `exceeds_soft_limit` is true — `"Large document (~177 pages) · ~39 min estimated"` — same visual treatment as any other in-line warning

  **Result area (above composer):**
  - Each translation run is a discrete block; blocks accumulate as user re-translates with different settings
  - **Run header** (subtle, one line): `Q3_Report.pdf → Spanish · Natural · "focus on methodology"` in `var(--color-text-muted)`, `var(--font-size-xs)` — provides context without dominating
  - **Sections stream in** like assistant replies: full-width, markdown rendered, no message bubbles; streaming cursor (blinking `▋`) at the active section's end
  - **Progress line** (inline, below active section): `Section 4 of 13 · 1m 22s elapsed` — updates as sections complete; disappears on job completion
  - **Run footer** (same visual as Chat message footer): `Copy · Save .md · Save .txt` on the left; `Spanish · Natural · 5m 31s` metadata on the right; appears once job status is `done`; partial export (Copy/Save) available from first completed section even mid-translation
  - **LLM busy state**: if `isStreaming` is true from ChatProvider, composer shows `"LLM in use by chat"` and Translate button shows a "Stop Chat" link action; once streaming ends, Translate becomes available
  - Navigation away mid-translation: `TranslateProvider` holds state, user returns to see in-progress result unchanged

  **Sidebar:**
  - Translate nav item positioned between Chat and Files (natural hierarchy)
  - Spinner visible when `isTranslating` from `TranslateProvider`
  - Uses `ri-translate-2` (or `ri-translate-ai-2` if confirmed available in current Remix Icons version)

  **Page structure:**
  - Delete current `TranslatePage.tsx` and `TranslatePage.css` (Phase 5 implementation)
  - New `TranslatePage.tsx` ≤ 250 lines — all layout comes from `composer.css`
  - No bespoke layout CSS in the new `TranslatePage.css` — only page-specific overrides for translate-unique elements (run header, run footer, result block spacing)

- **Exit criteria:**
  - [ ] Empty state: composer centered, visually identical to Chat empty state (spot-checked side by side)
  - [ ] File chip: appears above textarea on file selection; textarea adjusts height correctly via `computeScopedTopPadding()`
  - [ ] First Translate: composer slides to bottom; sections stream in above with markdown rendering
  - [ ] Run header visible above each run; correct file/language/tone/steering shown
  - [ ] Progress line updates during translation; disappears on completion
  - [ ] Run footer appears on completion with Copy, Save .md, Save .txt, metadata
  - [ ] Re-translate with different settings appends a new run block above
  - [ ] Steering textarea: typed text is included in translation prompt (backend receives it)
  - [ ] LLM busy state displayed and resolved correctly
  - [ ] Large-document warning fires for files > 50 pages
  - [ ] `TranslatePage.tsx` ≤ 250 lines; `TranslatePage.css` contains no layout rules (all in `composer.css`)
  - [ ] Navigation away mid-translation and return: state preserved
  - [ ] Sidebar spinner active during translation

---

## Phase 12 — Files Page Translate Integration ⏳ NOT STARTED

- **Goal:** Make the Files page a natural entry point for translation. A user browsing indexed files can send any file directly to the Translate page with one click, arriving with the file pre-loaded as a chip in the composer.
- **Scope:**
  - **`FileTable.tsx`**: add optional `onTranslate?: (file: IndexedFile) => void` prop (already partially done in earlier work; confirm backwards-compatible — existing `FilesPage` usage passes no `onTranslate`, so no translate icon renders)
  - **Translate icon in file row**: appears when `onTranslate` is provided; positioned after the Chat icon in the action column; uses `ri-translate-2` (or `ri-translate-ai-2`); tooltip: `"Translate this file"`; column width adjusted to accommodate
  - **`FilesPage.tsx`**: passes `onTranslate` handler that calls `navigate('/translate', { state: { scopedFileId, scopedFileName, scopedFileSourceProvider: 'filesystem' } })`
  - **`FileDetail`** modal retained unchanged — translate is an addition, not a replacement for the detail view
  - **`TranslatePage.tsx`**: on mount, reads `location.state`; if `scopedFileId` is present, immediately calls `setFile()` to pre-load the chip and trigger the estimate — user lands on Translate with file already attached, ready to configure and go
  - **Sidebar icon position**: Translate nav item sits between Chat and Files — reinforces that translate is discoverable from Files
- **Exit criteria:**
  - [ ] Translate icon visible in FileTable when `onTranslate` is provided; absent in FilesPage without prop (confirm no visual regression on Files page)
  - [ ] Clicking translate icon on any file row navigates to `/translate` with correct router state
  - [ ] Translate page mounts with file chip pre-loaded (no additional user action required)
  - [ ] Estimate appears immediately after chip loads
  - [ ] FileDetail modal still opens on row click (translate icon click does not trigger it — `stopPropagation` correctly applied)
  - [ ] No changes to FileDetail, FileFilters, or any other Files component

---

## Phase 13 — Section Titles for Headerless Documents ⏳ NOT STARTED

- **Goal:** Improve the run header and progress indicator for documents where Docling detected no section headers (academic PDFs, EPUBs). Currently all sections show `(untitled)` in the run header and progress line.
- **Context:** Section titles come from `section_path` in the `chunks` table. NULL `section_path` → `(untitled)` fallback. Translation quality is unaffected; this is a cosmetic improvement for the result display.
- **Options (evaluate in order of cost):**
  1. **Page-number label** — derive from `page_number` of first chunk in section: `"Page 4"`. Zero new logic; uses existing DB field.
  2. **First-sentence label** — take first ≤ 60 chars of source text. Cheap, deterministic, no LLM call.
  3. **Accept `(untitled)`** — if section labels are subtle enough in the UI, this may not be worth doing at all.
- **Scope:** Whichever option is chosen, the change lives in `_path_to_title()` in `_build_sections()` (`routes_translate.py`) and propagates automatically to `section_title` in `translate_job_sections` rows and SSE events.
- **Priority:** Low — do after Phases 9–12.
- **Exit criteria:**
  - [ ] ML paper (13 untitled sections) shows meaningful labels — page numbers or first-sentence snippets
  - [ ] CISA DOCX (named sections) unchanged
  - [ ] No LLM calls added

---

## Implementation Notes

### Token budget — calibrated values (as of 2026-05-31)

```
Model profile observed speed:  ~45 tok/s (much faster than profile's 5 tok/s estimate)
Input tokens per section:      1,000  (TRANSLATE_BATCH_TARGET_TOKENS)
Output expansion (Spanish):    ~1.35×  → ~1,350 output tokens
Generation time per section:   1,350 / 45 ≈ 30s  (measured: 12–26s/section)
Section timeout:               360s  (generous headroom; actual never needed)
```

The original design assumed 12 tok/s. Post-eval reality is ~45 tok/s on this hardware.
`TRANSLATE_AVG_SECTION_SECONDS = 240` in `translate_policy.py` is wildly pessimistic and
should be updated in Phase 8 to ~20s once page_count investigation is complete.

### Token budget (original design — why 6K is correct)

```
System prompt:        ~200 tokens
Glossary block:       ~400 tokens  (25 terms × ~16 tokens/term)
Source section text:  6,000 tokens  ← target input
Output reserve:       9,000 tokens  ← handles 40% expansion (Finnish, German)
Safety margin:          100 tokens
─────────────────────────────────────
Total:               ~15,700 tokens  ✓ fits 16K context
```

V2 used 1,000 tokens per chunk. This single change reduces chunk count by 6× and is the largest single improvement available.

### Why sections, not token windows

Token-window chunking cuts mid-sentence, mid-paragraph, mid-table. Every boundary requires stitching heuristics (LCS dedup in V2). Docling already detected section boundaries during indexing and stored them in `char_to_header_level_ranges` per chunk. Using those boundaries means:
- Chunks align with `##` headings — natural break points
- Translated output is structurally valid Markdown from the start
- No stitching required — sections are concatenated with `\n\n`
- `section_done` SSE events map to visible document sections the user recognises

### Why one section at a time (no concurrency)

V2 ran up to 3 concurrent chunks via a semaphore, but the Qwen engine is single-instance — concurrent calls queued sequentially. The concurrency setting gave a false impression of parallelism while adding state management complexity. Sections run strictly in order: this gives correct progressive rendering (section 1 → 2 → 3 in document order), predictable memory usage, and a simpler worker loop.

### Glossary call specifics

- Model call: same LLM, same engine interface
- `response_format={"type": "json_object"}` enforces JSON output (supported by Qwen3 via the engine's structured output mode)
- Max output tokens: 500 (25 terms × ~16 tokens each, with room)
- Temperature: 0.1 (more deterministic than translation calls)
- Timeout: 30 seconds
- On any failure (timeout, parse error, empty result): log warning, set `glossary_json = null`, proceed with translation — glossary is best-effort, not required

### What to carry from `feature/translate` vs drop

| Keep | Drop |
|------|------|
| `TRANSLATE_PROVIDER = 'translate.local'` | `TRANSLATE_V2_ENABLED` feature flag |
| Upload/delete endpoint logic | V1 streaming endpoint (`POST /api/translate`) |
| `TranslateProvider` context shape (update fields) | `TranslatePanel.tsx` / `TranslatePanel.css` |
| DB job model concept (new schema, clean fields) | V2 chunk tables (`translate_job_chunks`) |
| SSE event pattern | LCS stitching logic (`_trim_duplicate_boundary`) |
| `translate.local` exclusion from retrieval/MCP | `_to_plain_text_translation()` post-processor |
| Sidebar spinner pattern | Chat ID upload tab |

### Files touched per phase

**Phase 1 (new):** `src/informity/translate_policy.py`, `src/informity/api/routes_translate.py`, `src/informity/db/sqlite.py` (migration + helpers), `src/informity/main.py` (2-line router registration)

**Phase 1 (modified):** `src/informity/llm/retrieval.py` (add `TRANSLATE_PROVIDER` to exclusion), `src/informity/mcp/tools_readonly.py` (same)

**Phase 2–4 (routes_translate.py extended):** glossary extraction function, section detection function, job worker, SSE stream endpoint, result export endpoint

**Phase 5 (new):** `src/frontend/src/pages/TranslatePage.tsx`, `src/frontend/src/pages/TranslatePage.css`, `src/frontend/src/context/TranslateProvider.tsx`, `src/frontend/src/context/useTranslateContext.ts`

**Phase 5 (modified):** `src/frontend/src/components/Sidebar.tsx`, `src/frontend/src/App.tsx`

**Phase 5 (deleted):** `src/frontend/src/components/translate/TranslatePanel.tsx`, `src/frontend/src/components/translate/TranslatePanel.css`

**Phase 6 (modified):** `src/informity/config.py` (new setting), `src/informity/api/routes_translate.py` (estimate endpoint), `src/frontend/src/App.tsx` (remove DEV guard), `src/frontend/src/pages/TranslatePage.tsx` (estimate display + warning)

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Section > 6K tokens (very long sections) | Split at paragraph boundary; sub-sections translated independently and concatenated; no stitching needed at paragraph level |
| Glossary JSON parse failure | Treat as empty; job continues without glossary; logged warning |
| LLM refuses to output only translated text | Prompt includes strict `"No commentary"` instruction; if output contains preamble, strip first line if it matches `"Here is the translation"` pattern (deterministic, non-semantic — allowed under compliance contract) |
| User starts translation on 500-page doc | Soft limit warning shown at file selection; estimate displayed; user proceeds knowingly |
| Chat/translate LLM collision | Double-gated: frontend disables Translate when `isStreaming`; backend returns 409 |
| translate.local files accumulate | Explicit delete on chip dismiss + 24h background sweep |
| Section fails after 2 retries | Marked failed, skipped with inline notice in UI; surrounding sections unaffected; partial export always available |
| Markdown rendering inconsistency | `react-markdown` (already used in chat); no new dependency |

---

## Non-Goals

- 1000-page documents in minutes — physically impossible with local inference; soft limit + honest estimate instead
- Translation memory or cross-session caching — deferred
- Chat-response translation ("translate this answer") — separate feature, not in this branch
- Concurrent section translation — false parallelism with single-LLM engine; dropped
- Natural language command input ("now translate to French") — deferred to future iteration

---

## Success Criteria

- A 10-page document translates in under 3 minutes and the user sees the first section within 45 seconds of clicking Translate.
- A 50-page document produces a well-formatted Markdown document in under 20 minutes; the user sees progress throughout.
- Key terminology is consistent across all sections.
- The translated Markdown renders correctly in any Markdown viewer (headers, tables, lists intact).
- `TranslatePage.tsx` is under 300 lines. `routes_translate.py` is under 400 lines. No feature flags anywhere.
- Navigating away and back during active translation does not interrupt the job or lose the result.
