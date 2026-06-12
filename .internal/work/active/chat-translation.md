# Informity AI — Chat Translation

**Status:** ✅ COMPLETED  
**Version:** 1.0  
**Last updated:** 2026-06-11  
**Scope:** Add in-thread translation of assistant chat replies without breaking existing Chat or Translate behavior, while keeping the result persisted as part of chat history and aligned with the broader translation lifecycle used by Translate.

---

## Goal

Let users translate any assistant reply directly from Chat and have the translated result live in the same conversation thread, with full reload/history support. The desired outcome is a first-class chat action that reuses existing translation behavior and settings, preserves the original answer, and stores translated variants as normal chat history entries instead of a separate translation-only persistence path.

---

## Document Role

- This document is the implementation roadmap for chat-internal message translation.
- It must remain app-contract compliant and preserve existing Chat and Translate functionality.
- Contract rules remain normative in:
  - `.internal/contracts/app-compliance-contract.md`
  - `.internal/contracts/chat-output-contract.md`

---

## App Compliance

- Translation must be implemented as a normal app flow, not as a prompt-only chat guess.
- Existing chat answer semantics must remain unchanged for non-translation messages.
- Existing file translation behavior must remain unchanged.
- Translation output must not bypass the existing output contract or introduce unsupported message formats.
- Any new metadata must be additive and backward-compatible.
- No hidden side channels, alternate persistence paths, or corpus-specific hacks.

---

## Implementation Summary

| Phase | Scope | Status |
|---|---|---|
| Phase 1 | Backend translation command and chat message metadata | ✅ COMPLETED |
| Phase 2 | Chat reply translation action and rendering | ✅ COMPLETED |
| Phase 3 | History reload, edits, and lifecycle behavior | ✅ COMPLETED |
| Phase 4 | Chat translation session persistence and reload recovery | ✅ COMPLETED |
| Phase 5 | Shared translation lifecycle across Chat + Translate | ✅ COMPLETED |
| Phase 6 | Validation, regression coverage, and rollout guardrails | ✅ COMPLETED |

---

## Phase 1 — Backend Translation Command and Chat Message Metadata ✅ COMPLETED

- Goal: add a backend path that translates a specific assistant reply into a selected language/tone and stores the translated result as a chat message in the same thread.
- Scope:
  - Define a message-level translation request contract.
  - Reuse the existing translation model/prompt policy from `src/informity/api/routes_translate.py` and `src/informity/translate_policy.py`.
  - Add additive metadata to chat messages for translation provenance, such as:
    - source message id
    - target language
    - tone
    - translation kind/version
    - stale/source-version marker when the source message changes later
  - Persist translated output in chat history as a normal assistant-visible message, not as a separate translation-only artifact.
  - Keep the original message untouched.
  - Do not expose a second translate action on translated messages.
- Exit criteria:
  - [x] Backend can translate a specific assistant reply by id.
  - [x] Translation metadata is stored with the chat history entry.
  - [x] Existing chat generation and file translation APIs remain unchanged.
  - [x] Result reloads cleanly from chat history.
  - [x] Translated messages remain distinct from source messages and do not chain-translate by default.

---

## Phase 2 — Chat Reply Translation Action and Rendering ✅ COMPLETED

- Goal: expose a translate action in the assistant reply footer and render translated messages clearly but inline with the same chat thread.
- Scope:
  - Add a reply-footer translate icon/action for assistant messages only.
  - Reuse the existing translation controls pattern where practical:
    - language selection
    - tone selection
    - start/confirm action
  - Render translated output as a chat message linked to the source assistant reply and keep the source visible.
  - Do not show translate controls under translated messages.
  - Make the translation visually identifiable without making it feel like a separate product surface.
- Exit criteria:
  - [x] Users can initiate translation directly from an assistant reply.
  - [x] Translation appears in the same chat thread.
  - [x] Original content remains visible.
  - [x] Translated messages do not expose a chained translation control.
  - [x] UI works without affecting normal chat interactions.

---

## Phase 3 — History Reload, Edits, and Lifecycle Behavior ✅ COMPLETED

- Goal: make translated replies survive reloads and behave predictably when the source conversation changes.
- Scope:
  - Ensure translated replies are included in chat history fetch/render flow.
  - Define behavior when:
    - the source reply is edited/regenerated,
    - the source message later changes and translations should be marked stale,
    - the same original reply is translated more than once.
  - Stack translations under the original message instead of replacing it.
  - Preserve existing chat navigation, history, export, and continuation behavior.
- Exit criteria:
  - [x] Reloading a chat shows translated replies in the expected place.
  - [x] Source/edit/regenerate interactions do not corrupt translations.
  - [x] Existing history behavior is unchanged for chats without translations.
  - [x] Stale translations are clearly identifiable without being deleted automatically.

---

## Phase 4 — Chat Translation Session Persistence and Reload Recovery ✅ COMPLETED

- Goal: preserve an in-flight chat translation request across navigation/reload enough to restore the in-thread status and complete the translation again if needed.
- Scope:
  - Persist the active chat translation request in session storage while it is running.
  - Reconstruct the in-thread translation placeholder after a reload when the source reply is still present.
  - Retry the same translation request on recovery so the backend can either reuse the completed result or finish the in-flight one.
  - Clear the pending request when the translation completes, fails, or the user starts a new chat.
- Exit criteria:
  - [x] A translation can reload cleanly while in progress.
  - [x] The same translation is visible consistently after navigation/reload.
  - [x] Chat and Translate continue to behave independently where they should.
  - [x] No new persistence path is introduced outside the shared app model.

---

## Phase 5 — Shared Translation Lifecycle Across Chat + Translate ✅ COMPLETED

- Goal: standardize how long-running translation jobs behave across Chat and Translate so users see a consistent in-progress, reload, and completion experience.
- Scope:
  - Reuse the same job lifecycle signals used by Translate where practical:
    - queued
    - running
    - completed
    - failed
  - Keep the in-thread status message and the final translated chat reply in sync with the stored job state.
  - Ensure the chat translation flow remains app-contract compliant and does not create a separate persistence path.
- Exit criteria:
  - [x] Chat and Translate use a consistent lifecycle vocabulary and recovery model.
  - [x] Shared status UI behavior is defined across both surfaces.
  - [x] No new persistence path is introduced outside the shared app model.

---

## Phase 6 — Validation, Regression Coverage, and Rollout Guardrails ✅ COMPLETED

- Goal: validate that chat translation does not regress chat, translate, or settings flows.
- Scope:
  - Add focused tests for footer translation initiation and history reload behavior.
  - Add backend tests for message translation storage and metadata round-trip.
  - Verify no regressions in:
    - ordinary chat generation
    - file translation jobs
    - chat history reopen
    - assistant reply export/copy/edit flows
  - Add rollout notes or feature gating if needed.
- Exit criteria:
  - [x] Focused backend and frontend tests pass.
  - [x] No regression in existing chat/translate tests.
  - [x] Feature remains app-contract compliant under review.

---

## Implementation Notes

### Technical direction

- Prefer a shared translation helper/prompt path over duplicating file-translation logic.
- Keep chat translation server-side so results are deterministic and auditable.
- Treat translated replies as first-class chat messages with metadata, not as ephemeral UI overlays.
- Keep the translation lifecycle consistent with the existing Translate job model where possible.
- Reuse pinned/default language settings and tone defaults from Translate settings wherever appropriate.
- Keep translations append-only by default; do not auto-retranslate a translation.

### Implementation targets

- `src/informity/api/routes_chat.py`
- `src/informity/api/routes_translate.py`
- `src/informity/db/models.py`
- `src/informity/db/sqlite.py`
- `src/frontend/src/components/chat/ChatMessage.tsx`
- `src/frontend/src/components/chat/ChatView.tsx`
- `src/frontend/src/types/api.ts`
- `src/frontend/src/api.ts`
- `src/frontend/src/pages/TranslatePage.tsx` (only if shared controls or helpers are reused)

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Translation becomes a second persistence system | Store the translated result as a normal chat message with metadata in the existing chat history path. |
| Chat UX becomes cluttered | Keep translation visually linked to the source reply and use compact footer controls. |
| Translation behavior drifts from Translate page | Reuse the same translation policy, language mapping, tone defaults, and no-think handling. |
| Source message edits make translations ambiguous | Store a source message reference, mark stale translations explicitly, and avoid auto-retranslating. |
| Normal chat or file translation regresses | Validate with focused tests and keep the implementation additive. |

---

## Non-Goals

- Replacing the Translate page.
- Creating a separate translation persistence subsystem.
- Allowing the assistant to infer translation requests without explicit routing.
- Auto-translating every chat reply by default.
- Cross-chat translation sharing.
- Chained translation controls on translated messages.

---

## Implementation Checklist

- [x] Backend translation request path defined.
- [x] Chat message translation metadata added.
- [x] Footer translate action added.
- [x] Translated replies reload from chat history.
- [x] Source edit/regenerate behavior specified, including stale-state rules.
- [x] Chat reload-recovery behavior implemented.
- [x] Shared lifecycle behavior defined for Chat + Translate.
- [x] Focused regression tests added.
- [x] Chat and Translate behavior remains intact.
