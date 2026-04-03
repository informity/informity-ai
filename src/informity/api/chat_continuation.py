# ==============================================================================
# Informity AI — Chat Continuation Helpers
# Continuation detection, pass controls, and completion/next-action resolution.
# ==============================================================================

import difflib
import re

from fastapi import HTTPException

from informity.config import settings
from informity.db.models import ChatMessage
from informity.llm.query_classifier import QueryClassification
from informity.llm.types import (
    ChatRole,
    CompletionMode,
    ContinuationResolutionReason,
    IntentProfileId,
    NextAction,
    OutputShape,
    QuerySubtype,
    StructuralGapReason,
    TimeoutReason,
)

_CONTINUATION_DUPLICATE_SIMILARITY_THRESHOLD = 0.9
_CONTINUATION_DUPLICATE_MIN_LENGTH = 240

_CONTINUATION_PHRASES = frozenset({
    'continue', 'continue please', 'please continue', 'go on', 'carry on',
    'resume', 'proceed', 'keep going', 'go ahead', 'next', 'next section',
    'next part', 'more', 'more please', 'the rest', 'rest',
})
_CONTINUATION_PATTERN = re.compile(
    r'^\s*(continue|go on|carry on|resume|proceed|keep going|go ahead)\b',
    re.IGNORECASE,
)
_CONTINUATION_STRUCTURED_OUTPUT_PATTERN = re.compile(
    r'\b(markdown\s+table|columns?\s*:|rows?\s+as|output\s+only\s+a\s+markdown\s+table)\b',
    re.IGNORECASE,
)
_TABLE_SEPARATOR_PATTERN = re.compile(r'^\s*\|?\s*:?-{3,}(?:\s*\|\s*:?-{3,})+\s*\|?\s*$')
_EMPTY_ORDERED_LIST_ITEM_PATTERN = re.compile(r'^\s*\d+\.\s*$')


def resolve_completion_state(
    *,
    completion_mode_override: CompletionMode | str | None,
    timeout_occurred: bool,
    timeout_reason: TimeoutReason | str | None,
    has_remaining_scope: bool,
) -> tuple[CompletionMode, bool]:
    terminal_timeout_reasons = {TimeoutReason.QUEUE_WAIT_TIMEOUT, TimeoutReason.FIRST_TOKEN_WATCHDOG_TIMEOUT}
    normalized_timeout_reason = str(timeout_reason or '').strip().lower()
    timeout_contributes_remaining_scope = (
        timeout_occurred and normalized_timeout_reason not in {reason.value for reason in terminal_timeout_reasons}
    )
    default_mode = CompletionMode.PARTIAL if timeout_occurred else CompletionMode.COMPLETE
    completion_mode = completion_mode_override or default_mode
    try:
        completion_mode = CompletionMode(str(completion_mode).strip().lower())
    except ValueError:
        completion_mode = default_mode
    resolved_remaining_scope = (
        has_remaining_scope
        or timeout_contributes_remaining_scope
        or completion_mode == CompletionMode.STOPPED
    )
    if resolved_remaining_scope and completion_mode == CompletionMode.COMPLETE:
        completion_mode = CompletionMode.SCOPED_COMPLETE
    if completion_mode == CompletionMode.COMPLETE:
        resolved_remaining_scope = False
    return completion_mode, resolved_remaining_scope


def resolve_next_action(
    *,
    stopped_by_user: bool,
    timeout_occurred: bool,
    has_remaining_scope: bool,
    continuation_resolution_reason: (
        ContinuationResolutionReason | StructuralGapReason | TimeoutReason | str | None
    ),
) -> tuple[NextAction, str | None]:
    if stopped_by_user:
        return NextAction.REGENERATE, 'stopped'
    normalized_reason = str(continuation_resolution_reason or '').strip().lower()
    if normalized_reason in {TimeoutReason.QUEUE_WAIT_TIMEOUT.value, TimeoutReason.FIRST_TOKEN_WATCHDOG_TIMEOUT.value}:
        return NextAction.NONE, None
    if normalized_reason in {ContinuationResolutionReason.DUPLICATE_CONTINUATION_DETECTED.value}:
        return NextAction.REGENERATE, 'stalled'
    if not has_remaining_scope:
        return NextAction.NONE, None
    if normalized_reason in {ContinuationResolutionReason.CONTINUATION_PASS_BUDGET_EXHAUSTED.value}:
        return NextAction.CONTINUE, 'budget_exhausted'
    if timeout_occurred:
        return NextAction.CONTINUE, 'timeout'
    return NextAction.CONTINUE, 'unresolved_content'


def enforce_completion_action_consistency(
    *,
    completion_mode: CompletionMode,
    has_remaining_scope: bool,
    next_action: NextAction,
    next_action_reason: str | None,
) -> tuple[CompletionMode, bool]:
    normalized_completion_mode = completion_mode
    normalized_has_remaining_scope = has_remaining_scope
    if next_action == NextAction.NONE:
        normalized_has_remaining_scope = False
        if normalized_completion_mode in {CompletionMode.SCOPED_COMPLETE, CompletionMode.PARTIAL}:
            normalized_completion_mode = CompletionMode.COMPLETE
        return normalized_completion_mode, normalized_has_remaining_scope
    if next_action == NextAction.REGENERATE:
        if (next_action_reason or '').strip().lower() == 'stalled':
            normalized_has_remaining_scope = False
            if normalized_completion_mode in {CompletionMode.SCOPED_COMPLETE, CompletionMode.PARTIAL}:
                normalized_completion_mode = CompletionMode.COMPLETE
        return normalized_completion_mode, normalized_has_remaining_scope
    if next_action == NextAction.ASSISTANT_SWITCH:
        normalized_has_remaining_scope = False
        if normalized_completion_mode in {CompletionMode.SCOPED_COMPLETE, CompletionMode.PARTIAL}:
            normalized_completion_mode = CompletionMode.COMPLETE
        return normalized_completion_mode, normalized_has_remaining_scope
    if not normalized_has_remaining_scope:
        normalized_has_remaining_scope = True
    if next_action == NextAction.CONTINUE and normalized_completion_mode == CompletionMode.COMPLETE:
        normalized_completion_mode = CompletionMode.SCOPED_COMPLETE
    return normalized_completion_mode, normalized_has_remaining_scope


def is_continuation_request(question: str) -> bool:
    normalized = ' '.join((question or '').strip().split())
    if not normalized:
        return False
    if normalized.casefold().rstrip('.!?') in _CONTINUATION_PHRASES:
        return True
    return bool(_CONTINUATION_PATTERN.search(normalized))


def resolve_continuation_anchor_question(*, question: str, history: list[ChatMessage]) -> str:
    normalized_question = (question or '').strip()
    if not is_continuation_request(normalized_question):
        return normalized_question
    if '##' in normalized_question:
        return normalized_question
    for message in reversed(history):
        if message.role != ChatRole.USER:
            continue
        candidate = str(message.content or '').strip()
        if not candidate:
            continue
        if is_continuation_request(candidate):
            continue
        return candidate
    return normalized_question


def build_auto_continue_pass_prompt(
    *,
    auto_continue_prompt: str,
    original_question: str,
) -> str:
    lines = [
        auto_continue_prompt.strip(),
        '',
        'Original request:',
        original_question.strip(),
    ]
    return '\n'.join(lines).strip()


def continuation_requires_structured_output(question: str) -> bool:
    return bool(_CONTINUATION_STRUCTURED_OUTPUT_PATTERN.search(str(question or '')))


def normalize_continuation_classification(
    *,
    classification: QueryClassification,
    continuation_anchor_question: str,
) -> QueryClassification:
    classification.is_continuation = True
    classification.route_candidate = IntentProfileId.CONTINUATION_OR_REFINEMENT
    if 'deterministic_continuation_route_enforced' not in classification.reason_codes:
        classification.reason_codes.append('deterministic_continuation_route_enforced')
    if not continuation_requires_structured_output(continuation_anchor_question):
        classification.response_shape = OutputShape.NARRATIVE_SYNTHESIS
        if classification.subtype == QuerySubtype.EXTRACT_STRUCTURED_VALUES:
            classification.subtype = None
            if 'deterministic_continuation_structured_subtype_cleared' not in classification.reason_codes:
                classification.reason_codes.append('deterministic_continuation_structured_subtype_cleared')
    return classification


def is_duplicate_continuation_pass(previous_answer: str | None, current_answer: str) -> bool:
    if not previous_answer:
        return False
    left = previous_answer.strip()
    right = (current_answer or '').strip()
    if not left or not right:
        return False
    if left == right:
        return True
    if min(len(left), len(right)) < _CONTINUATION_DUPLICATE_MIN_LENGTH:
        return False
    similarity = difflib.SequenceMatcher(a=left, b=right).ratio()
    return similarity >= _CONTINUATION_DUPLICATE_SIMILARITY_THRESHOLD


def build_continuing_status_message() -> str:
    return 'Continuing response...'


def enforce_continuation_chat_binding(*, question: str, chat_id: str | None) -> None:
    if chat_id:
        return
    if not is_continuation_request(question):
        return
    raise HTTPException(
        status_code=409,
        detail='Continuation requests require an existing chat. Use Continue from the same chat thread.',
    )


def has_continue_worthy_gap(has_remaining_scope_signal: bool) -> bool:
    return has_remaining_scope_signal


def detect_structural_incomplete_reason(answer: str) -> StructuralGapReason | None:
    text = str(answer or '')
    if not text.strip():
        return None
    if text.count('```') % 2 != 0:
        return StructuralGapReason.UNCLOSED_CODE_FENCE
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    last_line = lines[-1].rstrip()
    has_markdown_table = any(_TABLE_SEPARATOR_PATTERN.match(line) for line in lines)
    if has_markdown_table and last_line.lstrip().startswith('|') and not last_line.rstrip().endswith('|'):
        return StructuralGapReason.TRUNCATED_MARKDOWN_TABLE_ROW
    if _EMPTY_ORDERED_LIST_ITEM_PATTERN.match(last_line):
        return StructuralGapReason.TRUNCATED_MARKDOWN_LIST_ITEM
    return None


def mark_structural_output_gap(answer: str) -> StructuralGapReason | None:
    return detect_structural_incomplete_reason(answer)


def resolve_auto_continue_policy() -> tuple[bool, int, str]:
    enabled = bool(settings.chat_auto_continue_enabled)
    default_rounds = max(0, int(settings.chat_auto_continue_default_max_rounds))
    hard_cap = max(0, int(settings.chat_auto_continue_hard_cap))
    max_rounds = min(default_rounds, hard_cap)
    prompt = str(settings.chat_auto_continue_prompt or '').strip()
    if not prompt:
        prompt = (
            'Continue with the remaining sections from your last answer. '
            'Keep the same structure and avoid repeating completed sections.'
        )
    return enabled, max_rounds, prompt
