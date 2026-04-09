# ==============================================================================
# Informity AI — Term Dictionary Builder
# Post-index deterministic extraction and versioned dictionary build.
# ==============================================================================

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

import aiosqlite
import structlog

from informity.config import settings
from informity.db.sqlite import (
    delete_term_dictionary_version,
    finalize_term_dictionary_build_run,
    get_latest_term_dictionary_build_run,
    get_term_dictionary_current_version,
    get_term_dictionary_source_rows,
    insert_term_alias,
    insert_term_entry,
    insert_term_evidence,
    set_term_dictionary_current_version,
    start_term_dictionary_build_run,
    update_term_dictionary_build_run_progress,
)
from informity.indexer.term_dictionary_quality import evaluate_term_dictionary_quality
from informity.llm.term_dictionary import normalize_term_text, term_dictionary_enabled

log = structlog.get_logger(__name__)

_TERM_DEFINITION_PATTERN = re.compile(
    r'\b([A-Za-z][A-Za-z0-9/&-]*(?:\s+[A-Za-z][A-Za-z0-9/&-]*){1,6})\s+\(([A-Z][A-Z0-9]{1,9})\)'
)
_ACRONYM_DEFINITION_PATTERN = re.compile(
    r'\b([A-Z][A-Z0-9]{1,9})\s+\(([A-Za-z][A-Za-z0-9/&-]*(?:\s+[A-Za-z][A-Za-z0-9/&-]*){1,6})\)'
)
_SINGLE_LETTER_ACRONYM_SEQUENCE_PATTERN = re.compile(
    r'\b([A-Z](?:\s+[A-Z0-9]){1,9})\s*(?=\()'
)
_SINGLE_LETTER_ACRONYM_IN_PARENS_PATTERN = re.compile(
    r'\(([A-Z](?:\s+[A-Z0-9]){1,9})\)'
)
_OCR_HYPHENATED_LINEBREAK_PATTERN = re.compile(r'([A-Za-z]{2,})-\s*\n\s*([A-Za-z]{2,})')
_OCR_SPACED_HYPHEN_PATTERN = re.compile(r'([A-Za-z]{2,})\s*-\s*([A-Za-z]{2,})')
_WORD_TOKEN_PATTERN = re.compile(r'[a-z0-9]+')
_BOILERPLATE_PHRASES: tuple[str, ...] = (
    'see instructions',
    'see instruction',
    'see note above',
    'refer to instructions',
    'refer to note above',
    'for more information',
)


def _builder_enabled() -> bool:
    if not term_dictionary_enabled():
        return False
    return bool(settings.term_dictionary_build_enabled)


def _batch_size() -> int:
    try:
        value = int(settings.term_dictionary_build_batch_size)
    except (TypeError, ValueError):
        value = 500
    return max(50, min(2000, value))


@dataclass(slots=True)
class _Candidate:
    canonical: str
    normalized_canonical: str
    term_type: str
    confidence: float
    aliases: dict[str, tuple[str, float]] = field(default_factory=dict)  # normalized_alias -> (raw_alias, confidence)
    evidence: list[tuple[int | None, int | None, str, str]] = field(default_factory=list)


def _trim_snippet(text: str, limit: int = 220) -> str:
    snippet = ' '.join(str(text or '').split())
    return snippet[:limit]


def _preprocess_text_for_matching(text: str) -> str:
    value = str(text or '')
    if not value:
        return ''
    value = _OCR_HYPHENATED_LINEBREAK_PATTERN.sub(r'\1\2', value)
    value = _OCR_SPACED_HYPHEN_PATTERN.sub(r'\1-\2', value)
    value = _SINGLE_LETTER_ACRONYM_SEQUENCE_PATTERN.sub(
        lambda m: m.group(1).replace(' ', ''),
        value,
    )
    value = _SINGLE_LETTER_ACRONYM_IN_PARENS_PATTERN.sub(
        lambda m: f"({m.group(1).replace(' ', '')})",
        value,
    )
    return value


def _looks_like_person_name(long_term: str, acronym: str) -> bool:
    tokens = [token for token in re.split(r'\s+', long_term.strip()) if token]
    if len(tokens) != 2:
        return False
    if not all(token.isalpha() for token in tokens):
        return False
    if not all(token[0:1].isupper() and token[1:].islower() for token in tokens):
        return False
    initials = ''.join(token[0] for token in tokens).upper()
    return initials == acronym.strip().upper()


def _canonical_has_minimum_shape(canonical: str) -> bool:
    normalized = normalize_term_text(canonical)
    if not normalized:
        return False
    if any(phrase in normalized for phrase in _BOILERPLATE_PHRASES):
        return False
    tokens = [token for token in _WORD_TOKEN_PATTERN.findall(normalized) if token]
    if len(tokens) < 2:
        return False
    if len(set(tokens)) < 2:
        return False
    return not all(len(token) <= 2 for token in tokens)


def _score_candidate_confidence(*, long_term: str, acronym: str, extraction_method: str) -> float:
    score = 0.95
    initials = ''.join(token[0] for token in re.findall(r'[A-Za-z]+', long_term)).upper()
    if not initials or not acronym.upper().startswith(initials[: min(len(initials), len(acronym))]):
        score -= 0.15
    if any(ch.isdigit() for ch in long_term):
        score -= 0.10
    if extraction_method.endswith('ocr_normalized'):
        score -= 0.08
    if any(phrase in normalize_term_text(long_term) for phrase in _BOILERPLATE_PHRASES):
        score = min(score, 0.35)
    return max(0.0, min(1.0, score))


def _add_candidate(
    candidates: dict[str, _Candidate],
    *,
    canonical: str,
    alias: str,
    term_type: str,
    confidence: float,
    file_id: int | None,
    chunk_id: int | None,
    evidence_snippet: str,
    extraction_method: str,
) -> None:
    canonical_norm = normalize_term_text(canonical)
    alias_norm = normalize_term_text(alias)
    if not canonical_norm or not alias_norm:
        return
    if canonical_norm == alias_norm:
        return
    if not _canonical_has_minimum_shape(canonical):
        return

    existing = candidates.get(canonical_norm)
    if existing is None:
        existing = _Candidate(
            canonical=canonical.strip(),
            normalized_canonical=canonical_norm,
            term_type=term_type,
            confidence=confidence,
        )
        candidates[canonical_norm] = existing

    if confidence > existing.confidence:
        existing.confidence = confidence
    if alias_norm not in existing.aliases or confidence > existing.aliases[alias_norm][1]:
        existing.aliases[alias_norm] = (alias.strip(), confidence)

    if len(existing.evidence) < 4:
        existing.evidence.append((file_id, chunk_id, _trim_snippet(evidence_snippet), extraction_method))


def _extract_candidates_from_chunk(
    *,
    content: str,
    file_id: int | None,
    chunk_id: int | None,
    out: dict[str, _Candidate],
) -> None:
    raw_text = str(content or '')
    if not raw_text.strip():
        return

    def _extract_from_text(text: str, *, normalized_pass: bool) -> None:
        long_acronym_method = (
            'definition_pair_long_acronym_ocr_normalized'
            if normalized_pass
            else 'definition_pair_long_acronym'
        )
        acronym_long_method = (
            'definition_pair_acronym_long_ocr_normalized'
            if normalized_pass
            else 'definition_pair_acronym_long'
        )

        for match in _TERM_DEFINITION_PATTERN.finditer(text):
            long_term = match.group(1).strip()
            acronym = match.group(2).strip()
            if len(acronym) < 2:
                continue
            if bool(settings.term_dictionary_exclude_person_names) and _looks_like_person_name(long_term, acronym):
                continue
            _add_candidate(
                out,
                canonical=long_term,
                alias=acronym,
                term_type='acronym',
                confidence=_score_candidate_confidence(
                    long_term=long_term,
                    acronym=acronym,
                    extraction_method=long_acronym_method,
                ),
                file_id=file_id,
                chunk_id=chunk_id,
                evidence_snippet=match.group(0),
                extraction_method=long_acronym_method,
            )

        for match in _ACRONYM_DEFINITION_PATTERN.finditer(text):
            acronym = match.group(1).strip()
            long_term = match.group(2).strip()
            if len(acronym) < 2:
                continue
            if bool(settings.term_dictionary_exclude_person_names) and _looks_like_person_name(long_term, acronym):
                continue
            _add_candidate(
                out,
                canonical=long_term,
                alias=acronym,
                term_type='acronym',
                confidence=_score_candidate_confidence(
                    long_term=long_term,
                    acronym=acronym,
                    extraction_method=acronym_long_method,
                ),
                file_id=file_id,
                chunk_id=chunk_id,
                evidence_snippet=match.group(0),
                extraction_method=acronym_long_method,
            )

    _extract_from_text(raw_text, normalized_pass=False)
    preprocessed_text = _preprocess_text_for_matching(raw_text)
    if preprocessed_text and preprocessed_text != raw_text:
        _extract_from_text(preprocessed_text, normalized_pass=True)


async def rebuild_term_dictionary(
    db: aiosqlite.Connection,
    *,
    run_id: str | None = None,
) -> dict[str, object]:
    if not _builder_enabled():
        return {
            'status': 'skipped',
            'reason': 'disabled',
        }

    current_version = await get_term_dictionary_current_version(db)
    target_version = current_version + 1
    run_identifier = run_id or f'term-dict-{uuid.uuid4().hex[:12]}'
    processed_chunks = 0
    aliases_inserted = 0
    terms_inserted = 0
    last_chunk_id = 0
    candidates: dict[str, _Candidate] = {}

    # Keep history for diagnostics; do not overwrite previous runs.
    await start_term_dictionary_build_run(
        db,
        run_id=run_identifier,
        target_version=target_version,
    )
    log.info('term_dictionary_build_started', run_id=run_identifier, target_version=target_version)

    try:
        while True:
            rows = await get_term_dictionary_source_rows(
                db,
                after_chunk_id=last_chunk_id,
                limit=_batch_size(),
            )
            if not rows:
                break
            for row in rows:
                last_chunk_id = int(row['chunk_id'])
                processed_chunks += 1
                _extract_candidates_from_chunk(
                    content=row['content'],
                    file_id=row['file_id'],
                    chunk_id=row['chunk_id'],
                    out=candidates,
                )
            await update_term_dictionary_build_run_progress(
                db,
                run_id=run_identifier,
                last_processed_chunk_id=last_chunk_id,
                processed_chunks=processed_chunks,
            )

        # Conservative candidate filter to reduce dictionary noise.
        filtered_candidates = [
            candidate
            for candidate in candidates.values()
            if len(candidate.aliases) >= 1
            and candidate.confidence >= 0.65
        ]
        quality_gate = evaluate_term_dictionary_quality(
            total_candidates=len(candidates),
            kept_candidates=len(filtered_candidates),
            noise_rate_threshold=float(settings.term_dictionary_quality_noise_rate_threshold),
            min_candidates_for_gate=int(settings.term_dictionary_quality_min_candidates_for_gate),
            gate_enabled=bool(settings.term_dictionary_quality_gate_enabled),
        )
        if not quality_gate.passed:
            raise RuntimeError(f'term_dictionary_quality_gate_failed:{quality_gate.reason}')

        await delete_term_dictionary_version(db, dict_version=target_version)
        for candidate in filtered_candidates:
            status = 'active'
            term_id = await insert_term_entry(
                db,
                canonical_term=candidate.canonical,
                normalized_term=candidate.normalized_canonical,
                term_type=candidate.term_type,
                confidence=candidate.confidence,
                status=status,
                dict_version=target_version,
            )
            terms_inserted += 1

            for normalized_alias, (alias, alias_confidence) in candidate.aliases.items():
                await insert_term_alias(
                    db,
                    term_id=term_id,
                    alias=alias,
                    normalized_alias=normalized_alias,
                    alias_type='observed',
                    confidence=alias_confidence,
                )
                aliases_inserted += 1

            # Always include canonical alias to make exact canonical matching explicit.
            await insert_term_alias(
                db,
                term_id=term_id,
                alias=candidate.canonical,
                normalized_alias=candidate.normalized_canonical,
                alias_type='canonical',
                confidence=candidate.confidence,
            )
            aliases_inserted += 1

            for file_id, chunk_id, snippet, method in candidate.evidence:
                await insert_term_evidence(
                    db,
                    term_id=term_id,
                    file_id=file_id,
                    chunk_id=chunk_id,
                    evidence_snippet=snippet,
                    extraction_method=method,
                )

        await set_term_dictionary_current_version(db, target_version)
        await finalize_term_dictionary_build_run(
            db,
            run_id=run_identifier,
            status='completed',
            terms_inserted=terms_inserted,
            aliases_inserted=aliases_inserted,
        )
        log.info(
            'term_dictionary_build_completed',
            run_id=run_identifier,
            target_version=target_version,
            processed_chunks=processed_chunks,
            terms_inserted=terms_inserted,
            aliases_inserted=aliases_inserted,
        )
        return {
            'status': 'completed',
            'run_id': run_identifier,
            'target_version': target_version,
            'processed_chunks': processed_chunks,
            'terms_inserted': terms_inserted,
            'aliases_inserted': aliases_inserted,
        }
    except Exception as exc:  # noqa: BLE001 - final guard to always persist failed run status
        await finalize_term_dictionary_build_run(
            db,
            run_id=run_identifier,
            status='failed',
            terms_inserted=terms_inserted,
            aliases_inserted=aliases_inserted,
            error_summary=str(exc),
        )
        log.warning('term_dictionary_build_failed', run_id=run_identifier, error=str(exc))
        return {
            'status': 'failed',
            'run_id': run_identifier,
            'target_version': target_version,
            'processed_chunks': processed_chunks,
            'terms_inserted': terms_inserted,
            'aliases_inserted': aliases_inserted,
            'error': str(exc),
        }


async def get_term_dictionary_build_status(db: aiosqlite.Connection) -> dict[str, object]:
    latest = await get_latest_term_dictionary_build_run(db)
    current_version = await get_term_dictionary_current_version(db)
    return {
        'enabled': _builder_enabled(),
        'current_version': current_version,
        'latest_run': latest,
    }
