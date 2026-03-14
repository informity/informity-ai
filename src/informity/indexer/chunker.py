# ==============================================================================
# Informity AI — Text Chunker (v2)
# Simple Markdown header/paragraph splitting, one chunk size, sentence-aligned overlap
# ==============================================================================

import bisect
import re
from dataclasses import dataclass

import pysbd
import structlog
import tiktoken

from informity.config import settings

log = structlog.get_logger(__name__)

_TIKTOKEN_ENCODER = tiktoken.get_encoding('cl100k_base')
_SENTENCE_SEGMENTER = pysbd.Segmenter(language='en', clean=False)


@dataclass(frozen=True)
class ChunkData:
    # A single chunk of text.
    content:      str
    chunk_index:  int
    token_count:  int
    page_number:  int | None = None
    start_page:   int | None = None
    end_page:     int | None = None
    section_path: str | None = None
    block_type:   str | None = None  # Block type: 'table', 'form', 'narrative' (from docling provenance)
    parent_chunk_index: int | None = None  # For child chunks: index of parent chunk (before DB insertion)


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_TIKTOKEN_ENCODER.encode(text, disallowed_special=()))


def _lookup_range_with_starts(
    ranges: list[tuple[int, int, int | str]] | None,
    starts: list[int] | None,
    pos: int,
) -> int | str | None:
    # Same as _lookup_range but reuses precomputed start positions for speed.
    if not ranges or not starts:
        return None

    idx = bisect.bisect_right(starts, pos) - 1
    if idx >= 0:
        start, end, value = ranges[idx]
        if start <= pos < end:
            return value
    return None


def _lookup_page_span(
    page_ranges: list[tuple[int, int, int]] | None,
    page_starts: list[int] | None,
    chunk_start: int,
    chunk_end: int,
) -> tuple[int | None, int | None]:
    # Resolve page span for a chunk using start/end character positions.
    if not page_ranges:
        return None, None
    if chunk_end <= chunk_start:
        return None, None

    start_page = _lookup_range_with_starts(page_ranges, page_starts, chunk_start)
    end_page = _lookup_range_with_starts(page_ranges, page_starts, chunk_end - 1)
    return (
        int(start_page) if isinstance(start_page, int) else None,
        int(end_page) if isinstance(end_page, int) else None,
    )


def _is_header_only_chunk(content: str) -> bool:
    """
    Detect if a chunk contains only table/form headers without body content.

    This is a quality filter to prevent header-only chunks from being indexed.
    Header-only chunks provide no useful information for answering questions.

    Uses configurable thresholds from settings to allow tuning for different document types.
    Some documents (e.g., form templates, empty tables) genuinely contain header-only
    structures that provide little value for RAG.

    Args:
        content: Chunk content to check

    Returns:
        True if chunk appears to be header-only, False otherwise
    """
    if not content or len(content.strip()) < 50:
        return False

    lines = content.strip().split('\n')
    if not lines:
        return False

    # Count lines that look like table headers (start with | and contain separators)
    header_line_pattern = re.compile(r'^\s*\|.*\|\s*$')
    separator_line_pattern = re.compile(r'^\s*\|[\s\-:]+\|\s*$')

    header_lines = 0
    separator_lines = 0
    content_lines = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if separator_line_pattern.match(stripped):
            separator_lines += 1
        elif header_line_pattern.match(stripped):
            header_lines += 1
        else:
            content_lines += 1

    total_lines = header_lines + separator_lines + content_lines
    if total_lines == 0:
        return False

    # If most lines are headers/separators and there's minimal content, it's header-only
    header_ratio = (header_lines + separator_lines) / total_lines

    # Use configurable thresholds from settings
    header_ratio_threshold = settings.chunk_filter_header_ratio
    min_content_chars = settings.chunk_filter_min_content_chars
    min_content_lines = settings.chunk_filter_min_content_lines

    # Chunk is header-only if:
    # 1. Header ratio exceeds threshold, AND
    # 2. Content is short (< min_content_chars chars), OR
    # 3. Very few content lines (< min_content_lines lines of actual content)
    is_header_only = (
        header_ratio > header_ratio_threshold and
        (len(content) < min_content_chars or content_lines < min_content_lines)
    )

    return is_header_only


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
    char_to_page_ranges: list[tuple[int, int, int]] | None = None,
    char_to_block_type_ranges: list[tuple[int, int, str]] | None = None,
    char_to_header_level_ranges: list[tuple[int, int, int]] | None = None,
) -> list[ChunkData]:
    # Simple chunking: split on Markdown headers and paragraphs, sentence-aligned overlap.
    # Optionally assigns page numbers and section paths to chunks based on character positions.
    # Preserves table boundaries (tables are atomic units, never split mid-row).
    chunk_size = chunk_size or settings.chunk_size_tokens
    overlap = overlap or settings.chunk_overlap_tokens

    if not text.strip():
        return []

    # Track character position and section hierarchy for metadata assignment
    char_pos = 0
    section_stack: list[tuple[int, str]] = []  # [(header_level, "Section Name"), ...]
    page_starts = [r[0] for r in char_to_page_ranges] if char_to_page_ranges else None
    block_type_starts = [r[0] for r in char_to_block_type_ranges] if char_to_block_type_ranges else None
    header_level_starts = [r[0] for r in char_to_header_level_ranges] if char_to_header_level_ranges else None

    def _is_table_part(part: str) -> bool:
        """Check if a part is a markdown table (starts with | on first line)."""
        lines = part.strip().split('\n')
        if not lines:
            return False
        # Check if first non-empty line starts with |
        for line in lines:
            stripped = line.strip()
            if stripped:
                return stripped.startswith('|') and not stripped.startswith('||')
        return False

    # Split on double newlines (paragraphs) and Markdown headers
    # re.split with capturing group includes separators in result
    parts = re.split(r'(\n\n+|^#+\s+[^\n]+\n)', text, flags=re.MULTILINE)

    # Filter out empty parts
    parts = [p for p in parts if p.strip()]

    chunks: list[ChunkData] = []
    current_chunk: list[str] = []
    current_tokens = 0
    chunk_index = 0
    chunk_start_char_pos = 0  # Track where current chunk started
    chunk_start_section_stack: list[tuple[int, str]] = []  # Section stack when chunk started

    for part in parts:
        part_tokens = _count_tokens(part)

        # Check if this part is a markdown header and update section stack
        header_match = re.match(r'^(#+)\s+(.+)$', part.strip())
        if header_match:
            # Use char_to_header_level_ranges if available (more reliable than regex counting)
            if char_to_header_level_ranges:
                header_level_value = _lookup_range_with_starts(
                    char_to_header_level_ranges,
                    header_level_starts,
                    char_pos,
                )
                if header_level_value is not None:
                    header_level = header_level_value
                else:
                    header_level = len(header_match.group(1))
            else:
                header_level = len(header_match.group(1))
            header_text = header_match.group(2).strip()

            # Update section stack: remove headers at same or deeper level
            section_stack = [(level, name) for level, name in section_stack if level < header_level]
            # Add new header
            section_stack.append((header_level, header_text))

        # Check if this part is a table
        is_table = _is_table_part(part)

        # If adding this part would exceed chunk size, finalize current chunk
        would_exceed = current_tokens + part_tokens > chunk_size

        # Special handling for tables: if we're about to add a table and it would exceed,
        # finalize current chunk BEFORE the table, then add table as atomic unit
        if would_exceed and current_chunk and is_table:
            # Finalize current chunk before table
            chunk_content = '\n'.join(current_chunk)

            # Assign metadata using range lookup (binary search)
            chunk_end_char_pos = chunk_start_char_pos + len(chunk_content)
            start_page, end_page = _lookup_page_span(
                char_to_page_ranges,
                page_starts,
                chunk_start_char_pos,
                chunk_end_char_pos,
            )
            page_number = start_page

            block_type: str | None = None
            if char_to_block_type_ranges:
                block_type_value = _lookup_range_with_starts(
                    char_to_block_type_ranges,
                    block_type_starts,
                    chunk_start_char_pos,
                )
                if block_type_value is not None:
                    block_type = block_type_value

            section_path: str | None = None
            if chunk_start_section_stack:
                section_path = '/'.join([name for _, name in chunk_start_section_stack])

            chunks.append(ChunkData(
                content=chunk_content,
                chunk_index=chunk_index,
                token_count=current_tokens,
                page_number=page_number,
                start_page=start_page,
                end_page=end_page,
                section_path=section_path,
                block_type=block_type,
            ))
            chunk_index += 1

            # Start new chunk with table (no overlap, table is atomic)
            current_chunk = [part]
            current_tokens = part_tokens
            chunk_start_char_pos = char_pos
            chunk_start_section_stack = section_stack.copy()
            char_pos += len(part)
            continue

        # If adding this part would exceed chunk size (non-table), finalize current chunk
        if would_exceed and current_chunk:
            chunk_content = '\n'.join(current_chunk)

            # Assign metadata based on character position using range lookup (binary search)
            chunk_end_char_pos = chunk_start_char_pos + len(chunk_content)
            start_page, end_page = _lookup_page_span(
                char_to_page_ranges,
                page_starts,
                chunk_start_char_pos,
                chunk_end_char_pos,
            )
            page_number = start_page

            # Assign block type based on character position using range lookup
            block_type: str | None = None
            if char_to_block_type_ranges:
                block_type_value = _lookup_range_with_starts(
                    char_to_block_type_ranges,
                    block_type_starts,
                    chunk_start_char_pos,
                )
                if block_type_value is not None:
                    block_type = block_type_value

            # Build section path from section stack when chunk started
            section_path: str | None = None
            if chunk_start_section_stack:
                section_path = '/'.join([name for _, name in chunk_start_section_stack])

            chunks.append(ChunkData(
                content=chunk_content,
                chunk_index=chunk_index,
                token_count=current_tokens,
                page_number=page_number,
                start_page=start_page,
                end_page=end_page,
                section_path=section_path,
                block_type=block_type,
            ))
            chunk_index += 1

            # Start new chunk with overlap (sentence-aligned)
            overlap_text = _get_overlap_sentences(chunk_content, overlap)
            current_chunk = [overlap_text] if overlap_text else []
            current_tokens = _count_tokens(overlap_text)
            # Update character position and section stack for overlap
            if overlap_text:
                chunk_start_char_pos = char_pos - len(overlap_text)
                # For overlap, use current section stack (chunk continues in same section)
                chunk_start_section_stack = section_stack.copy()
            else:
                chunk_start_char_pos = char_pos
                chunk_start_section_stack = section_stack.copy()

        # Track section stack when starting a new chunk (before adding first part)
        if not current_chunk:
            chunk_start_section_stack = section_stack.copy()

        current_chunk.append(part)
        current_tokens += part_tokens
        char_pos += len(part)

    # Add final chunk
    if current_chunk:
        chunk_content = '\n'.join(current_chunk)

        # Assign metadata for final chunk using range lookup (binary search)
        chunk_end_char_pos = chunk_start_char_pos + len(chunk_content)
        start_page, end_page = _lookup_page_span(
            char_to_page_ranges,
            page_starts,
            chunk_start_char_pos,
            chunk_end_char_pos,
        )
        page_number = start_page

        # Assign block type for final chunk using range lookup
        block_type: str | None = None
        if char_to_block_type_ranges:
            block_type_value = _lookup_range_with_starts(
                char_to_block_type_ranges,
                block_type_starts,
                chunk_start_char_pos,
            )
            if block_type_value is not None:
                block_type = block_type_value

        section_path: str | None = None
        if chunk_start_section_stack:
            section_path = '/'.join([name for _, name in chunk_start_section_stack])

        chunks.append(ChunkData(
            content=chunk_content,
            chunk_index=chunk_index,
            token_count=_count_tokens(chunk_content),
            page_number=page_number,
            start_page=start_page,
            end_page=end_page,
            section_path=section_path,
            block_type=block_type,
        ))

    # Filter out header-only chunks if enabled (quality improvement: prevents indexing noise)
    # This is app-compliant because it's a quality filter at indexing time, not query-time cleaning.
    # Filtering is configurable to allow tuning for different document types (some documents
    # genuinely contain header-only structures that may or may not be useful for RAG).
    if settings.chunk_filter_header_only:
        filtered_chunks = [
            chunk for chunk in chunks
            if not _is_header_only_chunk(chunk.content)
        ]

        # Log if any chunks were filtered
        if len(filtered_chunks) < len(chunks):
            log.debug(
                'header_only_chunks_filtered',
                total_chunks=len(chunks),
                filtered_out=len(chunks) - len(filtered_chunks),
                remaining=len(filtered_chunks)
            )

        # Never return an empty chunk set for non-empty text. If filtering removes
        # everything, keep one original chunk so indexing does not create file rows
        # without chunks.
        if chunks and not filtered_chunks:
            log.warning(
                'header_only_filter_fallback_applied',
                total_chunks=len(chunks),
                fallback_chunk_index=chunks[0].chunk_index,
            )
            return [chunks[0]]

        return filtered_chunks
    else:
        # Filter disabled, return all chunks
        return chunks


def _get_overlap_sentences(text: str, overlap_tokens: int) -> str:
    # Get last N sentences that fit within overlap_tokens.
    if not text.strip():
        return ''

    sentences = _SENTENCE_SEGMENTER.segment(text)
    if not sentences:
        return ''

    overlap_sentences = []
    overlap_count = 0

    for sentence in reversed(sentences):
        sent_tokens = _count_tokens(sentence)
        if overlap_count + sent_tokens <= overlap_tokens:
            overlap_sentences.insert(0, sentence)
            overlap_count += sent_tokens
        else:
            break

    return ' '.join(overlap_sentences)


def create_child_chunks(
    parent_chunks: list[ChunkData],
    child_size: int | None = None,
    overlap: int | None = None,
) -> list[ChunkData]:
    """
    Create child chunks from parent chunks for Parent Document Retrieval.

    Child chunks are smaller (typically 100-150 tokens, 1-2 sentences) for precise search matching.
    Parent chunks provide context windows (typically 500-800 tokens) for the LLM.

    Args:
        parent_chunks: List of parent chunks (from chunk_text())
        child_size: Target size for child chunks in tokens (default: settings.chunk_child_size_tokens)
        overlap: Overlap between child chunks in tokens (default: settings.chunk_overlap_tokens)

    Returns:
        List of child ChunkData objects with parent_chunk_index set to their parent's chunk_index.
    """
    child_size = child_size or settings.chunk_child_size_tokens
    overlap = overlap or settings.chunk_overlap_tokens

    if not parent_chunks:
        return []

    child_chunks: list[ChunkData] = []
    child_index = 0

    for _parent_idx, parent in enumerate(parent_chunks):
        parent_text = parent.content

        # If parent is smaller than child_size, use it as-is (no need to split)
        if parent.token_count <= child_size:
            child_chunks.append(ChunkData(
                content=parent_text,
                chunk_index=child_index,
                token_count=parent.token_count,
                page_number=parent.page_number,
                start_page=parent.start_page,
                end_page=parent.end_page,
                section_path=parent.section_path,
                block_type=parent.block_type,
                parent_chunk_index=parent.chunk_index,
            ))
            child_index += 1
            continue

        # Split parent into child chunks
        # Use sentence segmentation for clean boundaries
        sentences = _SENTENCE_SEGMENTER.segment(parent_text)
        if not sentences:
            # Fallback: if sentence segmentation fails, use the parent as-is
            child_chunks.append(ChunkData(
                content=parent_text,
                chunk_index=child_index,
                token_count=parent.token_count,
                page_number=parent.page_number,
                start_page=parent.start_page,
                end_page=parent.end_page,
                section_path=parent.section_path,
                block_type=parent.block_type,
                parent_chunk_index=parent.chunk_index,
            ))
            child_index += 1
            continue

        # Build child chunks from sentences
        current_child: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = _count_tokens(sentence)

            # If adding this sentence would exceed child_size, finalize current child
            if current_tokens + sent_tokens > child_size and current_child:
                child_content = ' '.join(current_child)
                child_chunks.append(ChunkData(
                    content=child_content,
                    chunk_index=child_index,
                    token_count=current_tokens,
                    page_number=parent.page_number,  # Inherit from parent
                    start_page=parent.start_page,
                    end_page=parent.end_page,
                    section_path=parent.section_path,  # Inherit from parent
                    block_type=parent.block_type,  # Inherit from parent
                    parent_chunk_index=parent.chunk_index,
                ))
                child_index += 1

                # Start new child with overlap (last N sentences from previous child)
                overlap_text = _get_overlap_sentences(child_content, overlap)
                current_child = [overlap_text] if overlap_text else []
                current_tokens = _count_tokens(overlap_text)

            current_child.append(sentence)
            current_tokens += sent_tokens

        # Add final child chunk
        if current_child:
            child_content = ' '.join(current_child)
            child_chunks.append(ChunkData(
                content=child_content,
                chunk_index=child_index,
                token_count=_count_tokens(child_content),
                page_number=parent.page_number,
                start_page=parent.start_page,
                end_page=parent.end_page,
                section_path=parent.section_path,
                block_type=parent.block_type,
                parent_chunk_index=parent.chunk_index,
            ))
            child_index += 1

    return child_chunks
