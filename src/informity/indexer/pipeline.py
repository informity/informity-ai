# ==============================================================================
# Informity AI — Indexing Pipeline (v2)
# Linear flow: extract → chunk → embed → store
# ==============================================================================

import asyncio
import hashlib
import mimetypes
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Union

import aiosqlite
import structlog

from informity.db.models import Chunk, IndexedFile
from informity.db.sqlite import (
    delete_chunks_for_file,
    delete_file,
    get_file_by_path,
    get_file_by_source_identity,
    insert_chunks_batch,
    insert_file,
    update_file,
)
from informity.db.vectors import ChunkEmbedding, vector_store
from informity.indexer.chunker import chunk_text, create_child_chunks
from informity.indexer.classifier import classify_file, extract_year, generate_tags
from informity.indexer.embedder import embedder
from informity.indexer.post_process import post_process_extracted_text
from informity.scanner.extractors.base import (
    MAX_EXTRACTED_TEXT_PREVIEW,
    BaseExtractor,
    get_extractor,
)
from informity.scanner.extractors.text_utils import get_max_file_size_bytes
from informity.sources.base import FILESYSTEM_PROVIDER, SOURCE_ENTITY_FILE, IngestionItem
from informity.utils.file_utils import normalize_extension
from informity.utils.path_utils import normalize_path

if TYPE_CHECKING:
    from informity.scanner.crawler import ScannedFile

log = structlog.get_logger(__name__)
_INDEXER_RUNTIME_EXCEPTIONS = (aiosqlite.Error, sqlite3.Error, RuntimeError, ValueError, TypeError, OSError, TimeoutError)
_EMBEDDING_MODEL_MAX_TOKENS = 8192

@dataclass
class IndexResult:
    success: bool
    chunks_created: int
    error: str | None = None
    extractor: str | None = None
    ocr_used: bool = False
    error_code: str | None = None
    retryable: bool = True


def _parse_int_metadata(metadata: dict[str, str], key: str) -> int | None:
    # Parse an integer metadata value safely.
    raw_value = metadata.get(key)
    if raw_value is None:
        return None
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None


def _build_file_metadata(path: Path, doc_metadata: dict[str, str]) -> dict[str, object]:
    # Normalize extractor metadata for persistent file fields.
    mime_type, _ = mimetypes.guess_type(path.name)
    return {
        'extractor': doc_metadata.get('converter'),
        'encoding': doc_metadata.get('encoding'),
        'language': doc_metadata.get('language'),
        'mime_type': doc_metadata.get('mime_type') or mime_type,
        'ocr_used': doc_metadata.get('ocr_used') == 'true',
        'page_count': _parse_int_metadata(doc_metadata, 'page_count'),
        'tables_count': _parse_int_metadata(doc_metadata, 'tables_count'),
        'form_items_count': _parse_int_metadata(doc_metadata, 'form_items_count'),
        'key_value_items_count': _parse_int_metadata(doc_metadata, 'key_value_items_count'),
        'pictures_count': _parse_int_metadata(doc_metadata, 'pictures_count'),
        'document_hash': doc_metadata.get('document_hash'),
    }


async def _cleanup_partial_file_data(
    db: aiosqlite.Connection,
    file_id: int,
    *,
    path: str,
    reason: str,
) -> None:
    # Best-effort cleanup for partial index/reindex failures.
    try:
        await delete_chunks_for_file(db, file_id)
    except _INDEXER_RUNTIME_EXCEPTIONS as exc:
        log.warning(
            'partial_cleanup_chunks_failed',
            file_id=file_id,
            path=path,
            reason=reason,
            error=str(exc),
        )

    try:
        await asyncio.to_thread(vector_store.delete_by_file_id, file_id)
    except _INDEXER_RUNTIME_EXCEPTIONS as exc:
        log.warning(
            'partial_cleanup_vectors_failed',
            file_id=file_id,
            path=path,
            reason=reason,
            error=str(exc),
        )


# ==============================================================================
# Shared Helper — Chunk, Embed, Store
# ==============================================================================

async def _chunk_embed_store(
    db: aiosqlite.Connection,
    file_id: int,
    text: str,
    file_path: Path,
    filename: str,
    extension: str,
    category: str,
    year: int | None,
    char_to_page_ranges: list[tuple[int, int, int]] | None = None,
    char_to_block_type_ranges: list[tuple[int, int, str]] | None = None,
    char_to_header_level_ranges: list[tuple[int, int, int]] | None = None,
) -> IndexResult:
    # Shared logic for chunking, embedding, and storing with Parent Document Retrieval.
    # Two-pass insertion: (1) insert parent chunks, (2) insert child chunks with parent_id.
    # Only child chunks are embedded and stored in SQLite vector storage (they're what we search).
    try:
        # Step 1: Create parent chunks (current chunk_size_tokens, ~512 tokens)
        parent_chunks = chunk_text(
            text,
            char_to_page_ranges=char_to_page_ranges,
            char_to_block_type_ranges=char_to_block_type_ranges,
            char_to_header_level_ranges=char_to_header_level_ranges,
        )

        # Step 2: Insert parent chunks into SQLite (no embeddings yet)
        parent_chunk_models = [
            Chunk(
                file_id=file_id,
                chunk_index=parent.chunk_index,
                content=parent.content,
                token_count=parent.token_count,
                page_number=parent.page_number,
                start_page=parent.start_page,
                end_page=parent.end_page,
                section_path=parent.section_path,
                block_type=parent.block_type,
                parent_id=None,  # Parents have no parent
            )
            for parent in parent_chunks
        ]
        parent_chunk_ids = await insert_chunks_batch(db, file_id, parent_chunk_models)

        # Step 3: Create child chunks from parents (smaller, ~150 tokens for precise matching)
        child_chunks = create_child_chunks(parent_chunks)

        # Log chunking summary at INFO level for operational visibility
        log.info(
            'chunking_complete',
            file_id=file_id,
            filename=filename,
            parent_chunks=len(parent_chunks),
            child_chunks=len(child_chunks),
            total_chunks=len(parent_chunks) + len(child_chunks)
        )

        # Step 4: Map child chunks to their parent IDs
        # Build mapping: parent_chunk_index -> parent SQLite ID
        parent_index_to_id: dict[int, int] = {
            parent.chunk_index: parent_id
            for parent, parent_id in zip(parent_chunks, parent_chunk_ids, strict=True)
        }

        # Step 5: Insert child chunks with parent_id set
        child_chunk_models = [
            Chunk(
                file_id=file_id,
                chunk_index=child.chunk_index,
                content=child.content,
                token_count=child.token_count,
                page_number=child.page_number,
                start_page=child.start_page,
                end_page=child.end_page,
                section_path=child.section_path,
                block_type=child.block_type,
                parent_id=parent_index_to_id.get(child.parent_chunk_index) if child.parent_chunk_index is not None else None,
            )
            for child in child_chunks
        ]
        child_chunk_ids = await insert_chunks_batch(db, file_id, child_chunk_models)

        # Step 6: Embed and store child chunks in SQLite vector storage (only children are indexed for search)
        # Safety: Skip chunks that exceed embedding model's context window (nomic-embed-text-v1.5).
        # These chunks would fail to embed anyway and are likely corrupted by glyph sequences

        # Use embedder-effective batch size so outer batching aligns with
        # embedder safety caps (for example, MPS limits), minimizing double-splitting.
        batch_size = embedder.get_effective_batch_size()
        total_child_chunks = len(child_chunks)
        skipped_count = 0
        embedded_count = 0
        failed_batches = 0
        total_batches = (total_child_chunks + batch_size - 1) // batch_size

        # Log embedding start at INFO level
        log.info(
            'embedding_start',
            file_id=file_id,
            filename=filename,
            total_child_chunks=total_child_chunks,
            total_batches=total_batches,
            batch_size=batch_size
        )

        for batch_start in range(0, total_child_chunks, batch_size):
            batch_end = min(batch_start + batch_size, total_child_chunks)
            batch_child_chunks = child_chunks[batch_start:batch_end]
            batch_child_chunk_ids = child_chunk_ids[batch_start:batch_end]

            # Filter out chunks that exceed embedding model's context window
            valid_chunks = []
            valid_chunk_ids = []
            for child, chunk_id in zip(batch_child_chunks, batch_child_chunk_ids, strict=True):
                if child.token_count > _EMBEDDING_MODEL_MAX_TOKENS:
                    skipped_count += 1
                    log.warning(
                        'chunk_exceeds_embedding_context',
                        chunk_id=chunk_id,
                        file_id=file_id,
                        filename=filename,
                        token_count=child.token_count,
                        max_tokens=_EMBEDDING_MODEL_MAX_TOKENS,
                        action='skipping_embedding'
                    )
                    continue
                valid_chunks.append(child)
                valid_chunk_ids.append(chunk_id)

            if not valid_chunks:
                continue  # Skip empty batches

            # Embed this batch of child chunks
            try:
                texts = [child.content for child in valid_chunks]
                # Run embedding in thread pool to avoid blocking event loop
                embeddings = await asyncio.to_thread(embedder.embed_texts, texts)

                # Validate embeddings were generated
                if len(embeddings) != len(valid_chunks):
                    raise ValueError(
                        f'Embedding count mismatch: expected {len(valid_chunks)}, got {len(embeddings)}'
                    )

                # Store child chunks in SQLite vector storage
                chunk_embeddings = [
                    ChunkEmbedding(
                        chunk_id=valid_chunk_ids[i],
                        file_id=file_id,
                        file_path=str(file_path),
                        chunk_text=valid_chunks[i].content,
                        vector=embeddings[i],
                        year=year,
                        filename=filename,
                        extension=extension,
                        category=category,
                    )
                    for i in range(len(valid_chunks))
                ]

                # Store embeddings with error handling
                try:
                    stored_count = await vector_store.store_embeddings_async(chunk_embeddings)
                    if stored_count != len(chunk_embeddings):
                        raise RuntimeError(
                            f'SQLite vector store partial write: expected {len(chunk_embeddings)}, got {stored_count}'
                        )
                    embedded_count += stored_count
                    batch_num = (batch_start // batch_size) + 1

                    # DEBUG: Detailed per-batch info (for deep troubleshooting)
                    log.debug(
                        'batch_embedded_and_stored',
                        file_id=file_id,
                        filename=filename,
                        batch_size=len(valid_chunks),
                        stored=stored_count,
                        batch_num=batch_num
                    )

                    # INFO: Progress summary every 5 batches or on last batch
                    if batch_num % 5 == 0 or batch_num == total_batches:
                        log.info(
                            'embedding_progress',
                            file_id=file_id,
                            filename=filename,
                            batch_num=batch_num,
                            total_batches=total_batches,
                            embedded_so_far=embedded_count,
                            progress_pct=int((embedded_count / max(total_child_chunks - skipped_count, 1)) * 100)
                        )
                except _INDEXER_RUNTIME_EXCEPTIONS as store_exc:
                    failed_batches += 1
                    log.error(
                        'sqlite_vec_store_failed',
                        file_id=file_id,
                        filename=filename,
                        batch_start=batch_start,
                        batch_end=batch_end,
                        batch_size=len(valid_chunks),
                        error=str(store_exc),
                        error_type=type(store_exc).__name__,
                        action='continuing_with_next_batch'
                    )
                    # Continue with next batch - don't fail entire file
                    continue

            except _INDEXER_RUNTIME_EXCEPTIONS as embed_exc:
                failed_batches += 1
                log.error(
                    'embedding_batch_failed',
                    file_id=file_id,
                    filename=filename,
                    batch_start=batch_start,
                    batch_end=batch_end,
                    batch_size=len(valid_chunks),
                    error=str(embed_exc),
                    error_type=type(embed_exc).__name__,
                    action='continuing_with_next_batch'
                )
                # Continue with next batch - don't fail entire file
                continue

        # Log summary
        if skipped_count > 0:
            log.warning(
                'chunks_skipped_due_to_size',
                file_id=file_id,
                filename=filename,
                skipped_count=skipped_count,
                total_child_chunks=total_child_chunks
            )

        expected_embedded_count = total_child_chunks - skipped_count
        if failed_batches > 0 or embedded_count != expected_embedded_count:
            reason = (
                f'failed_batches={failed_batches}, '
                f'embedded_count={embedded_count}, expected={expected_embedded_count}'
            )
            log.error(
                'embedding_consistency_failed',
                file_id=file_id,
                filename=filename,
                reason=reason,
            )
            await _cleanup_partial_file_data(
                db=db,
                file_id=file_id,
                path=str(file_path),
                reason=reason,
            )
            return IndexResult(
                success=False,
                chunks_created=0,
                error=f'Embedding consistency failed: {reason}',
            )

        log.info(
            'embedding_complete',
            file_id=file_id,
            filename=filename,
            embedded_count=embedded_count,
            skipped_count=skipped_count,
            total_child_chunks=total_child_chunks
        )

        # Return total chunks created (parents + children)
        total_chunks = len(parent_chunks) + len(child_chunks)
        return IndexResult(success=True, chunks_created=total_chunks)
    except _INDEXER_RUNTIME_EXCEPTIONS as exc:
        log.error('chunk_embed_store_failed', file_id=file_id, error=str(exc), exc_info=True)
        return IndexResult(success=False, chunks_created=0, error=str(exc))


# ==============================================================================
# Public API — index_file
# ==============================================================================

async def index_file(
    db: aiosqlite.Connection,
    file_path_or_scanned: Union[Path, 'ScannedFile', IngestionItem],
    extractor: BaseExtractor | None = None,
    *,
    source_provider: str = FILESYSTEM_PROVIDER,
    entity_type: str = SOURCE_ENTITY_FILE,
) -> IndexResult:
    # Index a single file: extract → chunk → embed → store.
    # Accepts either (db, file_path, extractor) or (db, scanned: ScannedFile).
    try:
        scanned_file = None
        # Handle IngestionItem, ScannedFile, and Path
        # Check for ScannedFile-specific attributes (content_hash, filename, extension)
        if isinstance(file_path_or_scanned, IngestionItem):
            item = file_path_or_scanned
            source_provider = item.provider or source_provider
            entity_type = item.item_type or entity_type
            pseudo_path = str(item.metadata.get('path') or f'source://{source_provider}/{entity_type}/{item.source_item_id}')
            file_path = Path(pseudo_path)
            filename = str(item.metadata.get('filename') or item.title or item.source_item_id or 'source-item')
            extension = normalize_extension(str(item.metadata.get('extension') or '.txt'))
            modified_at = item.modified_at or datetime.now(UTC)
            content_hash = item.content_hash or hashlib.sha256(item.content_text.encode('utf-8')).hexdigest()
            size_bytes = int(item.size_bytes or item.metadata.get('size_bytes') or len((item.content_text or '').encode('utf-8')))
            # IngestionItem provides normalized text directly.
            doc_text = item.content_text or ''
            file_metadata = {
                'extractor': item.metadata.get('extractor'),
                'encoding': item.metadata.get('encoding'),
                'language': item.metadata.get('language'),
                'mime_type': item.metadata.get('mime_type'),
                'ocr_used': bool(item.metadata.get('ocr_used', False)),
                'page_count': item.metadata.get('page_count'),
                'tables_count': item.metadata.get('tables_count'),
                'form_items_count': item.metadata.get('form_items_count'),
                'key_value_items_count': item.metadata.get('key_value_items_count'),
                'pictures_count': item.metadata.get('pictures_count'),
                'document_hash': item.metadata.get('document_hash'),
            }
        elif hasattr(file_path_or_scanned, 'content_hash') and hasattr(file_path_or_scanned, 'filename'):
            # It's a ScannedFile
            scanned = file_path_or_scanned
            scanned_file = scanned
            file_path = scanned.path
            if extractor is None:
                extractor = get_extractor(file_path)
                if extractor is None:
                    return IndexResult(
                        success=False,
                        chunks_created=0,
                        error=f'No extractor for extension: {scanned.extension}',
                    )
            # Use scanned metadata (including pre-computed content_hash)
            size_bytes = scanned.size_bytes
            modified_at = scanned.modified_at
            filename = scanned.filename
            extension = scanned.extension
            content_hash = scanned.content_hash  # Already computed by crawler
            doc_text = ''
            file_metadata: dict[str, object] = {}
        else:
            # It's a Path
            file_path = file_path_or_scanned
            if extractor is None:
                return IndexResult(
                    success=False,
                    chunks_created=0,
                    error='Extractor is required when using Path',
                )
            # Get metadata from file system
            stat = file_path.stat()
            size_bytes = stat.st_size
            max_bytes = get_max_file_size_bytes()
            if size_bytes > max_bytes:
                return IndexResult(
                    success=False,
                    chunks_created=0,
                    error=(
                        f'File too large to index directly ({size_bytes / (1024 * 1024):.1f} MB). '
                        f'Max allowed: {max_bytes // (1024 * 1024)} MB.'
                    ),
                    error_code='file_too_large',
                    retryable=False,
                )
            modified_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
            filename = file_path.name
            extension = file_path.suffix
            # Compute content hash (Path doesn't have pre-computed hash)
            content_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            doc_text = ''
            file_metadata = {}

        # Log file indexing start at INFO level
        log.info(
            'indexing_file_start',
            path=str(file_path),
            filename=filename,
            extension=extension,
            size_bytes=size_bytes
        )

        if not isinstance(file_path_or_scanned, IngestionItem):
            # 1. Extract (run in thread pool to avoid blocking and help with memory)
            doc = await asyncio.to_thread(extractor.extract, file_path)
            if doc.error:
                retryable = doc.metadata.get('retryable', 'true').lower() != 'false'
                error_code = doc.metadata.get('error_code')
                log.error(
                    'extraction_failed',
                    path=str(file_path),
                    filename=filename,
                    error=doc.error,
                    error_code=error_code,
                    retryable=retryable,
                )
                return IndexResult(
                    success=False,
                    chunks_created=0,
                    error=doc.error,
                    error_code=error_code,
                    retryable=retryable,
                )
            doc_text = doc.text
            file_metadata = _build_file_metadata(file_path, doc.metadata)

        # 2. Classify
        category = classify_file(file_path, extension)
        year = extract_year(file_path, doc_text)
        tags = generate_tags(file_path)

        # 3. Insert file
        # content_hash already computed above (from ScannedFile or computed from Path)
        indexed_file = IndexedFile(
            source_provider=source_provider,
            entity_type=entity_type,
            source_item_id=(
                file_path_or_scanned.source_item_id
                if isinstance(file_path_or_scanned, IngestionItem)
                else str(normalize_path(file_path, expand_user=False))
            ),
            path=(
                str(file_path)
                if isinstance(file_path_or_scanned, IngestionItem)
                else str(normalize_path(file_path, expand_user=False))
            ),
            filename=filename,
            extension=extension,
            size_bytes=size_bytes,
            content_hash=content_hash,
            extracted_text_preview=(
                (doc.preview_text if not isinstance(file_path_or_scanned, IngestionItem) else '')
                or doc_text[:MAX_EXTRACTED_TEXT_PREVIEW]
            ),
            category=category,
            tags=tags,
            year=year,
            extractor=file_metadata['extractor'],
            encoding=file_metadata['encoding'],
            language=file_metadata['language'],
            mime_type=file_metadata['mime_type'],
            ocr_used=bool(file_metadata['ocr_used']),
            page_count=file_metadata['page_count'],
            tables_count=file_metadata['tables_count'],
            form_items_count=file_metadata['form_items_count'],
            key_value_items_count=file_metadata['key_value_items_count'],
            pictures_count=file_metadata['pictures_count'],
            document_hash=file_metadata['document_hash'],
            indexed_at=datetime.now(UTC),
            modified_at=modified_at,
        )
        existing_by_identity = await get_file_by_source_identity(
            db,
            source_provider=indexed_file.source_provider,
            entity_type=indexed_file.entity_type,
            source_item_id=indexed_file.source_item_id,
        )
        if isinstance(file_path_or_scanned, IngestionItem) and existing_by_identity is not None and existing_by_identity.id is not None:
            file_record = existing_by_identity
            try:
                await delete_chunks_for_file(db, file_record.id)
                await asyncio.to_thread(vector_store.delete_by_file_id, file_record.id)
            except _INDEXER_RUNTIME_EXCEPTIONS as exc:
                log.warning(
                    'ingestion_item_old_data_cleanup_failed',
                    file_id=file_record.id,
                    source_provider=indexed_file.source_provider,
                    entity_type=indexed_file.entity_type,
                    source_item_id=indexed_file.source_item_id,
                    error=str(exc),
                )

            file_record.source_provider = indexed_file.source_provider
            file_record.entity_type = indexed_file.entity_type
            file_record.source_item_id = indexed_file.source_item_id
            file_record.path = indexed_file.path
            file_record.filename = indexed_file.filename
            file_record.extension = indexed_file.extension
            file_record.size_bytes = indexed_file.size_bytes
            file_record.content_hash = indexed_file.content_hash
            file_record.extracted_text_preview = indexed_file.extracted_text_preview
            file_record.category = indexed_file.category
            file_record.tags = indexed_file.tags
            file_record.year = indexed_file.year
            file_record.extractor = indexed_file.extractor
            file_record.encoding = indexed_file.encoding
            file_record.language = indexed_file.language
            file_record.mime_type = indexed_file.mime_type
            file_record.ocr_used = indexed_file.ocr_used
            file_record.page_count = indexed_file.page_count
            file_record.tables_count = indexed_file.tables_count
            file_record.form_items_count = indexed_file.form_items_count
            file_record.key_value_items_count = indexed_file.key_value_items_count
            file_record.pictures_count = indexed_file.pictures_count
            file_record.document_hash = indexed_file.document_hash
            file_record.indexed_at = indexed_file.indexed_at
            file_record.modified_at = indexed_file.modified_at
            file_record = await update_file(db, file_record)
        else:
            try:
                file_record = await insert_file(db, indexed_file)
            except sqlite3.IntegrityError as exc:
                # Idempotency hardening: another writer inserted the same source identity concurrently.
                message = str(exc)
                if (
                    'UNIQUE constraint failed: files.path' not in message
                    and 'UNIQUE constraint failed: files.source_provider, files.entity_type, files.source_item_id' not in message
                ):
                    raise
                normalized_path = str(normalize_path(file_path, expand_user=False))
                log.warning(
                    'index_file_duplicate_identity_conflict',
                    path=normalized_path,
                    action='reindex_fallback',
                )
                if scanned_file is None:
                    # Build minimal ScannedFile from already-computed metadata.
                    from informity.scanner.crawler import ScannedFile as ScannerScannedFile
                    scanned_file = ScannerScannedFile(
                        path=normalize_path(file_path, expand_user=False),
                        filename=filename,
                        extension=extension.lower(),
                        size_bytes=size_bytes,
                        content_hash=content_hash,
                        modified_at=modified_at,
                    )
                return await reindex_file(
                    db,
                    scanned_file,
                    source_provider=source_provider,
                    entity_type=entity_type,
                )

        # 4. Post-process extracted text (clean glyph sequences, etc.)
        cleaned_text = post_process_extracted_text(doc_text)

        # 5. Chunk, embed, and store (shared logic)
        # Pass per-chunk metadata ranges from extractor (for docling formats with provenance)
        result = await _chunk_embed_store(
            db=db,
            file_id=file_record.id,
            text=cleaned_text,
            file_path=file_path,
            filename=filename,
            extension=extension,
            category=category.value,
            year=year,
            char_to_page_ranges=(
                doc.char_to_page_ranges
                if not isinstance(file_path_or_scanned, IngestionItem)
                else None
            ),
            char_to_block_type_ranges=(
                doc.char_to_block_type_ranges
                if not isinstance(file_path_or_scanned, IngestionItem)
                else None
            ),
            char_to_header_level_ranges=(
                doc.char_to_header_level_ranges
                if not isinstance(file_path_or_scanned, IngestionItem)
                else None
            ),
        )
        result.extractor = file_metadata['extractor'] if isinstance(file_metadata['extractor'], str) else None
        result.ocr_used = bool(file_metadata['ocr_used'])

        # If chunk/embed/store failed, delete the orphaned file record so it can be retried
        if not result.success:
            try:
                await delete_file(db, file_record.id)
                log.warning(
                    'file_record_deleted_after_failure',
                    file_id=file_record.id,
                    path=str(file_path),
                    error=result.error,
                )
            except _INDEXER_RUNTIME_EXCEPTIONS as exc:
                log.error(
                    'failed_to_delete_orphaned_file',
                    file_id=file_record.id,
                    path=str(file_path),
                    error=str(exc),
                    exc_info=True,
                )
            return result

        # Log file indexing completion at INFO level (we only reach here on success)
        log.info(
            'indexing_file_complete',
            path=str(file_path),
            filename=filename,
            chunks_created=result.chunks_created,
            file_id=file_record.id
        )
        return result
    except _INDEXER_RUNTIME_EXCEPTIONS as exc:
        log.error('index_file_failed', path=str(file_path), error=str(exc), exc_info=True)
        return IndexResult(success=False, chunks_created=0, error=str(exc))


async def index_ingestion_item(
    db: aiosqlite.Connection,
    item: IngestionItem,
) -> IndexResult:
    # Provider-agnostic indexing entrypoint for normalized source items.
    return await index_file(
        db,
        item,
        source_provider=item.provider or FILESYSTEM_PROVIDER,
        entity_type=item.item_type or SOURCE_ENTITY_FILE,
    )


# ==============================================================================
# Public API — reindex_file
# ==============================================================================

async def reindex_file(
    db: aiosqlite.Connection,
    scanned: 'ScannedFile',
    *,
    source_provider: str = FILESYSTEM_PROVIDER,
    entity_type: str = SOURCE_ENTITY_FILE,
) -> IndexResult:
    # Re-index a file that has changed on disk.
    # Removes old chunks and embeddings, then runs the full pipeline.
    #
    # Steps:
    #   1. Look up existing file in SQLite
    #   2. Remove old chunks from SQLite
    #   3. Remove old embeddings from SQLite vector storage
    #   4. Extract new text
    #   5. Update file record in SQLite
    #   6. Chunk, embed, and store (same as index_file steps 4-7)

    path = scanned.path

    try:
        # -- Look up existing file ------------------------------------------------
        normalized_path = str(normalize_path(path, expand_user=False))
        existing = await get_file_by_source_identity(
            db,
            source_provider=source_provider,
            entity_type=entity_type,
            source_item_id=normalized_path,
        )
        if existing is None:
            existing = await get_file_by_path(db, normalized_path)

        if existing is None or existing.id is None:
            # File not in DB — treat as new
            log.debug('reindex_as_new', path=str(path))
            extractor = get_extractor(path)
            if extractor is None:
                return IndexResult(
                    success=False,
                    chunks_created=0,
                    error=f'No extractor for extension: {scanned.extension}',
                )
            return await index_file(
                db,
                path,
                extractor,
                source_provider=source_provider,
                entity_type=entity_type,
            )

        file_id = existing.id

        # Log reindex start at INFO level
        log.info(
            'reindexing_file_start',
            path=str(path),
            filename=scanned.filename,
            extension=scanned.extension,
            file_id=file_id
        )

        # -- Extract text ---------------------------------------------------------
        extractor = get_extractor(path)
        if extractor is None:
            log.error(
                'reindex_no_extractor',
                path=str(path),
                filename=scanned.filename,
                extension=scanned.extension
            )
            return IndexResult(
                success=False,
                chunks_created=0,
                error=f'No extractor for extension: {scanned.extension}',
            )

        doc = await asyncio.to_thread(extractor.extract, path)
        if doc.error:
            retryable = doc.metadata.get('retryable', 'true').lower() != 'false'
            error_code = doc.metadata.get('error_code')
            log.error(
                'reindex_extraction_failed',
                path=str(path),
                filename=scanned.filename,
                error=doc.error,
                error_code=error_code,
                retryable=retryable,
            )
            return IndexResult(
                success=False,
                chunks_created=0,
                error=doc.error,
                error_code=error_code,
                retryable=retryable,
            )

        # -- Classify -------------------------------------------------------------
        category = classify_file(path, scanned.extension)
        year = extract_year(path, doc.text)
        tags = generate_tags(path)
        file_metadata = _build_file_metadata(path, doc.metadata)

        # -- Remove old data ------------------------------------------------------
        try:
            await delete_chunks_for_file(db, file_id)
            await asyncio.to_thread(vector_store.delete_by_file_id, file_id)
        except _INDEXER_RUNTIME_EXCEPTIONS as exc:
            log.error('old_data_cleanup_failed', path=str(path), error=str(exc), exc_info=True)
            # Continue anyway — we'll overwrite

        # -- Update file record ---------------------------------------------------
        # Use content_hash from ScannedFile (already computed by crawler)
        # No need to re-read file bytes - scanned.content_hash is already available
        existing.content_hash = scanned.content_hash
        existing.source_provider = source_provider
        existing.entity_type = entity_type
        existing.source_item_id = str(normalize_path(path, expand_user=False))
        existing.size_bytes = scanned.size_bytes
        existing.extracted_text_preview = doc.preview_text or doc.text[:MAX_EXTRACTED_TEXT_PREVIEW]
        existing.category = category
        existing.tags = tags
        existing.year = year
        existing.extractor = file_metadata['extractor']
        existing.encoding = file_metadata['encoding']
        existing.language = file_metadata['language']
        existing.mime_type = file_metadata['mime_type']
        existing.ocr_used = bool(file_metadata['ocr_used'])
        existing.page_count = file_metadata['page_count']
        existing.tables_count = file_metadata['tables_count']
        existing.form_items_count = file_metadata['form_items_count']
        existing.key_value_items_count = file_metadata['key_value_items_count']
        existing.pictures_count = file_metadata['pictures_count']
        existing.document_hash = file_metadata['document_hash']
        existing.indexed_at = datetime.now(UTC)
        existing.modified_at = scanned.modified_at

        try:
            await update_file(db, existing)
        except _INDEXER_RUNTIME_EXCEPTIONS as exc:
            log.error('file_update_failed', path=str(path), error=str(exc), exc_info=True)
            return IndexResult(success=False, chunks_created=0, error=str(exc))

        # -- Post-process extracted text (clean glyph sequences, etc.) ------------
        cleaned_text = post_process_extracted_text(doc.text)

        # -- Chunk, embed, and store (shared logic) ------------------------------
        # Pass per-chunk metadata ranges from extractor (for docling formats with provenance)
        result = await _chunk_embed_store(
            db=db,
            file_id=file_id,
            text=cleaned_text,
            file_path=path,
            filename=scanned.filename,
            extension=scanned.extension,
            category=category.value,
            year=year,
            char_to_page_ranges=doc.char_to_page_ranges,
            char_to_block_type_ranges=doc.char_to_block_type_ranges,
            char_to_header_level_ranges=doc.char_to_header_level_ranges,
        )
        result.extractor = file_metadata['extractor'] if isinstance(file_metadata['extractor'], str) else None
        result.ocr_used = bool(file_metadata['ocr_used'])

        if result.success:
            log.info(
                'reindexing_file_complete',
                path=str(path),
                filename=scanned.filename,
                chunks_created=result.chunks_created,
                file_id=file_id
            )
        else:
            log.error(
                'reindexing_file_failed',
                path=str(path),
                filename=scanned.filename,
                error=result.error,
                file_id=file_id
            )
            await _cleanup_partial_file_data(
                db=db,
                file_id=file_id,
                path=str(path),
                reason='reindex_chunk_embed_store_failed',
            )
            # Remove file record on failed re-index to avoid files-without-chunks or partial state.
            try:
                await delete_file(db, file_id)
            except _INDEXER_RUNTIME_EXCEPTIONS as exc:
                log.warning(
                    'reindex_failed_file_delete_failed',
                    file_id=file_id,
                    path=str(path),
                    error=str(exc),
                )

        return result

    except _INDEXER_RUNTIME_EXCEPTIONS as exc:
        log.error('reindex_file_failed', path=str(path), error=str(exc), exc_info=True)
        return IndexResult(success=False, chunks_created=0, error=str(exc))


# ==============================================================================
# Public API — remove_file
# ==============================================================================

async def remove_file(
    db: aiosqlite.Connection,
    file: IndexedFile,
) -> bool:
    # Remove a file and all associated data from SQLite (including vector storage).
    # Returns True if the file was successfully removed.
    if file.id is None:
        log.warning('remove_file_no_id', path=file.path)
        return False

    try:
        # Delete file metadata first; chunks should be removed by CASCADE.
        deleted = await delete_file(db, file.id)
        if not deleted:
            return False
    except _INDEXER_RUNTIME_EXCEPTIONS as exc:
        log.error(
            'file_delete_failed',
            file_id=file.id,
            path=file.path,
            error=str(exc),
            exc_info=True,
        )
        return False

    # Best-effort vector cleanup (defensive in case foreign keys are not enforced).
    try:
        await asyncio.to_thread(vector_store.delete_by_file_id, file.id)
    except _INDEXER_RUNTIME_EXCEPTIONS as exc:
        log.warning(
            'vector_delete_post_file_delete_failed',
            file_id=file.id,
            path=file.path,
            error=str(exc),
        )
    return True
