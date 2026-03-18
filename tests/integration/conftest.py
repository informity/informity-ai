# ==============================================================================
# Informity AI — Integration Test Fixtures
# Sets up a seeded SQLite corpus (no ML models) for retrieval pipeline tests.
#
# Strategy:
#   - embedder.embed_query  → deterministic unit vector (no embedding model)
#   - reranker.rerank       → identity (preserves order, sets score=0.5)
#   - settings.db_path      → temporary test DB seeded with 15 documents / 30 chunks
#   - vector_store          → points to the same temp DB via reset thread-local
#
# Tests assert structural properties (filter correctness, top-k, required fields)
# rather than semantic ranking quality, making them fast and non-flaky.
# ==============================================================================

import math
import sqlite3
import struct
import threading

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 768  # nomic-embed-text-v1.5 dimension (matches default settings)


def _make_unit_vector() -> list[float]:
    """768-dim unit vector (all-ones direction, L2-normalized)."""
    v = 1.0 / math.sqrt(EMBEDDING_DIM)
    return [v] * EMBEDDING_DIM


def _serialize_float32(vector: list[float]) -> bytes:
    """Serialize to little-endian float32 array — matches sqlite_vec format."""
    return struct.pack(f'<{len(vector)}f', *vector)


# ---------------------------------------------------------------------------
# Test corpus: 15 documents × 2 chunks = 30 chunks
# Tuple: (filename, extension, category, year_or_None, [chunk1, chunk2])
# ---------------------------------------------------------------------------

CORPUS: list[tuple[str, str, str, int | None, list[str]]] = [
    (
        'annual_report_2021.pdf', '.pdf', 'finance', 2021,
        [
            'Revenue for fiscal year 2021 increased by 12% to $1.2 billion.',
            'Operating expenses in 2021 were reduced by 5% through cost optimisation.',
        ],
    ),
    (
        'annual_report_2022.pdf', '.pdf', 'finance', 2022,
        [
            'Revenue for fiscal year 2022 reached $1.4 billion, exceeding all targets.',
            'Net income in 2022 grew to $180 million with improved operating margins.',
        ],
    ),
    (
        'annual_report_2023.pdf', '.pdf', 'finance', 2023,
        [
            'Revenue for fiscal year 2023 was $1.6 billion, driven by product expansion.',
            'Operating profit in 2023 increased 15 percent year over year.',
        ],
    ),
    (
        'quarterly_results_2022.pdf', '.pdf', 'finance', 2022,
        [
            'Q4 2022 results showed strong performance across all business units.',
            'Quarterly revenue in Q3 2022 was $340 million, up 8% quarter over quarter.',
        ],
    ),
    (
        'employee_handbook.pdf', '.pdf', 'hr', None,
        [
            'All employees are expected to adhere to the company code of conduct.',
            'Performance review cycles occur twice per year in June and December.',
        ],
    ),
    (
        'privacy_policy.md', '.md', 'legal', None,
        [
            'We collect personal data only with explicit user consent as required by GDPR.',
            'Data retention periods are defined in Schedule A of this Privacy Policy.',
        ],
    ),
    (
        'terms_of_service.md', '.md', 'legal', None,
        [
            'By accessing this service you agree to the terms and conditions outlined herein.',
            'Termination of service requires 30 days written notice from either party.',
        ],
    ),
    (
        'technical_specification.md', '.md', 'technical', None,
        [
            'The API supports REST and GraphQL endpoints with JSON serialisation.',
            'Authentication uses OAuth 2.0 with JWT tokens and a 24-hour expiry.',
        ],
    ),
    (
        'product_roadmap.md', '.md', 'technical', None,
        [
            'Phase 1 of the product roadmap targets mobile platform launch in Q2.',
            'The integration with third-party analytics platforms is planned for Q3.',
        ],
    ),
    (
        'meeting_notes_q1_2023.txt', '.txt', 'operations', 2023,
        [
            'Q1 2023 planning meeting covered budget allocation and headcount decisions.',
            'Action items from the Q1 meeting include hiring three engineers by March.',
        ],
    ),
    (
        'meeting_notes_q2_2023.txt', '.txt', 'operations', 2023,
        [
            'Q2 2023 retrospective highlighted delays in the product launch timeline.',
            'Team velocity in Q2 2023 improved after the process changes introduced in April.',
        ],
    ),
    (
        'sales_summary_2022.txt', '.txt', 'finance', 2022,
        [
            'Total sales in 2022 were $1.1 billion across all product lines.',
            'Top-performing region in 2022 was North America with 45% of total sales.',
        ],
    ),
    (
        'marketing_strategy.pdf', '.pdf', 'marketing', None,
        [
            'The brand awareness campaign targets professionals aged 25 to 40.',
            'Digital marketing budget is allocated 60% to paid social and 40% to SEO.',
        ],
    ),
    (
        'onboarding_guide.md', '.md', 'hr', None,
        [
            'New employees should complete the IT security training within the first week.',
            'The onboarding buddy programme pairs new hires with a senior team member.',
        ],
    ),
    (
        'release_notes.md', '.md', 'technical', None,
        [
            'Version 2.4.0 introduces dark mode support and performance improvements.',
            'Bug fixes in this release address the reported login timeout issue.',
        ],
    ),
]


# ---------------------------------------------------------------------------
# Session-scoped fixture: create and seed the corpus DB once per test run
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def corpus_db_path(tmp_path_factory):
    """
    Create a temporary SQLite database seeded with the test corpus.
    Returns the Path to the DB file.

    Uses sqlite-vec extension for vec_distance_cosine support.
    All vectors are the same 768-dim unit vector — semantically neutral,
    but sufficient for testing SQL-level filters and structural properties.
    """
    import sqlite_vec

    from informity.db.sqlite import _SCHEMA_SQL

    tmp_dir = tmp_path_factory.mktemp('retrieval_integration')
    db_path = tmp_dir / 'corpus.db'

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA foreign_keys=ON')
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.executescript(_SCHEMA_SQL)
    conn.execute('CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON chunks(parent_id)')

    unit_vec = _make_unit_vector()
    vec_blob = _serialize_float32(unit_vec)
    chunk_id = 1

    for file_id, (filename, extension, category, year, chunk_texts) in enumerate(CORPUS, start=1):
        path = f'/test/corpus/{filename}'
        conn.execute(
            """
            INSERT INTO files
                (id, path, filename, extension, category, year, size_bytes, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (file_id, path, filename, extension, category, year, 1024),
        )
        for chunk_index, chunk_text in enumerate(chunk_texts):
            conn.execute(
                """
                INSERT INTO chunks (id, file_id, chunk_index, content, parent_id)
                VALUES (?, ?, ?, ?, NULL)
                """,
                (chunk_id, file_id, chunk_index, chunk_text),
            )
            conn.execute(
                """
                INSERT INTO vec_chunks
                    (chunk_id, file_id, file_path, chunk_text, vector, year,
                     filename, extension, category)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (chunk_id, file_id, path, chunk_text, vec_blob,
                 year, filename, extension, category),
            )
            chunk_id += 1

    conn.execute(
        'INSERT INTO schema_version (version) VALUES (?)',
        (1,),
    )
    conn.commit()
    conn.close()

    return db_path


# ---------------------------------------------------------------------------
# Session-scoped autouse fixture: redirect DB and stub ML models
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session', autouse=True)
def _patch_db_and_models(corpus_db_path):
    """
    Redirect settings.db_path to the test corpus DB and stub all ML model
    calls. Must run before any test in this package uses retrieve_chunks.

    Patches (session-scoped via object.__setattr__ / patch.object):
      - informity.config.settings.db_path  → corpus_db_path
      - informity.indexer.embedder.embedder.embed_query → returns unit vector
      - informity.indexer.reranker.reranker.rerank      → preserves order, score=0.5
      - informity.db.vectors.vector_store._thread_local → reset so new connections
        pick up the new db_path
    """
    from unittest.mock import patch

    from informity.config import settings
    from informity.db.vectors import vector_store
    from informity.indexer.embedder import embedder
    from informity.indexer.reranker import reranker

    original_db_path = settings.db_path

    # Redirect DB path so vector_store and aiosqlite both open the test DB
    object.__setattr__(settings, 'db_path', corpus_db_path)
    # Force vector_store to create a fresh thread-local connection to the new DB
    vector_store._thread_local = threading.local()

    unit_vec = _make_unit_vector()

    def _mock_embed(query: str) -> list[float]:
        return unit_vec

    def _mock_rerank(query: str, chunks: list[dict]) -> list[dict]:
        return [{**chunk, 'score': 0.5} for chunk in chunks]

    with (
        patch.object(embedder, 'embed_query', side_effect=_mock_embed),
        patch.object(reranker, 'rerank', side_effect=_mock_rerank),
    ):
        yield

    # Restore
    object.__setattr__(settings, 'db_path', original_db_path)
    vector_store._thread_local = threading.local()
