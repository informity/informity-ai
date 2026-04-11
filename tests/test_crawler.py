# ==============================================================================
# Informity AI — Crawler Tests
# Tests directory traversal, ignore patterns, extension filtering,
# hash computation, and change detection.
# ==============================================================================

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from informity.db.models import IndexedFile
from informity.scanner.crawler import (
    ChangeSet,
    ScannedFile,
    _compute_file_hash_and_stat,
    _walk_directory,
    compare_with_db,
    scan_directories,
    should_ignore,
)

# ==============================================================================
# Helpers
# ==============================================================================


def _create_file_tree(base: Path, structure: dict) -> None:
    # Recursively create a directory structure from a nested dict.
    # Keys are names, values are either strings (file content) or dicts (subdirs).
    for name, content in structure.items():
        path = base / name
        if isinstance(content, dict):
            path.mkdir(parents=True, exist_ok=True)
            _create_file_tree(path, content)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")


def _sha256(text: str) -> str:
    # Compute SHA-256 of a string for test assertions.
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_indexed_file(
    path: str,
    content_hash: str,
    *,
    source_provider: str = "filesystem",
    entity_type: str = "file",
    source_item_id: str | None = None,
) -> IndexedFile:
    # Create a minimal IndexedFile for change detection tests.
    return IndexedFile(
        id=1,
        source_provider=source_provider,
        entity_type=entity_type,
        source_item_id=source_item_id or path,
        path=path,
        filename=Path(path).name,
        extension=Path(path).suffix,
        size_bytes=100,
        content_hash=content_hash,
        extracted_text_preview="preview",
        category="plaintext",
        modified_at=datetime.now(tz=UTC),
    )


# ==============================================================================
# Directory Traversal Tests
# ==============================================================================


class TestDirectoryTraversal:
    def test_finds_supported_files(self, tmp_path: Path) -> None:
        _create_file_tree(
            tmp_path,
            {
                "readme.txt": "hello",
                "notes.md": "# notes",
                "data.csv": "a,b,c",
            },
        )
        results = _walk_directory(
            tmp_path,
            ignore_patterns=[],
            extensions={".txt", ".md", ".csv"},
            follow_symlinks=False,
        )
        names = {p.name for p in results}
        assert names == {"readme.txt", "notes.md", "data.csv"}

    def test_recurses_subdirectories(self, tmp_path: Path) -> None:
        _create_file_tree(
            tmp_path,
            {
                "top.txt": "top",
                "sub": {
                    "middle.txt": "middle",
                    "deep": {
                        "bottom.txt": "bottom",
                    },
                },
            },
        )
        results = _walk_directory(
            tmp_path,
            ignore_patterns=[],
            extensions={".txt"},
            follow_symlinks=False,
        )
        names = {p.name for p in results}
        assert names == {"top.txt", "middle.txt", "bottom.txt"}

    def test_empty_directory(self, tmp_path: Path) -> None:
        results = _walk_directory(
            tmp_path,
            ignore_patterns=[],
            extensions={".txt"},
            follow_symlinks=False,
        )
        assert results == []

    def test_permission_error_handled(self, tmp_path: Path) -> None:
        restricted = tmp_path / "restricted"
        restricted.mkdir()
        restricted.chmod(0o000)
        try:
            results = _walk_directory(
                restricted,
                ignore_patterns=[],
                extensions={".txt"},
                follow_symlinks=False,
            )
            assert results == []
        finally:
            restricted.chmod(0o755)

    def test_skips_symlinks_by_default(self, tmp_path: Path) -> None:
        real_file = tmp_path / "real.txt"
        real_file.write_text("content")
        link = tmp_path / "link.txt"
        link.symlink_to(real_file)

        results = _walk_directory(
            tmp_path,
            ignore_patterns=[],
            extensions={".txt"},
            follow_symlinks=False,
        )
        names = {p.name for p in results}
        assert "real.txt" in names
        assert "link.txt" not in names

    def test_follows_symlinks_when_enabled(self, tmp_path: Path) -> None:
        real_file = tmp_path / "real.txt"
        real_file.write_text("content")
        link = tmp_path / "link.txt"
        link.symlink_to(real_file)

        results = _walk_directory(
            tmp_path,
            ignore_patterns=[],
            extensions={".txt"},
            follow_symlinks=True,
        )
        names = {p.name for p in results}
        assert "real.txt" in names
        assert "link.txt" in names


# ==============================================================================
# Ignore Pattern Tests
# ==============================================================================


class TestIgnorePatterns:
    def test_ignore_by_filename(self) -> None:
        assert should_ignore(Path("/project/.DS_Store"), [".DS_Store"])
        assert should_ignore(Path("/project/.git"), [".git"])

    def test_ignore_by_glob_pattern(self) -> None:
        assert should_ignore(Path("/project/archive.app"), ["*.app"])
        assert should_ignore(Path("/project/test.pyc"), ["*.pyc"])

    def test_ignore_by_path_component(self) -> None:
        assert should_ignore(Path("/project/node_modules/pkg/index.js"), ["node_modules"])
        assert should_ignore(Path("/home/user/.git/config"), [".git"])

    def test_no_match_returns_false(self) -> None:
        assert not should_ignore(Path("/project/readme.txt"), [".git", "node_modules"])
        assert not should_ignore(Path("/project/src/main.py"), ["*.app"])

    def test_multiple_patterns(self) -> None:
        patterns = [".git", "node_modules", "__pycache__", ".DS_Store", "*.app"]
        assert should_ignore(Path("/project/.git"), patterns)
        assert should_ignore(Path("/project/node_modules"), patterns)
        assert should_ignore(Path("/project/__pycache__"), patterns)
        assert should_ignore(Path("/project/.DS_Store"), patterns)
        assert should_ignore(Path("/project/MyApp.app"), patterns)
        assert not should_ignore(Path("/project/readme.md"), patterns)

    def test_ignore_filters_walk_results(self, tmp_path: Path) -> None:
        _create_file_tree(
            tmp_path,
            {
                "keep.txt": "keep",
                ".git": {
                    "config": "git stuff",
                },
                "node_modules": {
                    "pkg": {
                        "index.txt": "module",
                    },
                },
                "src": {
                    "main.txt": "main code",
                    "__pycache__": {
                        "cache.txt": "cached",
                    },
                },
            },
        )
        results = _walk_directory(
            tmp_path,
            ignore_patterns=[".git", "node_modules", "__pycache__"],
            extensions={".txt"},
            follow_symlinks=False,
        )
        names = {p.name for p in results}
        assert names == {"keep.txt", "main.txt"}


# ==============================================================================
# Extension Filtering Tests
# ==============================================================================


class TestExtensionFiltering:
    def test_includes_only_supported_extensions(self, tmp_path: Path) -> None:
        _create_file_tree(
            tmp_path,
            {
                "doc.txt": "text",
                "doc.pdf": "pdf data",
                "image.png": "png data",
                "video.mp4": "mp4 data",
                "notes.md": "markdown",
            },
        )
        results = _walk_directory(
            tmp_path,
            ignore_patterns=[],
            extensions={".txt", ".md"},
            follow_symlinks=False,
        )
        names = {p.name for p in results}
        assert names == {"doc.txt", "notes.md"}

    def test_case_insensitive_extensions(self, tmp_path: Path) -> None:
        _create_file_tree(
            tmp_path,
            {
                "upper.TXT": "text",
                "mixed.Md": "markdown",
                "lower.txt": "text too",
            },
        )
        results = _walk_directory(
            tmp_path,
            ignore_patterns=[],
            extensions={".txt", ".md"},
            follow_symlinks=False,
        )
        names = {p.name for p in results}
        assert "upper.TXT" in names
        assert "mixed.Md" in names
        assert "lower.txt" in names

    def test_empty_extension_set(self, tmp_path: Path) -> None:
        _create_file_tree(tmp_path, {"file.txt": "content"})
        results = _walk_directory(
            tmp_path,
            ignore_patterns=[],
            extensions=set(),
            follow_symlinks=False,
        )
        assert results == []

    def test_no_extension_file_excluded(self, tmp_path: Path) -> None:
        _create_file_tree(
            tmp_path,
            {
                "Makefile": "make stuff",
                "readme.txt": "text",
            },
        )
        results = _walk_directory(
            tmp_path,
            ignore_patterns=[],
            extensions={".txt"},
            follow_symlinks=False,
        )
        names = {p.name for p in results}
        assert names == {"readme.txt"}


# ==============================================================================
# Hash Computation Tests
# ==============================================================================


class TestHashComputation:
    def test_hash_matches_expected(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        content = "Hello, Informity!"
        f.write_text(content, encoding="utf-8")
        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
        result = _compute_file_hash_and_stat(str(f))
        assert result is not None
        assert result[0] == expected

    def test_empty_file_hash(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("")
        expected = hashlib.sha256(b"").hexdigest()
        result = _compute_file_hash_and_stat(str(f))
        assert result is not None
        assert result[0] == expected

    def test_hash_is_deterministic(self, tmp_path: Path) -> None:
        f = tmp_path / "det.txt"
        f.write_text("deterministic content")
        r1 = _compute_file_hash_and_stat(str(f))
        r2 = _compute_file_hash_and_stat(str(f))
        assert r1 is not None
        assert r2 is not None
        h1 = r1[0]
        h2 = r2[0]
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content A")
        f2.write_text("content B")
        r1 = _compute_file_hash_and_stat(str(f1))
        r2 = _compute_file_hash_and_stat(str(f2))
        assert r1 is not None
        assert r2 is not None
        assert r1[0] != r2[0]

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert _compute_file_hash_and_stat(str(tmp_path / "missing.txt")) is None

    def test_binary_file_hash(self, tmp_path: Path) -> None:
        f = tmp_path / "binary.bin"
        data = bytes(range(256))
        f.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        result = _compute_file_hash_and_stat(str(f))
        assert result is not None
        assert result[0] == expected


# ==============================================================================
# scan_directories Integration Tests
# ==============================================================================


class TestScanDirectories:
    def test_scan_returns_scanned_files(self, tmp_path: Path) -> None:
        _create_file_tree(
            tmp_path,
            {
                "readme.txt": "hello world",
                "notes.md": "# title",
            },
        )
        results = scan_directories(
            directories=[tmp_path],
            ignore_patterns=[],
            supported_extensions=[".txt", ".md"],
        )
        names = {sf.filename for sf in results}
        assert names == {"readme.txt", "notes.md"}

    def test_scan_uses_settings_supported_extensions_when_not_passed(self, tmp_path: Path) -> None:
        # When supported_extensions is not passed (e.g. API scan), crawler uses
        # settings.supported_extensions so that file types selected in Settings are respected.
        _create_file_tree(
            tmp_path,
            {
                "a.txt": "text",
                "b.pdf": "pdf",
                "c.md": "markdown",
            },
        )
        with patch('informity.scanner.crawler.settings') as mock_settings:
            mock_settings.supported_extensions = ['.txt', '.md']
            mock_settings.watched_directories = []  # unused when directories= is passed
            mock_settings.ignore_patterns = []
            mock_settings.follow_symlinks = False
            results = scan_directories(
                directories=[tmp_path],
                ignore_patterns=[],
                # Do not pass supported_extensions — must use settings
            )
        names = {sf.filename for sf in results}
        assert names == {'a.txt', 'c.md'}
        assert 'b.pdf' not in names

    def test_scan_populates_fields(self, tmp_path: Path) -> None:
        content = "test content for hashing"
        f = tmp_path / "data.txt"
        f.write_text(content, encoding="utf-8")

        results = scan_directories(
            directories=[tmp_path],
            ignore_patterns=[],
            supported_extensions=[".txt"],
        )
        assert len(results) == 1
        sf = results[0]
        assert sf.filename == "data.txt"
        assert sf.extension == ".txt"
        assert sf.size_bytes == f.stat().st_size
        assert sf.content_hash == _sha256(content)
        assert isinstance(sf.modified_at, datetime)
        assert sf.path == f.resolve()

    def test_scan_nonexistent_directory(self, tmp_path: Path) -> None:
        results = scan_directories(
            directories=[tmp_path / "nonexistent"],
            ignore_patterns=[],
            supported_extensions=[".txt"],
        )
        assert results == []

    def test_scan_empty_directory_list(self) -> None:
        results = scan_directories(
            directories=[],
            ignore_patterns=[],
            supported_extensions=[".txt"],
        )
        assert results == []

    def test_scan_respects_ignore_patterns(self, tmp_path: Path) -> None:
        _create_file_tree(
            tmp_path,
            {
                "keep.txt": "keep",
                ".git": {
                    "config.txt": "git",
                },
                "src": {
                    "main.txt": "code",
                },
            },
        )
        results = scan_directories(
            directories=[tmp_path],
            ignore_patterns=[".git"],
            supported_extensions=[".txt"],
        )
        names = {sf.filename for sf in results}
        assert "keep.txt" in names
        assert "main.txt" in names
        assert "config.txt" not in names

    def test_scan_multiple_directories(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "dir_a"
        dir_b = tmp_path / "dir_b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "a.txt").write_text("alpha")
        (dir_b / "b.txt").write_text("beta")

        results = scan_directories(
            directories=[dir_a, dir_b],
            ignore_patterns=[],
            supported_extensions=[".txt"],
        )
        names = {sf.filename for sf in results}
        assert names == {"a.txt", "b.txt"}


# ==============================================================================
# Change Detection Tests
# ==============================================================================


class TestChangeDetection:
    def test_all_new_files(self, tmp_path: Path) -> None:
        scanned = [
            ScannedFile(
                path=tmp_path / "new.txt",
                filename="new.txt",
                extension=".txt",
                size_bytes=100,
                content_hash="aaa",
                modified_at=datetime.now(tz=UTC),
            ),
        ]
        changeset = compare_with_db(scanned, db_files=[])
        assert len(changeset.new) == 1
        assert len(changeset.changed) == 0
        assert len(changeset.unchanged) == 0
        assert len(changeset.deleted) == 0

    def test_unchanged_file(self) -> None:
        path_str = "/docs/readme.txt"
        scanned = [
            ScannedFile(
                path=Path(path_str),
                filename="readme.txt",
                extension=".txt",
                size_bytes=100,
                content_hash="same_hash",
                modified_at=datetime.now(tz=UTC),
            ),
        ]
        db_files = [_make_indexed_file(path_str, "same_hash")]

        changeset = compare_with_db(scanned, db_files)
        assert len(changeset.new) == 0
        assert len(changeset.changed) == 0
        assert len(changeset.unchanged) == 1
        assert len(changeset.deleted) == 0

    def test_changed_file(self) -> None:
        path_str = "/docs/readme.txt"
        scanned = [
            ScannedFile(
                path=Path(path_str),
                filename="readme.txt",
                extension=".txt",
                size_bytes=200,
                content_hash="new_hash",
                modified_at=datetime.now(tz=UTC),
            ),
        ]
        db_files = [_make_indexed_file(path_str, "old_hash")]

        changeset = compare_with_db(scanned, db_files)
        assert len(changeset.new) == 0
        assert len(changeset.changed) == 1
        assert len(changeset.unchanged) == 0
        assert len(changeset.deleted) == 0

    def test_deleted_file(self) -> None:
        db_files = [_make_indexed_file("/docs/gone.txt", "hash")]
        changeset = compare_with_db(scanned=[], db_files=db_files)
        assert len(changeset.new) == 0
        assert len(changeset.changed) == 0
        assert len(changeset.unchanged) == 0
        assert len(changeset.deleted) == 1
        assert changeset.deleted[0].path == "/docs/gone.txt"

    def test_mixed_changes(self) -> None:
        scanned = [
            ScannedFile(
                path=Path("/docs/new.txt"),
                filename="new.txt",
                extension=".txt",
                size_bytes=50,
                content_hash="new_hash",
                modified_at=datetime.now(tz=UTC),
            ),
            ScannedFile(
                path=Path("/docs/same.txt"),
                filename="same.txt",
                extension=".txt",
                size_bytes=100,
                content_hash="unchanged_hash",
                modified_at=datetime.now(tz=UTC),
            ),
            ScannedFile(
                path=Path("/docs/edited.txt"),
                filename="edited.txt",
                extension=".txt",
                size_bytes=150,
                content_hash="updated_hash",
                modified_at=datetime.now(tz=UTC),
            ),
        ]
        db_files = [
            _make_indexed_file("/docs/same.txt", "unchanged_hash"),
            _make_indexed_file("/docs/edited.txt", "original_hash"),
            _make_indexed_file("/docs/removed.txt", "removed_hash"),
        ]

        changeset = compare_with_db(scanned, db_files)
        assert len(changeset.new) == 1
        assert changeset.new[0].filename == "new.txt"
        assert len(changeset.changed) == 1
        assert changeset.changed[0].filename == "edited.txt"
        assert len(changeset.unchanged) == 1
        assert changeset.unchanged[0].filename == "same.txt"
        assert len(changeset.deleted) == 1
        assert changeset.deleted[0].path == "/docs/removed.txt"

    def test_empty_scanned_and_db(self) -> None:
        changeset = compare_with_db(scanned=[], db_files=[])
        assert len(changeset.new) == 0
        assert len(changeset.changed) == 0
        assert len(changeset.unchanged) == 0
        assert len(changeset.deleted) == 0

    def test_compare_is_scoped_to_provider_and_entity_type(self) -> None:
        scanned = [
            ScannedFile(
                path=Path("/docs/keep.txt"),
                filename="keep.txt",
                extension=".txt",
                size_bytes=100,
                content_hash="same_hash",
                modified_at=datetime.now(tz=UTC),
            ),
        ]
        db_files = [
            _make_indexed_file("/docs/keep.txt", "same_hash", source_provider="filesystem", entity_type="file"),
            _make_indexed_file(
                "source://mail.apple/mail/msg-1",
                "mail_hash",
                source_provider="mail.apple",
                entity_type="mail",
                source_item_id="msg-1",
            ),
        ]

        changeset = compare_with_db(
            scanned,
            db_files=db_files,
            source_provider="filesystem",
            entity_type="file",
        )
        assert len(changeset.new) == 0
        assert len(changeset.changed) == 0
        assert len(changeset.unchanged) == 1
        assert len(changeset.deleted) == 0

    def test_changeset_dataclass_fields(self) -> None:
        changeset = ChangeSet(new=[], changed=[], unchanged=[], deleted=[])
        assert hasattr(changeset, "new")
        assert hasattr(changeset, "changed")
        assert hasattr(changeset, "unchanged")
        assert hasattr(changeset, "deleted")
