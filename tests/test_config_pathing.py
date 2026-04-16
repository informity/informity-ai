from pathlib import Path

from informity.config import APP_SLUG, DirNames, Settings


def test_desktop_session_uses_app_data_model_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app_data = tmp_path / "app-data"
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("INFORMITY_TAURI_SESSION_TOKEN", "desktop-session-token")
    monkeypatch.delenv("INFORMITY_MODELS_DIR", raising=False)

    settings = Settings(app_data_dir=app_data, cache_dir=cache_dir)
    app_data_resolved = app_data.resolve()
    cache_dir_resolved = cache_dir.resolve()

    assert settings.app_data_dir == app_data_resolved
    assert settings.cache_dir == cache_dir_resolved
    assert settings.db_path == app_data_resolved / DirNames.DB / f"{APP_SLUG}.db"
    assert settings.logs_dir == app_data_resolved / DirNames.LOGS
    assert settings.models_dir == app_data_resolved / DirNames.MODELS / DirNames.LLM


def test_non_desktop_session_uses_app_data_model_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app_data = tmp_path / "app-data"
    cache_dir = tmp_path / "cache"
    monkeypatch.delenv("INFORMITY_TAURI_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("INFORMITY_MODELS_DIR", raising=False)

    settings = Settings(app_data_dir=app_data, cache_dir=cache_dir)
    app_data_resolved = app_data.resolve()

    assert settings.models_dir == app_data_resolved / DirNames.MODELS / DirNames.LLM


def test_legacy_root_db_path_is_forced_to_canonical_path(tmp_path: Path) -> None:
    app_data = tmp_path / "app-data"
    legacy_db_path = app_data / f"{APP_SLUG}.db"

    settings = Settings(app_data_dir=app_data, db_path=legacy_db_path)
    app_data_resolved = app_data.resolve()

    assert settings.db_path == app_data_resolved / DirNames.DB / f"{APP_SLUG}.db"


def test_ensure_directories_removes_empty_legacy_root_db_file(tmp_path: Path) -> None:
    app_data = tmp_path / "app-data"
    app_data.mkdir(parents=True, exist_ok=True)
    legacy_db_path = app_data / f"{APP_SLUG}.db"
    legacy_db_path.touch()

    settings = Settings(app_data_dir=app_data)
    settings.ensure_directories()

    assert not legacy_db_path.exists()
