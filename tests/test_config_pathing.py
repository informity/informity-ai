from pathlib import Path

from informity.config import APP_SLUG, DirNames, Settings


def test_desktop_session_uses_app_data_model_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app_data = tmp_path / "app-data"
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("INFORMITY_TAURI_SESSION_TOKEN", "desktop-session-token")

    settings = Settings(app_data_dir=app_data, cache_dir=cache_dir)
    app_data_resolved = app_data.resolve()
    cache_dir_resolved = cache_dir.resolve()

    assert settings.app_data_dir == app_data_resolved
    assert settings.cache_dir == cache_dir_resolved
    assert settings.db_path == app_data_resolved / DirNames.DB / f"{APP_SLUG}.db"
    assert settings.logs_dir == app_data_resolved / DirNames.LOGS
    assert settings.models_dir == app_data_resolved / DirNames.MODELS / DirNames.LLM
    assert settings.query_classifier_models_dir == (
        app_data_resolved / DirNames.MODELS / DirNames.QUERY_CLASSIFIER_MODELS
    )
    assert settings.diagnostics_models_dir.as_posix().endswith(
        f"/{DirNames.TOOLS}/{DirNames.DIAGNOSTICS}/{DirNames.DIAGNOSTICS_MODELS}"
    )


def test_non_desktop_session_uses_cache_model_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app_data = tmp_path / "app-data"
    cache_dir = tmp_path / "cache"
    monkeypatch.delenv("INFORMITY_TAURI_SESSION_TOKEN", raising=False)

    settings = Settings(app_data_dir=app_data, cache_dir=cache_dir)
    cache_dir_resolved = cache_dir.resolve()

    assert settings.models_dir == cache_dir_resolved / DirNames.LLM
    assert settings.query_classifier_models_dir == (
        cache_dir_resolved / DirNames.QUERY_CLASSIFIER_MODELS
    )
    assert settings.diagnostics_models_dir.as_posix().endswith(
        f"/{DirNames.TOOLS}/{DirNames.DIAGNOSTICS}/{DirNames.DIAGNOSTICS_MODELS}"
    )
