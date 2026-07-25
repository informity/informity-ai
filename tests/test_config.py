"""Test module for tests test config."""

from __future__ import annotations

import builtins
import json

from informity import config
from informity.config import _get_default_supported_extensions


def test_default_supported_extensions_filter_excluded_extractable_extensions(monkeypatch) -> None:
    """Test default supported extensions filter excluded extractable extensions."""
    from informity.scanner.extractors import base as extractor_base

    monkeypatch.setattr(
        extractor_base,
        "get_all_extractable_extensions",
        lambda: [".pdf", ".txt", ".json", ".yaml", ".csv", ".toml", ".md"],
    )

    extensions = _get_default_supported_extensions()
    assert ".pdf" in extensions
    assert ".txt" in extensions
    assert ".csv" in extensions
    assert ".md" in extensions
    assert ".json" not in extensions
    assert ".yaml" not in extensions
    assert ".toml" not in extensions


def test_default_supported_extensions_uses_file_types_when_extractor_import_unavailable(
    monkeypatch,
) -> None:
    """Test default supported extensions uses file types when extractor import unavailable."""
    original_import = builtins.__import__

    def _patched_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        """Internal helper for patched import."""
        if name == "informity.scanner.extractors.base":
            raise ImportError("simulated extractor import failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _patched_import)

    extensions = _get_default_supported_extensions()
    assert ".pdf" in extensions
    assert ".jpg" in extensions
    assert ".txt" in extensions
    assert ".docx" in extensions
    assert ".json" not in extensions
    assert ".yaml" not in extensions
    assert ".yml" not in extensions
    assert ".toml" not in extensions


def test_load_config_file_values_migrates_legacy_specialization_keys(tmp_path, monkeypatch) -> None:
    """Test load config file values migrates legacy specialization keys."""
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "enable_specializations": True,
                "enabled_specialization_ids": ["legal", "financial", ""],
                "ui_theme": "onyx",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "_config_path_for_loader", lambda: config_path)

    loaded = config._load_config_file_values()

    assert loaded["enable_specializations"] is True
    assert loaded["enabled_specialization_ids"] == ["legal", "financial"]
    assert "enable_chat_roles" not in loaded
    assert "enabled_chat_role_ids" not in loaded

    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    assert persisted["enable_specializations"] is True
    assert persisted["enabled_specialization_ids"] == ["legal", "financial"]
    assert "enable_chat_roles" not in persisted
    assert "enabled_chat_role_ids" not in persisted


def test_reset_to_factory_defaults_uses_qwen3_6_35b(tmp_path, monkeypatch) -> None:
    """Test reset to factory defaults uses qwen3 6 35b."""
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(config, "_config_path_for_loader", lambda: config_path)
    monkeypatch.setattr(config, "ensure_private_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(config, "_apply_thread_limits_early", lambda: None)

    reset_settings = config.reset_to_factory_defaults()

    assert reset_settings.llm_model_id == "qwen3.6:35b"
    assert reset_settings.llm_model_filename == "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    assert persisted["llm_model_id"] == "qwen3.6:35b"
    assert persisted["llm_model_filename"] == "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
