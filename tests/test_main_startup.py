from informity import main
from informity.config import settings


def test_startup_bootstrap_mode_enables_temporary_non_privacy(monkeypatch) -> None:
    original_full_privacy = settings.full_privacy
    original_llm_local_only = settings.llm_local_only
    original_embedding_offline = settings.embedding_offline

    calls: list[dict[str, bool]] = []

    monkeypatch.setattr(main, "are_required_models_cached", lambda: False)
    monkeypatch.setattr(
        main,
        "configure_hf_environment",
        lambda **kwargs: calls.append(kwargs),
    )

    try:
        settings.full_privacy = True
        settings.llm_local_only = True
        settings.embedding_offline = True

        assert main._enable_startup_bootstrap_mode_if_needed() is True
        assert settings.full_privacy is False
        assert settings.llm_local_only is False
        assert settings.embedding_offline is False
        assert calls == [{"fail_on_missing_full_privacy_models": False}]
    finally:
        settings.full_privacy = original_full_privacy
        settings.llm_local_only = original_llm_local_only
        settings.embedding_offline = original_embedding_offline


def test_startup_bootstrap_mode_noops_when_models_cached(monkeypatch) -> None:
    original_full_privacy = settings.full_privacy
    original_llm_local_only = settings.llm_local_only
    original_embedding_offline = settings.embedding_offline

    calls: list[dict[str, bool]] = []

    monkeypatch.setattr(main, "are_required_models_cached", lambda: True)
    monkeypatch.setattr(
        main,
        "configure_hf_environment",
        lambda **kwargs: calls.append(kwargs),
    )

    try:
        settings.full_privacy = True
        settings.llm_local_only = True
        settings.embedding_offline = True

        assert main._enable_startup_bootstrap_mode_if_needed() is False
        assert settings.full_privacy is True
        assert settings.llm_local_only is True
        assert settings.embedding_offline is True
        assert calls == []
    finally:
        settings.full_privacy = original_full_privacy
        settings.llm_local_only = original_llm_local_only
        settings.embedding_offline = original_embedding_offline
