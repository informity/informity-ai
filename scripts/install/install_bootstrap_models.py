# ==============================================================================
# Informity AI — Bootstrap models for install script
# Downloads embedding model, reranker (cross-encoder), and optional LLM
# into app data, then writes config.json with embedding_offline
# and llm_local_only set to true so the app always uses cached models after install.
# Run from repo root: uv run python scripts/install/install_bootstrap_models.py
# Requires: INFORMITY_APP_DATA_DIR (default: ~/.informity)
# and install.conf.json
# ==============================================================================

"""Bootstrap the models needed for a local Informity installation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

APP_DATA_DIRNAME = ".informity"  # pylint: disable=invalid-name


# CRITICAL: Set HF cache paths BEFORE any imports that might initialize
# huggingface_hub. This ensures models are downloaded to app data cache, not
# ~/.cache/huggingface/hub/.
def _default_app_data_dir() -> Path:
    """ default app data dir."""
    return Path.home() / APP_DATA_DIRNAME


def _setup_hf_cache_early() -> None:
    """Set HF_HOME and HF_HUB_CACHE env vars before any HF imports."""
    raw_cache_dir = os.environ.get("INFORMITY_CACHE_DIR", "")
    if raw_cache_dir:
        raw_path = Path(raw_cache_dir)
        cache_dir = raw_path.resolve() if not raw_path.is_absolute() else raw_path
    else:
        # Match config.py default: ~/.informity/cache
        cache_dir = _default_app_data_dir() / "cache"

    hf_home = cache_dir / "huggingface"
    hf_hub = hf_home / "hub"
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_hub)


# Import default reranker model from config
# Note: We import from config module which may have dependencies, but this constant
# is defined early and doesn't require any heavy imports
try:
    from informity import config as _informity_config

    APP_DATA_DIRNAME = _informity_config.APP_DATA_DIRNAME  # pylint: disable=invalid-name
    DEFAULT_RERANKER_MODEL = _informity_config.DEFAULT_RERANKER_MODEL  # pylint: disable=invalid-name
    _is_hf_model_cached = _informity_config.is_hf_model_cached
except ImportError:
    # Fallback if import fails (shouldn't happen in normal usage)
    # sentence-transformers uses cross-encoder/ prefix
    DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # pylint: disable=invalid-name

    def _is_hf_model_cached(_model_name: str, _hf_hub_cache: Path) -> bool:
        """ is hf model cached."""
        return False


# Set HF cache paths immediately
_setup_hf_cache_early()


def _app_data_dir() -> Path:
    """Resolve app data dir. Default matches config.py (~/.informity)."""
    raw = os.environ.get("INFORMITY_APP_DATA_DIR", "")
    if raw:
        p = Path(raw)
        return p.resolve() if not p.is_absolute() else p
    return _default_app_data_dir()


def _load_install_config(config_path: Path) -> dict:
    """Load and validate the installer configuration JSON."""
    if not config_path.exists():
        raise SystemExit(f"Install config not found: {config_path}")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("Install config must be a JSON object.")
    return data


def _get_repo_root() -> Path:
    """Find project root (directory containing pyproject.toml)."""
    current = Path(__file__).resolve().parent.parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()


def _ensure_dirs(app_data: Path) -> None:
    """Create directories for models, cache, and user data."""
    from informity.config import DirNames

    cache_root = _get_cache_dir()
    (app_data / DirNames.MODELS / DirNames.LLM).mkdir(parents=True, exist_ok=True)
    (cache_root / DirNames.HUGGINGFACE / DirNames.HUB).mkdir(parents=True, exist_ok=True)
    (cache_root / DirNames.DOCLING).mkdir(parents=True, exist_ok=True)
    (app_data / DirNames.LOGS).mkdir(parents=True, exist_ok=True)


def _download_embedding_model(_app_data: Path, model_id: str) -> None:
    """Download embedding model using sentence-transformers (PyTorch)."""
    hf_hub_cache = _get_cache_dir() / "huggingface" / "hub"
    if _is_hf_model_cached(model_id, hf_hub_cache):
        print(f"Embedding model already cached: {model_id}")
        return

    print(f"Downloading embedding model: {model_id}")
    try:
        from sentence_transformers import SentenceTransformer

        # HF cache paths already set by _setup_hf_cache_early() at module import
        model = SentenceTransformer(model_id, trust_remote_code=True)
        model.encode(["bootstrap"])
    except Exception as e:
        raise SystemExit(f"Failed to download embedding model: {e}") from e
    print("Embedding model cached.")


def _download_reranker_model(_app_data: Path, model_id: str) -> None:
    """Download the cross-encoder reranker using sentence-transformers (PyTorch)."""
    hf_hub_cache = _get_cache_dir() / "huggingface" / "hub"
    if _is_hf_model_cached(model_id, hf_hub_cache):
        print(f"Reranker model already cached: {model_id}")
        return

    print(f"Downloading reranker (cross-encoder): {model_id}")
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_id)
        model.predict([["bootstrap", "dummy passage"]])
    except Exception as e:
        raise SystemExit(f"Failed to download reranker model: {e}") from e
    print("Reranker model cached.")


def _get_cache_dir() -> Path:
    """Resolve cache root the same way config.py does (app_data_dir/cache by default)."""
    from informity.config import DirNames

    raw = os.environ.get("INFORMITY_CACHE_DIR", "")
    if raw:
        p = Path(raw)
        return p.resolve() if not p.is_absolute() else p
    return _app_data_dir() / DirNames.CACHE


def _download_docling_models(_app_data: Path) -> None:
    """Download docling models into the unified cache directory.

    Store under {cache_dir}/docling so docling finds them at runtime when
    DOCLING_ARTIFACTS_PATH is set to that path (same as app's docling extractor).
    """
    from informity.config import DirNames

    cache_dir = _get_cache_dir()
    docling_cache = cache_dir / DirNames.DOCLING
    docling_cache.mkdir(parents=True, exist_ok=True)

    from informity.config import (
        _is_docling_cached,
        ensure_docling_rapidocr_cache_compat,
        get_docling_download_options,
    )

    if _is_docling_cached(cache_dir):
        ensure_docling_rapidocr_cache_compat(cache_dir)
        print(f"Docling models already cached: {docling_cache}")
        return

    os.environ["DOCLING_ARTIFACTS_PATH"] = str(docling_cache)
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    print(f"Downloading docling models -> {docling_cache}")
    try:
        from docling.utils.model_downloader import download_models

        download_models(
            output_dir=docling_cache,
            **get_docling_download_options(progress=True),
        )
    except ImportError:
        print(
            "⚠️  docling.utils.model_downloader not available; "
            "docling models will download on first use."
        )
        print("   This is fine, but Full Privacy mode may show warnings until models are cached.")
    except (OSError, RuntimeError, ValueError) as e:
        print(f"⚠️  Failed to download docling models: {e}")
        print(
            "   Docling will download models on first use (may show warnings in Full Privacy mode)."
        )
    else:
        ensure_docling_rapidocr_cache_compat(cache_dir)
        print("Docling models cached.")


def _download_llm(app_data: Path, llm: dict) -> None:
    """ download llm."""
    from informity.config import DirNames
    from informity.llm.model_bootstrap import GGUFModelSpec, download_gguf_model

    repo_id = str(llm.get("repo_id") or "").strip()
    revision = llm.get("revision") or None
    filename = str(llm.get("filename") or "").strip()
    local_fname = str(llm.get("local_filename") or filename).strip()
    expected_sha256 = str(llm.get("sha256") or "").strip().lower() or None
    if not repo_id or not filename:
        raise SystemExit("llm must have repo_id and filename in install config.")

    models_dir = app_data / DirNames.MODELS / DirNames.LLM
    target_path = models_dir / local_fname
    if target_path.exists():
        print(f"LLM already present: {target_path}")
        return

    print(f"Downloading LLM: {repo_id} / {filename}")
    try:
        download_gguf_model(
            spec=GGUFModelSpec(
                repo_id=repo_id,
                filename=filename,
                expected_sha256=expected_sha256,
                revision=revision,
                model_label="llm",
            ),
            target_path=target_path,
        )
    except Exception as exc:
        raise SystemExit(f"Failed to download LLM: {exc}") from exc
    print(f"LLM saved as {target_path.name}")


def _verify_models_cached(install_config: dict) -> bool:
    """Verify that all required models are cached before enabling Full Privacy."""
    from informity.config import DirNames
    from informity.llm.model_bootstrap import CLASSIFIER_GGUF_SPEC

    cache_dir = _get_cache_dir()
    hf_hub_cache = cache_dir / DirNames.HUGGINGFACE / DirNames.HUB

    embedding_model = install_config.get("embedding_model") or "nomic-ai/nomic-embed-text-v1.5"
    if not _is_hf_model_cached(embedding_model, hf_hub_cache):
        return False

    reranker_model = install_config.get("reranker_model") or DEFAULT_RERANKER_MODEL
    if not _is_hf_model_cached(reranker_model, hf_hub_cache):
        return False

    from informity.config import _is_docling_cached

    if not _is_docling_cached(cache_dir):
        return False

    if install_config.get("llm") and isinstance(install_config["llm"], dict):
        app_data = _app_data_dir()
        models_dir = app_data / DirNames.MODELS / DirNames.LLM
        local_fname = install_config["llm"].get("local_filename") or install_config["llm"].get(
            "filename"
        )
        if local_fname:
            model_path = models_dir / local_fname
            if not model_path.exists() or not model_path.is_file():
                return False

    classifier_dir = _app_data_dir() / DirNames.MODELS / DirNames.CLASSIFIER
    classifier_path = classifier_dir / CLASSIFIER_GGUF_SPEC.filename
    return classifier_path.is_file()


def _write_offline_config(app_data: Path, install_config: dict) -> None:
    """ write offline config."""
    config_path = app_data / "config.json"
    existing: dict = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {}
        except (json.JSONDecodeError, OSError):
            pass

    if not _verify_models_cached(install_config):
        print(
            "⚠️  Warning: Not all models are cached. "
            "Full Privacy will be enabled after models are downloaded."
        )
        print(
            "   The app will allow model downloads on first run, then enable "
            "Full Privacy automatically."
        )
        existing["full_privacy"] = False
        existing["embedding_offline"] = False
        existing["llm_local_only"] = False
    else:
        existing["full_privacy"] = True
        existing["embedding_offline"] = True
        existing["llm_local_only"] = True
        print("✓ All models cached. Full Privacy enabled.")

    if install_config.get("embedding_model"):
        existing["embedding_model"] = install_config["embedding_model"]
    if install_config.get("reranker_model"):
        existing["rag_reranker_model"] = install_config["reranker_model"]
    if install_config.get("llm") and isinstance(install_config["llm"], dict):
        local_fname = install_config["llm"].get("local_filename") or install_config["llm"].get(
            "filename"
        )
        if local_fname:
            existing["llm_model_filename"] = local_fname

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(existing, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    privacy_status = "enabled" if existing.get("full_privacy") else "deferred"
    print(f"Config written: {config_path} (full_privacy={privacy_status})")


def _download_classifier(_app_data: Path) -> None:
    """ download classifier."""
    from informity.config import DirNames
    from informity.llm.model_bootstrap import CLASSIFIER_GGUF_SPEC, download_gguf_model

    models_dir = _app_data / DirNames.MODELS / DirNames.CLASSIFIER
    target_path = models_dir / CLASSIFIER_GGUF_SPEC.filename
    if target_path.exists():
        print(f"Classifier already present: {target_path}")
        return

    print(
        f"Downloading classifier: {CLASSIFIER_GGUF_SPEC.repo_id} / {CLASSIFIER_GGUF_SPEC.filename}"
    )
    try:
        download_gguf_model(
            spec=CLASSIFIER_GGUF_SPEC,
            target_path=target_path,
        )
    except Exception as exc:
        raise SystemExit(f"Failed to download classifier: {exc}") from exc
    print(f"Classifier saved as {target_path.name}")


def main() -> int:
    """Download and verify install-time models, then write offline config."""
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "install.conf.json"
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1]).resolve()

    app_data = _app_data_dir()
    print(f"App data dir: {app_data}")

    install_config = _load_install_config(config_path)
    _ensure_dirs(app_data)

    embedding_model = install_config.get("embedding_model") or "nomic-ai/nomic-embed-text-v1.5"
    _download_embedding_model(app_data, embedding_model)

    reranker_model = install_config.get("reranker_model") or DEFAULT_RERANKER_MODEL
    _download_reranker_model(app_data, reranker_model)

    _download_docling_models(app_data)

    if install_config.get("llm") and isinstance(install_config["llm"], dict):
        _download_llm(app_data, install_config["llm"])
    else:
        from informity.config import DirNames

        print(
            "No LLM in install config; skip. "
            f"Place a .gguf in {app_data}/{DirNames.MODELS}/{DirNames.LLM}/ "
            "if needed."
        )

    _download_classifier(app_data)

    _write_offline_config(app_data, install_config)
    print("Bootstrap done. Run the app; it will use cached models only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
