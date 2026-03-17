# ==============================================================================
# Informity AI — Bootstrap models for install script
# Downloads embedding model, reranker (cross-encoder), optional LLM, and
# classifier LLM into app data, then writes config.json with embedding_offline
# and llm_local_only set to true so the app always uses cached models after install.
# Run from repo root: uv run python scripts/bootstrap_models.py
# Requires: INFORMITY_APP_DATA_DIR (default: ./data) and install.conf.json
# ==============================================================================

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# CRITICAL: Set HF cache paths BEFORE any imports that might initialize huggingface_hub
# This ensures models are downloaded to project cache, not ~/.cache/huggingface/hub/
def _setup_hf_cache_early() -> None:
    """Set HF_HOME and HF_HUB_CACHE env vars before any HF imports."""
    # Get repo root without importing informity.config (which might import HF libs)
    repo_root = Path(__file__).resolve().parent.parent
    raw_cache_dir = os.environ.get('INFORMITY_CACHE_DIR', '')
    if raw_cache_dir:
        cache_dir = Path(raw_cache_dir).resolve() if not Path(raw_cache_dir).is_absolute() else Path(raw_cache_dir)
    else:
        cache_dir = repo_root / '.cache'
    
    hf_home = cache_dir / 'huggingface'
    hf_hub = hf_home / 'hub'
    os.environ['HF_HOME'] = str(hf_home)
    os.environ['HF_HUB_CACHE'] = str(hf_hub)

# Set HF cache paths immediately
_setup_hf_cache_early()

# Import default reranker model from config
# Note: We import from config module which may have dependencies, but this constant
# is defined early and doesn't require any heavy imports
try:
    from informity.config import _DEFAULT_RERANKER_MODEL
except ImportError:
    # Fallback if import fails (shouldn't happen in normal usage)
    # sentence-transformers uses cross-encoder/ prefix
    _DEFAULT_RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'


# Resolve app data dir so install always targets the project's data/ unless overridden.
# When run via install.sh, INFORMITY_APP_DATA_DIR is set to repo_root/data.
# When run directly (e.g. uv run python scripts/bootstrap_models.py), use repo-relative
# data/ so install goes to the right place regardless of cwd.
def _app_data_dir() -> Path:
    raw = os.environ.get('INFORMITY_APP_DATA_DIR', '')
    if raw:
        p = Path(raw)
        return p.resolve() if not p.is_absolute() else p
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / 'data'


def _load_install_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise SystemExit(f'Install config not found: {config_path}')
    data = json.loads(config_path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise SystemExit('Install config must be a JSON object.')
    return data


def _get_repo_root() -> Path:
    """Find project root (directory containing pyproject.toml)."""
    current = Path(__file__).resolve().parent.parent
    while current != current.parent:
        if (current / 'pyproject.toml').exists():
            return current
        current = current.parent
    return Path.cwd()


def _ensure_dirs(app_data: Path) -> None:
    """Create directories for models, cache, and user data."""
    # Import here to avoid circular imports
    from informity.config import DirNames
    
    # Use _get_cache_dir() to respect INFORMITY_CACHE_DIR env var (not hardcoded to repo root)
    cache_root = _get_cache_dir()
    (app_data / DirNames.MODELS / DirNames.LLM).mkdir(parents=True, exist_ok=True)  # app_data/models/chat-llm/
    (app_data / DirNames.MODELS / DirNames.QUERY_CLASSIFIER_MODELS).mkdir(parents=True, exist_ok=True)  # app_data/models/query-classifier-llm/
    (_get_repo_root() / DirNames.TOOLS / DirNames.DIAGNOSTICS / DirNames.DIAGNOSTICS_MODELS).mkdir(parents=True, exist_ok=True)  # tools/diagnostics/models/
    (cache_root / DirNames.HUGGINGFACE / DirNames.HUB).mkdir(parents=True, exist_ok=True)  # .cache/huggingface/hub/
    (cache_root / DirNames.DOCLING).mkdir(parents=True, exist_ok=True)  # .cache/docling/
    # User data (in app_data, cleared by reset.sh)
    # Note: vectors_dir removed - vectors now stored in SQLite via sqlite-vec
    (app_data / DirNames.LOGS).mkdir(parents=True, exist_ok=True)


def _download_embedding_model(app_data: Path, model_id: str) -> None:
    """Download embedding model using sentence-transformers (PyTorch)."""
    hf_hub_cache = _get_cache_dir() / 'huggingface' / 'hub'
    if _is_hf_model_cached(model_id, hf_hub_cache):
        print(f'Embedding model already cached: {model_id}')
        return

    print(f'Downloading embedding model: {model_id}')
    try:
        from sentence_transformers import SentenceTransformer
        
        # HF cache paths already set by _setup_hf_cache_early() at module import
        # sentence-transformers downloads models on first use and caches them in HF cache
        # Initialize model to trigger download (trust_remote_code=True required for nomic model)
        model = SentenceTransformer(model_id, trust_remote_code=True)
        # Trigger full load so all assets are cached
        model.encode(['bootstrap'])
    except Exception as e:
        raise SystemExit(f'Failed to download embedding model: {e}') from e
    print('Embedding model cached.')


def _download_reranker_model(app_data: Path, model_id: str) -> None:
    """Download the cross-encoder reranker using sentence-transformers (PyTorch)."""
    hf_hub_cache = _get_cache_dir() / 'huggingface' / 'hub'
    if _is_hf_model_cached(model_id, hf_hub_cache):
        print(f'Reranker model already cached: {model_id}')
        return

    print(f'Downloading reranker (cross-encoder): {model_id}')
    try:
        from sentence_transformers import CrossEncoder
        
        # HF cache paths already set by _setup_hf_cache_early() at module import
        # sentence-transformers downloads models on first use and caches them in HF cache
        # Initialize model to trigger download
        model = CrossEncoder(model_id)
        # Trigger full load so all assets are cached
        model.predict([['bootstrap', 'dummy passage']])
    except Exception as e:
        raise SystemExit(f'Failed to download reranker model: {e}') from e
    print('Reranker model cached.')


def _get_cache_dir() -> Path:
    """Resolve cache root the same way the app does (so install and runtime use the same path)."""
    from informity.config import DirNames
    repo_root = _get_repo_root()
    raw = os.environ.get('INFORMITY_CACHE_DIR', '')
    if raw:
        p = Path(raw)
        cache_dir = p.resolve() if not p.is_absolute() else p
    else:
        cache_dir = repo_root / DirNames.CACHE
    return cache_dir


def _download_docling_models(app_data: Path) -> None:
    """Download docling models into the unified cache directory.

    Store under {cache_dir}/docling so docling finds them at runtime when
    DOCLING_ARTIFACTS_PATH is set to that path (same as app's docling extractor).
    """
    from informity.config import DirNames

    cache_dir = _get_cache_dir()
    docling_cache = cache_dir / DirNames.DOCLING  # .cache/docling/ (docling creates subdirs inside)

    # Ensure cache directory exists
    docling_cache.mkdir(parents=True, exist_ok=True)

    if _is_docling_cached(cache_dir):
        print(f'Docling models already cached: {docling_cache}')
        return

    # Set docling artifacts path so download_models() writes here and runtime finds them here
    os.environ['DOCLING_ARTIFACTS_PATH'] = str(docling_cache)
    
    # Allow network for this one-time download
    os.environ.pop('HF_HUB_OFFLINE', None)
    os.environ.pop('TRANSFORMERS_OFFLINE', None)
    
    print(f'Downloading docling models -> {docling_cache}')
    try:
        from docling.utils.model_downloader import download_models
        # Download core models needed for document conversion
        # with_layout=True: layout detection models
        # with_tableformer=True: table extraction models
        # with_code_formula=True: formula extraction
        # Other models are optional and can be downloaded later if needed
        # Note: download_models expects a Path object, not a string
        download_models(
            output_dir=docling_cache,  # Path object, not string
            force=False,  # Don't re-download if already present
            progress=True,
            with_layout=True,
            with_tableformer=True,
            with_code_formula=True,
            # Skip optional models for faster install (can be added later if needed)
            with_picture_classifier=False,
            with_smolvlm=False,
            with_granitedocling=False,
            with_granitedocling_mlx=False,
            with_smoldocling=False,
            with_smoldocling_mlx=False,
            with_granite_vision=False,
            with_granite_chart_extraction=False,
            # Enable RapidOCR for image-only PDF extraction (always download, respects Full Privacy after install)
            with_rapidocr=True,
            with_easyocr=False,  # RapidOCR is sufficient and lighter than EasyOCR
        )
    except ImportError:
        print('⚠️  docling.utils.model_downloader not available; docling models will download on first use.')
        print('   This is fine, but Full Privacy mode may show warnings until models are cached.')
    except Exception as e:
        print(f'⚠️  Failed to download docling models: {e}')
        print('   Docling will download models on first use (may show warnings in Full Privacy mode).')
    else:
        print('Docling models cached.')


def _download_llm(app_data: Path, llm: dict) -> None:
    # Use original Hugging Face names (e.g. Qwen3-14B-Q5_K_M.gguf); no lowercasing.
    repo_id       = llm.get('repo_id') or ''
    filename      = llm.get('filename') or ''
    local_fname   = llm.get('local_filename') or filename
    if not repo_id or not filename:
        raise SystemExit('llm must have repo_id and filename in install config.')

    from informity.config import DirNames
    
    models_dir = app_data / DirNames.MODELS / DirNames.LLM  # app_data/models/chat-llm/
    target_path = models_dir / local_fname
    if target_path.exists():
        print(f'LLM already present: {target_path}')
        return

    print(f'Downloading LLM: {repo_id} / {filename}')
    from huggingface_hub import hf_hub_download
    
    # HF cache paths already set by _setup_hf_cache_early() at module import
    # Get the cache dir to construct hf_hub path
    cache_dir = _get_cache_dir()
    hf_home = cache_dir / DirNames.HUGGINGFACE
    hf_hub  = hf_home / DirNames.HUB
    
    models_dir.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id   = repo_id,
        filename  = filename,
        local_dir = str(models_dir),
        cache_dir = str(hf_hub),  # Use unified HF cache, not models_dir/.cache
    )
    downloaded_path = Path(downloaded)
    if downloaded_path.name != local_fname and downloaded_path.exists():
        downloaded_path.rename(target_path)
    print(f'LLM saved as {target_path.name}')


def _download_classifier_llm(app_data: Path, classifier_llm: dict) -> None:
    """Download classifier LLM (small model for query classification)."""
    repo_id       = classifier_llm.get('repo_id') or ''
    filename      = classifier_llm.get('filename') or ''
    local_fname   = classifier_llm.get('local_filename') or filename
    if not repo_id or not filename:
        raise SystemExit('classifier_llm must have repo_id and filename in install config.')

    from informity.config import DirNames
    
    models_dir = app_data / DirNames.MODELS / DirNames.QUERY_CLASSIFIER_MODELS  # app_data/models/query-classifier-llm/
    target_path = models_dir / local_fname
    if target_path.exists():
        print(f'Classifier LLM already present: {target_path}')
        return

    print(f'Downloading classifier LLM: {repo_id} / {filename}')
    from huggingface_hub import hf_hub_download
    
    # HF cache paths already set by _setup_hf_cache_early() at module import
    cache_dir = _get_cache_dir()
    hf_home = cache_dir / DirNames.HUGGINGFACE
    hf_hub  = hf_home / DirNames.HUB
    
    models_dir.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id   = repo_id,
        filename  = filename,
        local_dir = str(models_dir),
        cache_dir = str(hf_hub),  # Use unified HF cache, not models_dir/.cache
    )
    downloaded_path = Path(downloaded)
    if downloaded_path.name != local_fname and downloaded_path.exists():
        downloaded_path.rename(target_path)
    print(f'Classifier LLM saved as {target_path.name}')


def _is_hf_model_cached(model_name: str, hf_hub_cache: Path) -> bool:
    """Check if a HuggingFace model is cached (standalone version for bootstrap)."""
    if not hf_hub_cache.exists():
        return False
    # HuggingFace stores models with structure: models--{org}--{model_name}/snapshots/{hash}/
    # e.g., "nomic-ai/nomic-embed-text-v1.5" -> "models--nomic-ai--nomic-embed-text-v1.5"
    model_dir_pattern = f'models--{model_name.replace("/", "--")}'
    model_dir = hf_hub_cache / model_dir_pattern
    if not model_dir.exists():
        return False
    try:
        # Check for HuggingFace model files in snapshots subdirectory
        snapshots_dir = model_dir / 'snapshots'
        if not snapshots_dir.exists():
            return False
        
        # Check each snapshot directory
        for snapshot_dir in snapshots_dir.iterdir():
            if not snapshot_dir.is_dir():
                continue
            # HuggingFace models have config.json and model weights (.bin, .safetensors, or .onnx)
            has_config = (snapshot_dir / 'config.json').exists()
            has_weights = (
                any(snapshot_dir.rglob('*.bin')) or
                any(snapshot_dir.rglob('*.safetensors')) or
                any(snapshot_dir.rglob('*.onnx'))
            )
            if has_config and has_weights:
                return True
        return False
    except Exception:
        return False


def _is_docling_cached(cache_dir: Path) -> bool:
    """Check if docling runtime artifacts are cached (standalone for bootstrap)."""
    from informity.config import DirNames
    docling_cache = cache_dir / DirNames.DOCLING

    # Native docling artifact cache
    try:
        if docling_cache.exists():
            for item in docling_cache.iterdir():
                if item.is_dir():
                    if any(item.rglob('*.bin')) or any(item.rglob('*.safetensors')) or any(item.rglob('*.onnx')):
                        return True
                elif item.suffix in ('.bin', '.safetensors', '.onnx', '.pt', '.pth'):
                    return True
    except Exception:
        pass
    return False


def _verify_models_cached(install_config: dict) -> bool:
    """Verify that all required models are cached before enabling Full Privacy."""
    from informity.config import DirNames
    from pathlib import Path
    
    cache_dir = _get_cache_dir()
    hf_hub_cache = cache_dir / DirNames.HUGGINGFACE / DirNames.HUB
    
    # Check embedding model (project cache only - no fallback to default)
    embedding_model = install_config.get('embedding_model') or 'nomic-ai/nomic-embed-text-v1.5'
    if not _is_hf_model_cached(embedding_model, hf_hub_cache):
        return False
    
    # Check reranker model (project cache only - no fallback to default)
    reranker_model = install_config.get('reranker_model') or _DEFAULT_RERANKER_MODEL
    if not _is_hf_model_cached(reranker_model, hf_hub_cache):
        return False
    
    # Check docling models
    if not _is_docling_cached(cache_dir):
        return False
    
    # Check LLM model (if configured)
    if install_config.get('llm') and isinstance(install_config['llm'], dict):
        from informity.config import DirNames
        app_data = _app_data_dir()
        models_dir = app_data / DirNames.MODELS / DirNames.LLM
        local_fname = install_config['llm'].get('local_filename') or install_config['llm'].get('filename')
        if local_fname:
            model_path = models_dir / local_fname
            if not model_path.exists() or not model_path.is_file():
                return False
    
    # Check classifier LLM model (if configured)
    if install_config.get('classifier_llm') and isinstance(install_config['classifier_llm'], dict):
        from informity.config import DirNames
        app_data = _app_data_dir()
        models_dir = app_data / DirNames.MODELS / DirNames.QUERY_CLASSIFIER_MODELS
        local_fname = install_config['classifier_llm'].get('local_filename') or install_config['classifier_llm'].get('filename')
        if local_fname:
            model_path = models_dir / local_fname
            if not model_path.exists() or not model_path.is_file():
                return False
    
    return True


def _write_offline_config(app_data: Path, install_config: dict) -> None:
    config_path = app_data / 'config.json'
    existing: dict = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding='utf-8'))
            if not isinstance(existing, dict):
                existing = {}
        except (json.JSONDecodeError, OSError):
            pass

    # Verify models are cached before enabling Full Privacy
    if not _verify_models_cached(install_config):
        print('⚠️  Warning: Not all models are cached. Full Privacy will be enabled after models are downloaded.')
        print('   The app will allow model downloads on first run, then enable Full Privacy automatically.')
        # Don't enable Full Privacy yet - let runtime handle it after models are cached
        existing['full_privacy']       = False
        existing['embedding_offline']  = False
        existing['llm_local_only']     = False
    else:
        # All models cached - safe to enable Full Privacy
        existing['full_privacy']       = True
        existing['embedding_offline']  = True
        existing['llm_local_only']     = True
        print('✓ All models cached. Full Privacy enabled.')
    
    if install_config.get('embedding_model'):
        existing['embedding_model'] = install_config['embedding_model']
    if install_config.get('reranker_model'):
        existing['rag_reranker_model'] = install_config['reranker_model']
    if install_config.get('llm') and isinstance(install_config['llm'], dict):
        # Use original HF filename when local_filename omitted (no lowercasing).
        local_fname = install_config['llm'].get('local_filename') or install_config['llm'].get('filename')
        if local_fname:
            existing['llm_model_filename'] = local_fname  # This should be Qwen3-14B-Q5_K_M.gguf
    
    if install_config.get('classifier_llm') and isinstance(install_config['classifier_llm'], dict):
        # Use original HF filename when local_filename omitted (no lowercasing).
        local_fname = install_config['classifier_llm'].get('local_filename') or install_config['classifier_llm'].get('filename')
        if local_fname:
            existing['classifier_llm_model'] = local_fname  # This should be Qwen2.5-3B-Instruct-Q4_K_M.gguf

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(existing, indent=2, default=str) + '\n',
        encoding='utf-8',
    )
    privacy_status = 'enabled' if existing.get('full_privacy') else 'deferred'
    print(f'Config written: {config_path} (full_privacy={privacy_status})')


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / 'install.conf.json'
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1]).resolve()

    app_data = _app_data_dir()
    print(f'App data dir: {app_data}')

    install_config = _load_install_config(config_path)
    _ensure_dirs(app_data)

    embedding_model = install_config.get('embedding_model') or 'nomic-ai/nomic-embed-text-v1.5'
    _download_embedding_model(app_data, embedding_model)

    reranker_model = install_config.get('reranker_model') or _DEFAULT_RERANKER_MODEL
    _download_reranker_model(app_data, reranker_model)

    # Download docling models for document extraction (PDF, DOCX, etc.)
    _download_docling_models(app_data)

    if install_config.get('llm') and isinstance(install_config['llm'], dict):
        _download_llm(app_data, install_config['llm'])
    else:
        from informity.config import DirNames
        print(
            'No LLM in install config; skip. '
            f'Place a .gguf in {app_data}/{DirNames.MODELS}/{DirNames.LLM}/ if needed.'
        )

    # Download classifier LLM (small model for query classification)
    if install_config.get('classifier_llm') and isinstance(install_config['classifier_llm'], dict):
        _download_classifier_llm(app_data, install_config['classifier_llm'])
    else:
        from informity.config import DirNames, _DEFAULT_CLASSIFIER_LLM_MODEL_FILENAME
        print(
            'No classifier_llm in install config; skip. '
            f'Default classifier model ({_DEFAULT_CLASSIFIER_LLM_MODEL_FILENAME}) '
            f'will be expected in {app_data}/{DirNames.MODELS}/{DirNames.QUERY_CLASSIFIER_MODELS}/.'
        )

    _write_offline_config(app_data, install_config)
    print('Bootstrap done. Run the app; it will use cached models only.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
