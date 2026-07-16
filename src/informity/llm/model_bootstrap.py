"""Shared helpers for downloading and validating GGUF model files."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import structlog

from informity.config import configure_hf_environment
from informity.exceptions import LLMError

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class GGUFModelSpec:
    """Specification for a downloadable GGUF model artifact."""

    repo_id: str
    filename: str
    expected_sha256: str | None = None
    revision: str | None = None
    model_label: str = "model"


CLASSIFIER_GGUF_SPEC = GGUFModelSpec(
    repo_id="LocalScribe/Qwen3.5-4B-Q4_K_M.gguf",
    filename="Qwen3.5-4B-Q4_K_M.gguf",
    expected_sha256=("00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"),
    model_label="classifier",
)


def _compute_sha256(path: Path) -> str:
    """compute sha256."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_parent_dir(path: Path) -> None:
    """ensure parent dir."""
    path.parent.mkdir(parents=True, exist_ok=True)


def download_gguf_model(
    *,
    spec: GGUFModelSpec,
    target_path: Path,
    progress_callback: Callable[[int, int | None, float], None] | None = None,
    cancel_event: Event | None = None,
) -> Path:
    """
    Download a GGUF model from Hugging Face Hub into `target_path`.

    The download is streamed to a temporary file, supports resume/cancel,
    verifies SHA-256 when provided, and skips if the target already exists
    with a matching hash.
    """
    repo_id = spec.repo_id
    filename = spec.filename
    expected_sha256 = spec.expected_sha256
    model_label = spec.model_label
    revision = spec.revision
    target_path = Path(target_path)

    if target_path.exists():
        if expected_sha256:
            actual_sha256 = _compute_sha256(target_path)
            if actual_sha256.lower() != expected_sha256.strip().lower():
                with suppress(OSError):
                    target_path.unlink(missing_ok=True)
                raise LLMError(
                    f"{model_label} integrity verification failed for {filename}: "
                    f"expected {expected_sha256}, got {actual_sha256}."
                )
        log.info(
            "gguf_model_already_present",
            model_label=model_label,
            repo=repo_id,
            filename=filename,
            target=str(target_path),
        )
        return target_path

    log.info(
        "downloading_gguf_model",
        model_label=model_label,
        repo=repo_id,
        filename=filename,
        revision=revision,
        target=str(target_path),
    )
    start = time.perf_counter()

    try:
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers, get_session

        configure_hf_environment(fail_on_missing_full_privacy_models=False)
        _ensure_parent_dir(target_path)

        tmp_path = target_path.parent / f"{target_path.name}.incomplete"
        bytes_done = int(tmp_path.stat().st_size) if tmp_path.exists() else 0

        url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
        headers = build_hf_headers()
        if bytes_done > 0:
            headers["Range"] = f"bytes={bytes_done}-"

        def _extract_total_bytes(response_obj: object, completed: int) -> int | None:
            """extract total bytes."""
            headers_obj = getattr(response_obj, "headers", None)
            if headers_obj is None:
                return None
            content_range = headers_obj.get("Content-Range", "")
            content_length = headers_obj.get("Content-Length")
            if content_range and "/" in content_range:
                with suppress(ValueError):
                    return int(content_range.split("/")[-1])
                return None
            if content_length:
                with suppress(ValueError):
                    return completed + int(content_length)
            return None

        def _status_code(response_obj: object) -> int | None:
            """status code."""
            status = getattr(response_obj, "status_code", None)
            return status if isinstance(status, int) else None

        def _iter_chunks(response_obj: object, size: int):
            """iter chunks."""
            iter_bytes = getattr(response_obj, "iter_bytes", None)
            if callable(iter_bytes):
                yield from iter_bytes(chunk_size=size)
                return
            iter_content = getattr(response_obj, "iter_content", None)
            if callable(iter_content):
                yield from iter_content(chunk_size=size)
                return
            raise LLMError("Unsupported HTTP response stream interface")

        session = get_session()
        last_report = time.perf_counter()
        report_interval_s = 0.20
        chunk_size = 1024 * 1024

        def _consume_response(response_obj: object) -> int | None:
            """consume response."""
            nonlocal bytes_done, last_report
            response_obj.raise_for_status()

            if bytes_done > 0 and _status_code(response_obj) == 200:
                with suppress(OSError):
                    tmp_path.unlink(missing_ok=True)
                bytes_done = 0

            total_bytes_local = _extract_total_bytes(response_obj, bytes_done)
            with tmp_path.open("ab" if bytes_done > 0 else "wb") as f:
                for chunk in _iter_chunks(response_obj, chunk_size):
                    if cancel_event is not None and cancel_event.is_set():
                        raise RuntimeError("download cancelled")
                    if not chunk:
                        continue
                    f.write(chunk)
                    bytes_done += len(chunk)

                    now = time.perf_counter()
                    if progress_callback and (now - last_report >= report_interval_s):
                        elapsed = max(now - start, 0.001)
                        speed_bps = float(bytes_done / elapsed)
                        progress_callback(bytes_done, total_bytes_local, speed_bps)
                        last_report = now
            return total_bytes_local

        total_bytes: int | None = None
        session_stream = getattr(session, "stream", None)
        if callable(session_stream):
            with session.stream("GET", url, headers=headers, timeout=(10, 60)) as response:
                total_bytes = _consume_response(response)
        else:
            response = session.get(url, headers=headers, stream=True, timeout=(10, 60))
            total_bytes = _consume_response(response)

        if progress_callback:
            elapsed = max(time.perf_counter() - start, 0.001)
            speed_bps = float(bytes_done / elapsed)
            progress_callback(bytes_done, total_bytes or bytes_done, speed_bps)

        tmp_path.replace(target_path)
        if expected_sha256:
            actual_sha256 = _compute_sha256(target_path)
            if actual_sha256.lower() != expected_sha256.strip().lower():
                with suppress(OSError):
                    target_path.unlink(missing_ok=True)
                raise LLMError(
                    f"{model_label} integrity verification failed for {filename}: "
                    f"expected {expected_sha256}, got {actual_sha256}."
                )

    except ImportError as exc:
        raise LLMError(f"huggingface-hub is not installed: {exc}") from exc
    except OSError as exc:
        raise LLMError(
            f"Failed to download {model_label} model from {repo_id}/{filename}: {exc}"
        ) from exc
    except Exception as exc:
        raise LLMError(
            f"Failed to download {model_label} model from {repo_id}/{filename}: {exc}"
        ) from exc

    elapsed_s = time.perf_counter() - start
    size_mb = target_path.stat().st_size / (1024 * 1024) if target_path.exists() else 0
    log.info(
        "gguf_model_downloaded",
        model_label=model_label,
        repo=repo_id,
        filename=filename,
        size_mb=round(size_mb, 1),
        elapsed_s=round(elapsed_s, 1),
    )
    return target_path
