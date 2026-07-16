"""Module for api routes debug."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from informity.config import settings
from informity.llm.five_q_classifier import ClassifierContext, FiveQClassifier

router = APIRouter(tags=["debug"])


class DebugClassifyRequest(BaseModel):
    """Class docstring."""
    message: str = Field(min_length=1)
    model_filename: str | None = None


class DebugClassifyResponse(BaseModel):
    """Class docstring."""
    model_name: str
    raw_output: str
    decision: dict[str, object]


def _resolve_classifier_model_path(model_filename: str | None) -> Path | None:
    """Internal helper for resolve classifier model path."""
    if not model_filename:
        return None
    filename = str(model_filename or "").strip()
    if not filename.endswith(".gguf"):
        raise HTTPException(status_code=400, detail="model_filename must be a .gguf file")
    for models_dir in (settings.classifier_models_dir, settings.models_dir):
        if models_dir is None:
            continue
        candidate = Path(models_dir) / filename
        if candidate.exists():
            return candidate
    raise HTTPException(status_code=404, detail=f"Model not found: {filename}")


@lru_cache(maxsize=4)
def _get_classifier(model_path_str: str | None) -> FiveQClassifier:
    """Internal helper for get classifier."""
    model_path = Path(model_path_str) if model_path_str else None
    return FiveQClassifier(model_path=model_path)


@router.post("/api/debug/classify", response_model=DebugClassifyResponse)
async def debug_classify(request: DebugClassifyRequest) -> DebugClassifyResponse:
    """Debug classify."""
    model_path = _resolve_classifier_model_path(request.model_filename)
    classifier = _get_classifier(str(model_path) if model_path is not None else None)
    result = classifier.classify(request.message, ClassifierContext())
    return DebugClassifyResponse(
        model_name=result.model_name,
        raw_output=result.raw_output,
        decision=result.decision.__dict__,
    )
