"""Module for llm classifier service."""

from __future__ import annotations

from functools import lru_cache

import structlog

from informity.llm.five_q_classifier import FiveQClassifier

log = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def get_classifier() -> FiveQClassifier:
    """
    Process-wide classifier singleton.

    Created once and reused for the lifetime of the process. The cached
    instance keeps its own loaded engine so the classifier stays warm across
    sequential chat requests.
    """
    classifier = FiveQClassifier()
    log.info(
        "classifier_service_created",
        classifier_id=id(classifier),
        classifier_model=getattr(classifier, "_model_filename", None),
    )
    return classifier


__all__ = ["get_classifier"]
