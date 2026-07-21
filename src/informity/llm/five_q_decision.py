"""Module for llm five q decision."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from informity.llm.types import IntentProfileId, QuerySubtype, QueryType

FiveQSource = Literal["index_metadata", "document_content", "chat_history", "app_knowledge"]
FiveQScope = Literal["targeted", "broad", "none"]
FiveQOperation = Literal["lookup", "count_enumerate", "summarize_synthesize", "compare"]


@dataclass(frozen=True)
class FiveQDecision:
    """Class docstring."""
    source: FiveQSource
    scope: FiveQScope = "none"
    operation: FiveQOperation = "lookup"
    partitions: list[str] = field(default_factory=list)
    subqueries: list[str] = field(default_factory=list)
    exhaustive: bool = False
    confidence: float = 0.0

    def derive_intent(self) -> QueryType:
        """Derive intent."""
        if self.source == "index_metadata":
            return QueryType.METADATA
        if self.source in {"chat_history", "app_knowledge"}:
            return QueryType.SIMPLE
        if self.scope == "broad" or self.exhaustive:
            return QueryType.COVERAGE
        return QueryType.FOCUSED

    def derive_route_candidate(self) -> IntentProfileId:
        """Derive route candidate."""
        if self.source == "index_metadata":
            return IntentProfileId.METADATA_INVENTORY
        if self.source == "chat_history":
            return IntentProfileId.CONTINUATION_OR_REFINEMENT
        if self.source == "app_knowledge":
            return IntentProfileId.CLARIFICATION_OR_DISAMBIGUATION
        if self.scope == "broad" and self.operation == "compare":
            return IntentProfileId.COMPARATIVE_ANALYSIS
        if self.exhaustive:
            return IntentProfileId.CROSS_DOCUMENT_SYNTHESIS
        if self.scope == "broad":
            return IntentProfileId.CROSS_DOCUMENT_SYNTHESIS
        return IntentProfileId.TARGETED_FACT_LOOKUP

    def derive_subtype(self) -> QuerySubtype | None:
        """Derive subtype."""
        if (
            self.source == "document_content"
            and self.partitions
            and any(len(partition) == 4 and partition.isdigit() for partition in self.partitions)
        ):
            return QuerySubtype.AGGREGATE_BY_PERIOD
        if self.operation == "compare":
            return QuerySubtype.COMPARATIVE
        return None


__all__ = ["FiveQDecision", "FiveQOperation", "FiveQScope", "FiveQSource"]
