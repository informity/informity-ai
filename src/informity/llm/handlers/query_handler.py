# ==============================================================================
# Informity AI — Query Handler Protocol
# Protocol that all query handlers must implement
# ==============================================================================

from collections.abc import AsyncGenerator
from typing import Protocol, runtime_checkable

import aiosqlite

from informity.api.schemas import ChatSourceReference
from informity.db.models import ChatMessage
from informity.llm.query_classifier import QueryClassification


@runtime_checkable
class QueryHandler(Protocol):
    """Protocol that all query handlers must implement."""

    def matches(self, classification: QueryClassification) -> bool:
        """
        Check if this handler should process the query.

        Args:
            classification: Query classification result

        Returns:
            True if this handler should process the query
        """
        ...

    async def handle(
        self,
        question:       str,
        classification: QueryClassification,
        history:        list[ChatMessage] | None,
        db:             aiosqlite.Connection,
        trace:          object | None,
        diagnostics_context: dict[str, object] | None = None,
        chat_id: str | None = None,
        file_ids: list[int] | None = None,
    ) -> AsyncGenerator[str | list[ChatSourceReference] | tuple[str, object]]:
        """
        Process the query and yield tokens/sources.

        Args:
            question: User's question
            classification: Query classification result
            history: Chat history (optional)
            db: Database connection
            trace: Trace writer (optional)

        Yields:
            str tokens (response text) followed by list[ChatSourceReference] (sources)
        """
        ...
