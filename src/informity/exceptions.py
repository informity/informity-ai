# ==============================================================================
# Informity AI — Custom Exceptions
# All application-specific exceptions. Each includes a detail message and
# an optional source_path for file-related errors.
# ==============================================================================

from pathlib import Path

# ==============================================================================
# Base Exception
# ==============================================================================

class InformityError(Exception):
    # Base exception for all Informity errors.
    # All custom exceptions inherit from this so callers can catch broadly
    # when needed, or narrowly by specific subclass.

    def __init__(self, detail: str, source_path: Path | None = None) -> None:
        self.detail      = detail
        self.source_path = source_path
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if self.source_path is not None:
            # User-facing exception text should not leak absolute filesystem paths.
            return f'{self.detail} [source: {self.source_path.name}]'
        return self.detail


# ==============================================================================
# Indexing Errors
# ==============================================================================

class IndexingError(InformityError):
    # Raised when the indexing pipeline fails (chunking, embedding, or storage).
    # Examples: embedding model failure, vector storage write error.
    pass


# ==============================================================================
# LLM Errors
# ==============================================================================

class LLMError(InformityError):
    # Raised when the LLM engine encounters an error.
    # Examples: model file missing, inference failure, out of memory.
    pass


# ==============================================================================
# Configuration Errors
# ==============================================================================

class ConfigurationError(InformityError):
    # Raised when configuration is invalid or incompatible.
    # Examples: Full Privacy enabled but models not cached, invalid settings combination.
    pass

