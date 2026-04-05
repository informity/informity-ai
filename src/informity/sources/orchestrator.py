from __future__ import annotations

import structlog

from informity.scanner.crawler import ScannedFile
from informity.sources.filesystem_adapter import FilesystemSourceAdapter
from informity.sources.registry import SourceAdapterRegistry

log = structlog.get_logger(__name__)


class SourceIngestionOrchestrator:
    def __init__(self, registry: SourceAdapterRegistry) -> None:
        self.registry = registry

    def discover_filesystem_scanned_files(
        self,
        *,
        directories: list,
        ignore_patterns: list[str],
        supported_extensions: list[str],
        follow_symlinks: bool,
    ) -> list[ScannedFile]:
        adapter = self.registry.get('filesystem')
        refs = adapter.discover(
            {
                'directories': directories,
                'ignore_patterns': ignore_patterns,
                'supported_extensions': supported_extensions,
                'follow_symlinks': follow_symlinks,
            }
        )

        scanned_files: list[ScannedFile] = []
        for ref in refs:
            try:
                item = adapter.fetch(ref)
            except (RuntimeError, ValueError, TypeError, OSError) as exc:
                log.warning(
                    'source_item_fetch_failed',
                    provider='filesystem',
                    source_item_id=ref.item_id,
                    locator=ref.locator,
                    error=str(exc),
                )
                continue

            scanned = item.metadata.get('scanned_file')
            if isinstance(scanned, ScannedFile):
                scanned_files.append(scanned)
                continue

            log.warning(
                'source_item_missing_scanned_file',
                provider='filesystem',
                source_item_id=item.source_item_id,
            )
        return scanned_files


def build_default_orchestrator() -> SourceIngestionOrchestrator:
    registry = SourceAdapterRegistry()
    registry.register(FilesystemSourceAdapter())
    return SourceIngestionOrchestrator(registry)
