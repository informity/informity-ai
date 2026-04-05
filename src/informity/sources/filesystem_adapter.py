from __future__ import annotations

from pathlib import Path
from typing import Any

from informity.scanner.crawler import scan_directories, scanned_file_for_path
from informity.sources.base import IngestionItem, SourceItemRef
from informity.utils.path_utils import normalize_path


class FilesystemSourceAdapter:
    provider = 'filesystem'

    def discover(self, scope: dict[str, Any]) -> list[SourceItemRef]:
        directories = scope.get('directories')
        ignore_patterns = scope.get('ignore_patterns')
        supported_extensions = scope.get('supported_extensions')
        follow_symlinks = scope.get('follow_symlinks')

        scanned = scan_directories(
            directories=directories,
            ignore_patterns=ignore_patterns,
            supported_extensions=supported_extensions,
            follow_symlinks=follow_symlinks,
        )

        refs: list[SourceItemRef] = []
        for item in scanned:
            normalized_path = str(normalize_path(item.path, expand_user=False))
            refs.append(
                SourceItemRef(
                    provider=self.provider,
                    item_id=normalized_path,
                    locator=normalized_path,
                    item_type='file',
                    modified_at=item.modified_at,
                )
            )
        return refs

    def fetch(self, ref: SourceItemRef) -> IngestionItem:
        path = Path(ref.locator)
        scanned = scanned_file_for_path(path)
        if scanned is None:
            raise FileNotFoundError(f'Could not read source item: {ref.locator}')

        normalized_path = str(normalize_path(scanned.path, expand_user=False))
        return IngestionItem(
            provider=self.provider,
            source_item_id=normalized_path,
            item_type='file',
            title=scanned.filename,
            author=None,
            modified_at=scanned.modified_at,
            content_text='',
            content_hash=scanned.content_hash,
            metadata={
                'path': normalized_path,
                'scanned_file': scanned,
            },
            attachments=[],
        )

    def dedupe_key(self, item: IngestionItem) -> str:
        return item.source_item_id

    def canonical_id(self, item: IngestionItem) -> str:
        return item.source_item_id
