"""Test module for tests test boundary rules."""

from __future__ import annotations

from informity.scanner.extractors.boundary_rules import join_structured_blocks


def test_join_structured_blocks_preserves_block_boundaries() -> None:
    """Test join structured blocks preserves block boundaries."""
    text = join_structured_blocks(
        ["First paragraph.  ", "", "Second paragraph.", "Third paragraph."]
    )

    assert text == "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
