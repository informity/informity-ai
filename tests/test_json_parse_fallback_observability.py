"""Test module for tests test json parse fallback observability."""

from unittest.mock import patch

from informity.db.utils import parse_json_sources, parse_json_tags


def test_parse_json_tags_logs_warning_on_invalid_json() -> None:
    """Test parse json tags logs warning on invalid json."""
    with patch("informity.db.utils.log.warning") as warning_mock:
        value = parse_json_tags("{invalid")
    assert value == []
    warning_mock.assert_called_once()


def test_parse_json_sources_logs_warning_when_shape_is_not_list() -> None:
    """Test parse json sources logs warning when shape is not list."""
    with patch("informity.db.utils.log.warning") as warning_mock:
        value = parse_json_sources('{"filename":"report.pdf"}')
    assert value == []
    warning_mock.assert_called_once()
