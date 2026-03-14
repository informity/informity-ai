from unittest.mock import patch

from informity.db.utils import parse_json_sources, parse_json_tags
from informity.utils.json_utils import parse_json_safe


def test_parse_json_safe_logs_warning_on_invalid_json() -> None:
    with patch('informity.utils.json_utils.log.warning') as warning_mock:
        value = parse_json_safe('{invalid', default={})
    assert value == {}
    warning_mock.assert_called_once()


def test_parse_json_tags_logs_warning_on_invalid_json() -> None:
    with patch('informity.db.utils.log.warning') as warning_mock:
        value = parse_json_tags('{invalid')
    assert value == []
    warning_mock.assert_called_once()


def test_parse_json_sources_logs_warning_when_shape_is_not_list() -> None:
    with patch('informity.db.utils.log.warning') as warning_mock:
        value = parse_json_sources('{"filename":"report.pdf"}')
    assert value == []
    warning_mock.assert_called_once()
