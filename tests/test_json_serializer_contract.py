from datetime import UTC, datetime

import pytest

from informity.utils.json_utils import serialize_api_response, serialize_config, serialize_trace


def test_serialize_config_coerces_non_json_types_to_string() -> None:
    payload = {'at': datetime(2026, 3, 12, 12, 0, tzinfo=UTC)}
    text = serialize_config(payload)
    assert '"at": "2026-03-12 12:00:00+00:00"' in text


def test_serialize_trace_coerces_non_json_types_to_string() -> None:
    payload = {'at': datetime(2026, 3, 12, 12, 0, tzinfo=UTC)}
    text = serialize_trace(payload)
    assert '"at": "2026-03-12 12:00:00+00:00"' in text


def test_serialize_api_response_rejects_non_json_types() -> None:
    payload = {'at': datetime(2026, 3, 12, 12, 0, tzinfo=UTC)}
    with pytest.raises(TypeError, match='requires JSON-serializable payload values'):
        serialize_api_response(payload)
