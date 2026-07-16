"""Test module for tests test chat request schema."""

from pydantic import ValidationError

from informity.api.schemas import ChatRequest


def test_chat_request_forbids_unknown_fields() -> None:
    """Test chat request forbids unknown fields."""
    try:
        ChatRequest.model_validate(
            {
                "message": "hello",
                "file_id": 123,
            }
        )
        raise AssertionError("Expected ValidationError for unknown field file_id")
    except ValidationError as exc:
        assert "Extra inputs are not permitted" in str(exc)
