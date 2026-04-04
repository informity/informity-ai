import pytest
from pydantic import ValidationError

from informity.api.schemas import ChatStopRequest


def test_chat_stop_request_accepts_stream_id_only() -> None:
    request = ChatStopRequest(stream_id='stream-1', chat_id='chat-1')
    assert request.stream_id == 'stream-1'
    assert request.request_id is None


def test_chat_stop_request_accepts_request_id_only() -> None:
    request = ChatStopRequest(request_id='request-1', chat_id='chat-1')
    assert request.request_id == 'request-1'
    assert request.stream_id is None


def test_chat_stop_request_requires_identifier() -> None:
    with pytest.raises(ValidationError):
        ChatStopRequest(chat_id='chat-1')
