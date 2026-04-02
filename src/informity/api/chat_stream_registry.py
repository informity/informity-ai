# ==============================================================================
# Informity AI — Active Chat Stream Registry
# Owns lifecycle and stop-signal state for in-flight chat streams.
# ==============================================================================

import asyncio
from dataclasses import dataclass


@dataclass
class ActiveChatStream:
    chat_id: str
    stop_event: asyncio.Event
    stopped_by_user: bool = False


class ChatStreamRegistry:
    def __init__(self) -> None:
        self._streams: dict[str, ActiveChatStream] = {}
        self._lock = asyncio.Lock()

    async def register(self, stream_id: str, chat_id: str, stop_event: asyncio.Event) -> None:
        async with self._lock:
            self._streams[stream_id] = ActiveChatStream(
                chat_id=chat_id,
                stop_event=stop_event,
            )

    async def unregister(self, stream_id: str) -> None:
        async with self._lock:
            self._streams.pop(stream_id, None)

    async def mark_stopped_by_user(self, stream_id: str, chat_id: str | None) -> bool:
        async with self._lock:
            stream = self._streams.get(stream_id)
            if stream is None:
                return False
            if chat_id and stream.chat_id != chat_id:
                return False
            stream.stopped_by_user = True
            stream.stop_event.set()
            return True

    def is_stopped_by_user(self, stream_id: str) -> bool:
        stream = self._streams.get(stream_id)
        return bool(stream and stream.stopped_by_user)


CHAT_STREAM_REGISTRY = ChatStreamRegistry()
