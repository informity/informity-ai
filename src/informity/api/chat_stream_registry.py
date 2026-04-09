# ==============================================================================
# Informity AI — Active Chat Stream Registry
# Owns lifecycle and stop-signal state for in-flight chat streams.
# ==============================================================================

import asyncio
from dataclasses import dataclass


@dataclass
class ActiveChatStream:
    chat_id: str
    request_id: str
    stop_event: asyncio.Event
    task: asyncio.Task[object] | None = None
    stopped_by_user: bool = False


@dataclass
class StopOutcome:
    status: str
    stream_id: str | None = None
    chat_id: str | None = None
    request_id: str | None = None


class ChatStreamRegistry:
    def __init__(self) -> None:
        self._streams: dict[str, ActiveChatStream] = {}
        self._streams_by_request_id: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        stream_id: str,
        chat_id: str,
        request_id: str,
        stop_event: asyncio.Event,
        task: asyncio.Task[object] | None = None,
    ) -> None:
        async with self._lock:
            self._streams[stream_id] = ActiveChatStream(
                chat_id=chat_id,
                request_id=request_id,
                stop_event=stop_event,
                task=task,
            )
            self._streams_by_request_id[request_id] = stream_id

    async def unregister(self, stream_id: str) -> bool:
        async with self._lock:
            stream = self._streams.pop(stream_id, None)
            if stream is not None:
                self._streams_by_request_id.pop(stream.request_id, None)
                return True
            return False

    async def mark_stopped_by_user(
        self,
        *,
        stream_id: str | None,
        request_id: str | None,
        chat_id: str | None,
    ) -> StopOutcome:
        async with self._lock:
            stream: ActiveChatStream | None = None
            resolved_stream_id: str | None = None
            if stream_id:
                stream = self._streams.get(stream_id)
                resolved_stream_id = stream_id if stream is not None else None
            elif request_id:
                mapped_stream_id = self._streams_by_request_id.get(request_id)
                stream = self._streams.get(mapped_stream_id) if mapped_stream_id else None
                resolved_stream_id = mapped_stream_id if stream is not None else None
            if stream is None:
                return StopOutcome(status='not_found')
            if chat_id and stream.chat_id != chat_id:
                return StopOutcome(status='not_found')
            if stream.stopped_by_user:
                return StopOutcome(
                    status='already_terminal',
                    stream_id=resolved_stream_id,
                    chat_id=stream.chat_id,
                    request_id=stream.request_id,
                )
            stream.stopped_by_user = True
            stream.stop_event.set()
            if stream.task is not None and not stream.task.done():
                stream.task.cancel()
            return StopOutcome(
                status='stopped_now',
                stream_id=resolved_stream_id,
                chat_id=stream.chat_id,
                request_id=stream.request_id,
            )

    def is_stopped_by_user(self, stream_id: str) -> bool:
        stream = self._streams.get(stream_id)
        return bool(stream and stream.stopped_by_user)

    async def has_stream(self, stream_id: str) -> bool:
        async with self._lock:
            return stream_id in self._streams

    async def stop_all(self) -> int:
        # Stop all active streams (used by global operations like reset-all).
        async with self._lock:
            active_streams = list(self._streams.items())
            for _, stream in active_streams:
                if stream.stopped_by_user:
                    continue
                stream.stopped_by_user = True
                stream.stop_event.set()
                if stream.task is not None and not stream.task.done():
                    stream.task.cancel()
            return len(active_streams)


CHAT_STREAM_REGISTRY = ChatStreamRegistry()
