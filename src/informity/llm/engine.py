# ==============================================================================
# Informity AI — LLM Engine
# Lazy-loads a GGUF model via xllamacpp with Metal acceleration.
# Provides async streaming generation. Downloads the model from Hugging Face
# Hub if not present locally.
#
# Runtime: xllamacpp (CommonParams + Server, no-server in-process path).
# Token counting: tiktoken cl100k_base approximation (±15%); 100-token safety
# margin in _truncate_messages_to_fit absorbs the variance.
# Chat template: read from GGUF metadata via gguf.GGUFReader at load time.
# EOS suppression (min_tokens): not available in xllamacpp — pipeline-level
# word-count gate (generation_runtime.py) is the enforcement mechanism.
# ==============================================================================

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import threading
import time
from collections.abc import AsyncGenerator
from contextlib import redirect_stderr, suppress
from pathlib import Path

import structlog

from informity.config import settings
from informity.exceptions import LLMError
from informity.llm.model_adapter import get_profile, get_profile_for_filename
from informity.utils.directory_utils import ensure_file_directory

log = structlog.get_logger(__name__)
_PROMPT_RENDER_EXCEPTIONS = (ValueError, TypeError, AttributeError, RuntimeError)

# ==============================================================================
# Constants
# ==============================================================================

DEFAULT_HF_FILENAME = 'Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf'

# Sentinel put on the queue when the stream worker is done
_STREAM_END: object = object()


# ==============================================================================
# Token counting (tiktoken approximation)
# ==============================================================================

def _count_tokens(text: str) -> int:
    # Approximate token count using tiktoken cl100k_base.
    # Accuracy vs. Qwen3 tokenizer: ±15%. The 100-token safety margin in
    # _truncate_messages_to_fit absorbs this variance for budget management.
    if not text:
        return 0
    import tiktoken
    enc = tiktoken.get_encoding('cl100k_base')
    return len(enc.encode(text))


# ==============================================================================
# Finish reason normalisation
# ==============================================================================

def _normalize_finish_reason(reason: str | None) -> str | None:
    if not reason:
        return None
    reason_lower = str(reason).lower().strip()
    if reason_lower in ('stop', 'eos', 'end_of_sequence'):
        return 'stop'
    if reason_lower in ('length', 'max_tokens', 'max_tokens_reached'):
        return 'length'
    if reason_lower == 'cancelled':
        return 'cancelled'
    log.debug('llm_unknown_finish_reason', reason=reason)
    return reason_lower


# ==============================================================================
# HuggingFace cache cleanup
# ==============================================================================

def remove_models_dir_cache() -> None:
    # Remove any nested .cache directories left by huggingface_hub after download.
    for models_dir in [settings.models_dir, settings.diagnostics_models_dir]:
        if models_dir is None:
            continue
        cache_dir = models_dir / '.cache'
        if cache_dir.is_dir():
            try:
                shutil.rmtree(cache_dir)
                log.info('models_dir_cache_removed', path=str(cache_dir))
            except OSError as exc:
                log.warning('models_dir_cache_remove_failed', path=str(cache_dir), error=str(exc))


# ==============================================================================
# GGUF metadata
# ==============================================================================

def _read_gguf_chat_template(model_path: Path) -> str:
    # Read the chat template from GGUF metadata using gguf.GGUFReader.
    # Returns empty string if not found or on any error.
    try:
        from gguf import GGUFReader  # type: ignore[import-untyped]
        reader = GGUFReader(str(model_path), mode='r')
        field = reader.fields.get('tokenizer.chat_template')
        if field is not None and field.parts:
            return bytes(field.parts[-1]).decode('utf-8')
    except Exception as exc:
        log.debug('gguf_template_read_failed', error=str(exc))
    return ''


# ==============================================================================
# Prompt rendering
# ==============================================================================

def _messages_to_prompt(chat_template: str, messages: list[dict[str, str]]) -> str:
    # Convert chat messages to a prompt string.
    # Uses the GGUF's embedded Jinja2 chat template when available (preferred),
    # falls back to ChatML when the template is empty or fails to render.
    if chat_template:
        try:
            prompt = _render_gguf_template(chat_template, messages)
            if prompt and prompt.strip():
                # Strip trailing <think> prefill that some GGUF builds add.
                # The model will generate <think> itself; having it in the prompt
                # breaks our streaming <think> block detection.
                prompt = re.sub(r'\s*<think>\s*$', '', prompt)
                log.debug(
                    'gguf_template_used',
                    prompt_len  = len(prompt),
                    prompt_tail = repr(prompt[-80:]),
                )
                return prompt
            log.warning('gguf_template_empty', msg='GGUF template rendered to empty string; using ChatML fallback')
        except _PROMPT_RENDER_EXCEPTIONS as exc:
            log.warning('gguf_template_render_failed', error=str(exc))
    else:
        log.debug('gguf_template_not_found', msg='No chat template in GGUF metadata')

    prompt = _fallback_chatml_prompt(messages)
    log.debug('chatml_fallback_used', prompt_len=len(prompt), prompt_tail=repr(prompt[-80:]))
    return prompt


def _render_gguf_template(template_str: str, messages: list[dict[str, str]]) -> str:
    from jinja2 import BaseLoader, Environment
    env = Environment(loader=BaseLoader())
    env.globals['raise_exception'] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
    template = env.from_string(template_str)
    return template.render(messages=messages, add_generation_prompt=True)


def _fallback_chatml_prompt(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for msg in messages:
        parts.append(f'<|im_start|>{msg["role"]}\n{msg["content"]}<|im_end|>\n')
    parts.append('<|im_start|>assistant\n')
    return ''.join(parts)


# ==============================================================================
# Context budget truncation
# ==============================================================================

def _truncate_messages_to_fit(
    chat_template: str,
    messages: list[dict[str, str]],
    context_length: int,
    max_tokens: int,
    force_chatml: bool = False,
) -> tuple[list[dict[str, str]], dict]:
    """
    Truncate messages to fit within context_length.

    Strategy:
    1. Remove history messages from start (oldest first).
    2. Truncate system message context chunks from end.
    3. Keep current question intact.

    Token counting uses tiktoken cl100k_base (±15% vs Qwen3 tokenizer).
    The 100-token safety_margin absorbs this variance.
    """
    safety_margin = 100
    available_budget = context_length - max_tokens - safety_margin

    def _count_prompt(msgs: list[dict[str, str]]) -> int:
        if force_chatml:
            return _count_tokens(_fallback_chatml_prompt(msgs))
        return _count_tokens(_messages_to_prompt(chat_template, msgs))

    total_tokens = _count_prompt(messages)
    truncation_info = {
        'truncated': False,
        'original_tokens': total_tokens,
        'available_budget': available_budget,
        'history_messages_removed': 0,
        'system_content_truncated': False,
    }

    if total_tokens <= available_budget:
        return messages, truncation_info

    truncation_info['truncated'] = True
    truncated_messages = [msg.copy() for msg in messages]

    # Strategy 1: remove oldest history messages
    if len(truncated_messages) > 2:
        history_start = 1
        history_end = len(truncated_messages) - 1
        for i in range(history_start, history_end):
            test_messages = [truncated_messages[0]] + truncated_messages[i + 1:]
            test_tokens = _count_prompt(test_messages)
            if test_tokens <= available_budget:
                truncated_messages = test_messages
                truncation_info['history_messages_removed'] = i - history_start + 1
                total_tokens = _count_prompt(truncated_messages)
                if total_tokens <= available_budget:
                    truncation_info['final_tokens'] = total_tokens
                    return truncated_messages, truncation_info
                break

    # Strategy 2: truncate system message context chunks from end
    system_content = truncated_messages[0]['content']
    context_marker = 'Context:\n'
    marker_pos = system_content.find(context_marker)

    if marker_pos != -1:
        system_prompt_part = system_content[:marker_pos + len(context_marker)]
        context_part = system_content[marker_pos + len(context_marker):]
        chunks: list[str] = []
        if context_part:
            parts = context_part.split('\n\n')
            current_chunk: list[str] = []
            for part in parts:
                if part.strip().startswith('[Source:'):
                    if current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [part]
                else:
                    if current_chunk:
                        current_chunk.append(part)
                    else:
                        chunks.append(part)
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))

        if chunks:
            for i in range(len(chunks) - 1, -1, -1):
                remaining_chunks = chunks[:i]
                new_context = '\n\n'.join(remaining_chunks) if remaining_chunks else ''
                new_system_content = system_prompt_part + new_context
                test_messages = [{'role': 'system', 'content': new_system_content}] + truncated_messages[1:]
                test_tokens = _count_prompt(test_messages)
                if test_tokens <= available_budget:
                    truncated_messages[0]['content'] = new_system_content
                    truncation_info['system_content_truncated'] = True
                    truncation_info['chunks_removed'] = len(chunks) - len(remaining_chunks)
                    truncation_info['final_tokens'] = _count_prompt(truncated_messages)
                    return truncated_messages, truncation_info

    truncation_info['final_tokens'] = _count_prompt(truncated_messages)
    truncation_info['warning'] = 'Prompt still exceeds budget after truncation'
    return truncated_messages, truncation_info


# ==============================================================================
# Stream worker — runs in a background thread
# ==============================================================================

def _run_stream_worker(
    server: object,
    messages: list[dict[str, str]],
    max_tok: int,
    temp: float,
    top_p_val: float,
    stop_seqs: list[str],
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[str | object],
    exception_holder: list[BaseException],
    cancel_event: threading.Event,
) -> None:
    # Run the blocking xllamacpp generation call in a background thread.
    # Sends messages via handle_chat_completions (OpenAI-compatible chat API)
    # with stream=True. Each text chunk from the delta stream is pushed to
    # the asyncio queue via call_soon_threadsafe so the event loop is never
    # blocked.
    #
    # Response format: OpenAI chat completions streaming delta —
    #   {"choices": [{"delta": {"content": "token"}, "finish_reason": null}]}
    # Final chunk: {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    #
    # Cancellation: when cancel_event is set (consumer disconnect or timeout),
    # the callback stops pushing tokens. C++ generation may continue briefly
    # until the current n_predict budget is exhausted; output is discarded.
    try:
        payload = json.dumps({
            'messages':    messages,
            'max_tokens':  max_tok,
            'temperature': temp,
            'top_p':       top_p_val,
            'stop':        stop_seqs or [],
            'stream':      True,
        })

        finish_reason: str | None = None

        def _callback(chunk: object) -> None:
            nonlocal finish_reason
            if cancel_event.is_set():
                return

            # Parse chunk — xllamacpp delivers JSON dicts or JSON strings
            if isinstance(chunk, dict):
                data = chunk
            elif isinstance(chunk, (str, bytes)):
                raw = chunk if isinstance(chunk, str) else chunk.decode('utf-8', errors='replace')
                # Strip SSE "data: " prefix if present
                if raw.startswith('data:'):
                    raw = raw[5:].strip()
                try:
                    data = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    # Treat the raw string as token text directly
                    if raw:
                        loop.call_soon_threadsafe(queue.put_nowait, raw)
                    return
            else:
                return

            if not isinstance(data, dict):
                return

            # OpenAI chat completions streaming format: choices[0].delta.content
            choices = data.get('choices') or []
            choice = choices[0] if choices else {}
            delta = choice.get('delta') or {}
            token = delta.get('content') or ''

            # Fallback: llama.cpp native completions format uses top-level 'content'
            if not token:
                token = data.get('content') or ''

            if token and not cancel_event.is_set():
                loop.call_soon_threadsafe(queue.put_nowait, token)

            # Finish reason from OpenAI format
            fr = choice.get('finish_reason')
            if fr:
                finish_reason = _normalize_finish_reason(fr)
            elif data.get('stop', False):
                # Legacy llama.cpp server format
                if data.get('stopped_eos', False) or data.get('stopped_word', False):
                    finish_reason = 'stop'
                elif data.get('stopped_limit', False):
                    finish_reason = 'length'
                else:
                    finish_reason = _normalize_finish_reason(data.get('stop_type'))

        server.handle_chat_completions(payload, _callback)  # type: ignore[attr-defined]

        if cancel_event.is_set():
            finish_reason = 'cancelled'

        loop.call_soon_threadsafe(
            queue.put_nowait,
            ('__finish_reason__', finish_reason),
        )

    except BaseException as exc:
        exception_holder.append(exc)
    finally:
        with suppress(RuntimeError):
            loop.call_soon_threadsafe(queue.put_nowait, _STREAM_END)


# ==============================================================================
# LLMEngine — lazy-loading xllamacpp wrapper
# ==============================================================================

class LLMEngine:
    # Wraps an xllamacpp Server with lazy loading, automatic download,
    # and async streaming generation. Configured for Apple Metal GPU by default.

    def __init__(self) -> None:
        self._server: object | None = None
        self._chat_template: str = ''

    # -- Internal server accessor ---------------------------------------------

    @property
    def _loaded_server(self) -> object:
        if self._server is None:
            self._load_model()
        return self._server  # type: ignore[return-value]

    # -- Public state ----------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._server is not None

    def ensure_ready(self) -> None:
        """
        Ensure the model is fully loaded and ready for use.
        Triggers lazy loading and runs a minimal completion to verify
        Metal GPU initialisation is complete.

        Raises:
            LLMError: If model loading or verification fails.
        """
        server = self._loaded_server
        try:
            warmup_payload = json.dumps({'prompt': 'test', 'n_predict': 1, 'temperature': 0.0})
            server.handle_completions(warmup_payload)  # type: ignore[attr-defined]
            log.debug('llm_model_ready_verified', msg='Model readiness verified via warmup completion')
        except Exception as exc:
            raise LLMError(f'Model verification failed — model may not be fully ready: {exc}') from exc

    def unload(self) -> None:
        """Unload the model and free GPU memory."""
        if self._server is not None:
            del self._server
            self._server = None
            self._chat_template = ''
            import gc
            gc.collect()
            log.debug('llm_model_unloaded')

    def count_tokens(self, text: str) -> int:
        # Count tokens using tiktoken cl100k_base (±15% vs Qwen3 tokenizer).
        # Used by the RAG pipeline for context budget management.
        return _count_tokens(text)

    # -- Model path -----------------------------------------------------------

    def _get_model_path(self) -> Path:
        return settings.models_dir / settings.llm_model_filename

    # -- Model loading --------------------------------------------------------

    def _load_model(self, model_filename: str | None = None) -> None:
        """
        Load the GGUF model via xllamacpp.

        When llm_local_only is True, only load from models_dir; never download.
        When False, download from Hugging Face if the file is missing.

        Args:
            model_filename: Optional filename for profile lookup. If None, uses
                            settings.llm_model_filename. Allows the model path
                            and profile to be resolved independently.
        """
        if self._server is not None:
            self.unload()

        model_path = self._get_model_path()
        profile_filename = model_filename if model_filename is not None else settings.llm_model_filename

        if not model_path.exists():
            local_only = settings.full_privacy or settings.llm_local_only
            if local_only:
                raise LLMError(
                    f'LLM model not found at {model_path}. '
                    'Place your GGUF file in the models directory '
                    f'({settings.models_dir}) or turn off Full Privacy Mode (Settings) '
                    'or set INFORMITY_FULL_PRIVACY=false to allow download.'
                )
            log.info('model_not_found_locally', path=str(model_path), filename=profile_filename)
            self._download_model(model_path)

        profile = get_profile_for_filename(profile_filename)
        ctx_len = profile.context_length

        log.info(
            'loading_llm_model',
            path           = str(model_path),
            context_length = ctx_len,
            n_batch        = 512,
            n_threads      = settings.llm_cpu_threads,
        )

        start = time.perf_counter()

        try:
            from xllamacpp import CommonParams, Server  # type: ignore[import-untyped]

            params = CommonParams()
            params.model.path             = str(model_path)
            params.n_ctx                  = ctx_len
            params.n_gpu_layers           = -1    # Offload all layers to Metal GPU
            params.n_batch                = 512   # Reduce peak CPU during prompt prefill
            params.cpuparams.n_threads    = settings.llm_cpu_threads  # Cap CPU threads
            params.cpuparams_batch.n_threads = settings.llm_cpu_threads

            # Read chat template from GGUF metadata before constructing Server,
            # while we still have direct file access.
            self._chat_template = _read_gguf_chat_template(model_path)

            # Suppress C-layer output (Metal init messages, n_ctx warnings, etc.)
            with open(os.devnull, 'w') as devnull, redirect_stderr(devnull):
                self._server = Server(params)

        except ImportError as exc:
            raise LLMError(f'xllamacpp is not installed: {exc}') from exc
        except AttributeError as exc:
            raise LLMError(f'xllamacpp parameter mapping failed — API mismatch: {exc}') from exc
        except ValueError as exc:
            raise LLMError(f'Invalid model configuration: {exc}') from exc
        except RuntimeError as exc:
            raise LLMError(f'Failed to load LLM model "{model_path.name}": {exc}') from exc

        elapsed_ms = (time.perf_counter() - start) * 1000
        log.info(
            'llm_model_loaded',
            model               = model_path.name,
            elapsed_ms          = round(elapsed_ms, 1),
            chat_template_found = bool(self._chat_template),
        )

    # -- Model download -------------------------------------------------------

    def _download_model(
        self,
        target_path: Path,
        repo_id:     str | None = None,
        filename:    str | None = None,
    ) -> None:
        # Download the GGUF model from Hugging Face Hub to the local models dir.
        repo  = repo_id or settings.llm_hf_repo
        fname = filename or target_path.name

        log.info('downloading_llm_model', repo=repo, filename=fname, target=str(target_path))
        start = time.perf_counter()

        try:
            from huggingface_hub import hf_hub_download

            from informity.config import DirNames, configure_hf_environment

            configure_hf_environment()
            hf_cache = settings.cache_dir / DirNames.HUGGINGFACE / DirNames.HUB if settings.cache_dir else None

            ensure_file_directory(target_path)

            download_kwargs: dict = {
                'repo_id':   repo,
                'filename':  fname,
                'local_dir': str(target_path.parent),
            }
            if hf_cache:
                download_kwargs['cache_dir'] = str(hf_cache)

            downloaded_path = hf_hub_download(**download_kwargs)

            downloaded = Path(downloaded_path)
            if downloaded.name != target_path.name and downloaded.exists():
                downloaded.rename(target_path)
                log.debug('model_renamed', from_name=downloaded.name, to_name=target_path.name)

        except ImportError as exc:
            raise LLMError(f'huggingface-hub is not installed: {exc}') from exc
        except OSError as exc:
            raise LLMError(f'Failed to download model from {repo}/{fname}: {exc}') from exc

        elapsed_s = time.perf_counter() - start
        size_mb   = target_path.stat().st_size / (1024 * 1024) if target_path.exists() else 0
        log.info(
            'llm_model_downloaded',
            repo=repo, filename=fname,
            size_mb=round(size_mb, 1), elapsed_s=round(elapsed_s, 1),
        )
        remove_models_dir_cache()

    # -- Synchronous chat completion ------------------------------------------

    def chat_complete(
        self,
        messages: list[dict],
        max_tokens: int = 400,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        response_format: dict | None = None,
    ) -> dict:
        """
        Synchronous (blocking) chat completion via xllamacpp.

        Uses server.handle_chat_completions with stream=False. Returns a dict
        compatible with the OpenAI chat completions format:
        {'choices': [{'message': {'content': '...'}}]}.

        Intended for internal callers (classifier, planner, warmup) that need
        deterministic, non-streaming responses.

        Raises:
            LLMError: If inference fails.
        """
        server = self._loaded_server

        payload_dict: dict = {
            'messages':    messages,
            'max_tokens':  max_tokens,
            'temperature': temperature,
            'stop':        stop or [],
            'stream':      False,
        }
        if response_format is not None:
            payload_dict['response_format'] = response_format
        payload = json.dumps(payload_dict)

        collected: list[dict] = []

        def _cb(chunk: object) -> None:
            if isinstance(chunk, dict):
                collected.append(chunk)
            elif isinstance(chunk, (str, bytes)):
                raw = chunk if isinstance(chunk, str) else chunk.decode('utf-8', errors='replace')
                if raw.startswith('data:'):
                    raw = raw[5:].strip()
                with suppress(json.JSONDecodeError, ValueError):
                    collected.append(json.loads(raw))

        try:
            server.handle_chat_completions(payload, _cb)  # type: ignore[attr-defined]
        except Exception as exc:
            raise LLMError(f'Chat completion inference failed: {exc}') from exc

        # Assemble content from collected chunks.
        # Non-streaming: {'choices': [{'message': {'content': '...'}}]}
        # Streaming delta: {'choices': [{'delta': {'content': '...'}}]}
        content_parts: list[str] = []
        for chunk in collected:
            for choice in chunk.get('choices', []):
                msg = choice.get('message', {})
                if msg.get('content'):
                    content_parts.append(msg['content'])
                delta = choice.get('delta', {})
                if delta.get('content'):
                    content_parts.append(delta['content'])

        return {'choices': [{'message': {'content': ''.join(content_parts)}}]}

    # -- Streaming generation -------------------------------------------------

    async def generate_stream(
        self,
        messages:        list[dict[str, str]],
        max_tokens:      int | None       = None,
        temperature:     float | None     = None,
        top_p:           float | None     = None,
        stop:            list[str] | None = None,
        force_chatml:    bool             = False,
        timeout_seconds: float | None     = None,
    ) -> AsyncGenerator[str | tuple[str, object]]:
        # Stream generated tokens one at a time as an async generator.
        # Renders the prompt via the GGUF's Jinja2 chat template, then drives
        # xllamacpp's handle_completions in a background thread. Tokens are
        # delivered to the asyncio event loop via a thread-safe queue.
        #
        # Args:
        #   messages:        Chat messages (system, user, assistant turns).
        #   max_tokens:      Maximum tokens to generate. Defaults to config value.
        #   temperature:     Sampling temperature. Defaults to config value.
        #   top_p:           Nucleus sampling. Defaults to 1.0.
        #   stop:            Stop sequences to halt generation.
        #   force_chatml:    When True, use ChatML format for token-budget
        #                    estimation in _truncate_messages_to_fit only.
        #                    Has no effect on actual generation — the server
        #                    applies the GGUF template internally via
        #                    handle_chat_completions regardless of this flag.
        #                    Reasoning suppression is controlled by /no_think
        #                    in the user message (model_adapter.prepare_messages).
        #   timeout_seconds: Wall-clock generation timeout. Defaults to 120s.
        #
        # Yields:
        #   str — individual token strings as generated.
        #   tuple[str, dict] — ('__timeout__', ...) marker on timeout.
        #
        # Raises:
        #   LLMError: If messages are empty or generation fails.
        if not messages:
            raise LLMError('Cannot generate from empty messages')

        max_tok   = max_tokens if max_tokens is not None else settings.llm_max_tokens
        temp      = temperature if temperature is not None else settings.llm_temperature
        top_p_val = 1.0 if top_p is None else top_p
        stop_seqs = stop if stop is not None else []
        wall_clock = 120.0 if timeout_seconds is None else float(timeout_seconds)

        profile     = get_profile()
        context_len = profile.context_length
        server      = self._loaded_server

        truncated_messages, truncation_info = _truncate_messages_to_fit(
            chat_template  = self._chat_template,
            messages       = messages,
            context_length = context_len,
            max_tokens     = max_tok,
            force_chatml   = force_chatml,
        )

        if truncation_info['truncated']:
            log.warning(
                'prompt_truncated',
                original_tokens          = truncation_info['original_tokens'],
                final_tokens             = truncation_info.get('final_tokens', truncation_info['original_tokens']),
                available_budget         = truncation_info['available_budget'],
                history_messages_removed = truncation_info.get('history_messages_removed', 0),
                system_content_truncated = truncation_info.get('system_content_truncated', False),
                chunks_removed           = truncation_info.get('chunks_removed', 0),
                warning                  = truncation_info.get('warning'),
            )

        messages = truncated_messages

        log.debug(
            'llm_streaming',
            messages_count  = len(messages),
            max_tokens      = max_tok,
            temperature     = temp,
            top_p           = top_p_val,
            timeout_seconds = wall_clock,
            context_length  = context_len,
        )

        start           = time.perf_counter()
        token_count     = 0
        first_token_ms: float | None = None
        total_text      = ''
        loop            = asyncio.get_running_loop()
        queue: asyncio.Queue[str | object] = asyncio.Queue()
        exception_holder: list[BaseException] = []
        cancel_event = threading.Event()

        worker = threading.Thread(
            target = _run_stream_worker,
            args   = (server, messages, max_tok, temp, top_p_val,
                      stop_seqs, loop, queue, exception_holder, cancel_event),
            name   = 'llm-stream-worker',
            daemon = True,
        )
        worker.start()

        finish_reason: str | None = None
        timeout_occurred = False
        timeout_reason: str | None = None

        # Think-block filter state — strips <think>...</think> from the token stream.
        # Qwen3 models emit a think block before the answer when reasoning is enabled.
        # These blocks should not be included in the displayed answer or word counts.
        _in_think_block = False
        _think_partial   = ''  # accumulates partial tag text to detect split-token boundaries

        try:
            while True:
                elapsed = time.perf_counter() - start
                if elapsed >= wall_clock:
                    log.warning(
                        'llm_stream_wall_clock_timeout',
                        elapsed_seconds  = round(elapsed, 1),
                        timeout_seconds  = wall_clock,
                        tokens_generated = token_count,
                        msg              = 'Hard timeout reached; stopping generation',
                    )
                    cancel_event.set()
                    timeout_occurred = True
                    timeout_reason = 'wall_clock_limit'
                    break

                remaining_timeout = wall_clock - elapsed
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=remaining_timeout)
                except TimeoutError:
                    log.warning(
                        'llm_stream_queue_timeout',
                        elapsed_seconds  = round(time.perf_counter() - start, 1),
                        timeout_seconds  = wall_clock,
                        tokens_generated = token_count,
                        msg              = 'Queue timeout; stopping generation',
                    )
                    cancel_event.set()
                    timeout_occurred = True
                    timeout_reason = 'queue_wait_timeout'
                    break

                if item is _STREAM_END:
                    # Flush the partial-tag safety buffer.  The inner loop keeps
                    # up to 6 chars buffered to detect a split '<think>' tag; on
                    # normal stream end those chars must be emitted or they are
                    # silently lost (e.g. the final word of a short answer).
                    if not _in_think_block and _think_partial:
                        if first_token_ms is None:
                            first_token_ms = (time.perf_counter() - start) * 1000
                        token_count += 1
                        total_text  += _think_partial
                        yield _think_partial
                        _think_partial = ''
                    break
                if isinstance(item, tuple) and len(item) == 2 and item[0] == '__finish_reason__':
                    finish_reason = item[1]
                    continue

                raw_token = str(item)

                # Think-block filtering: accumulate into a rolling buffer and strip
                # <think>...</think> regions before emitting to the consumer.
                _think_partial += raw_token
                emit_text = ''
                while _think_partial:
                    if not _in_think_block:
                        start_pos = _think_partial.find('<think>')
                        if start_pos == -1:
                            # No think block opening. Keep last 6 chars buffered in case
                            # '<think>' is split across tokens; emit the rest.
                            safe_len = max(0, len(_think_partial) - 6)
                            emit_text += _think_partial[:safe_len]
                            _think_partial = _think_partial[safe_len:]
                            break
                        # Emit text before the think block, then enter think mode.
                        emit_text += _think_partial[:start_pos]
                        _in_think_block = True
                        _think_partial = _think_partial[start_pos + len('<think>'):]
                    else:
                        end_pos = _think_partial.find('</think>')
                        if end_pos == -1:
                            # Still inside think block — discard buffered content,
                            # keep last 7 chars in case '</think>' is split.
                            _think_partial = _think_partial[max(0, len(_think_partial) - 7):]
                            break
                        # Discard up to and including </think>, exit think mode.
                        _in_think_block = False
                        _think_partial = _think_partial[end_pos + len('</think>'):]

                if not emit_text:
                    continue

                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - start) * 1000
                token_count += 1
                total_text  += emit_text
                yield emit_text

            worker.join(timeout=60.0)
            if worker.is_alive():
                log.warning('llm_stream_worker_timeout', msg='Worker thread did not exit within 60s')

            if timeout_occurred:
                timeout_notice = f'\n\n[Response truncated: generation time limit ({int(wall_clock)}s) reached]'
                total_text += timeout_notice
                yield timeout_notice
                yield ('__timeout__', {
                    'reason':          timeout_reason or 'unknown_timeout',
                    'elapsed_seconds': round(time.perf_counter() - start, 1),
                    'timeout_seconds': wall_clock,
                })

            if exception_holder:
                exc = exception_holder[0]
                if not isinstance(exc, Exception):
                    raise exc
                raise LLMError(f'LLM streaming failed: {exc}') from exc

        except LLMError:
            raise

        except RuntimeError as exc:
            raise LLMError(f'LLM streaming failed: {exc}') from exc

        except GeneratorExit:
            cancel_event.set()
            log.debug(
                'llm_stream_cancelled',
                tokens_generated = token_count,
                msg              = 'Stream cancelled (GeneratorExit); worker signaled to stop',
            )
            return

        except asyncio.CancelledError:
            cancel_event.set()
            log.debug(
                'llm_stream_cancelled',
                tokens_generated = token_count,
                msg              = 'Stream cancelled (CancelledError); worker signaled to stop',
            )
            raise

        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            log.info(
                'llm_stream_completed',
                messages_count   = len(messages),
                tokens           = token_count,
                output_length    = len(total_text),
                elapsed_ms       = round(elapsed_ms, 1),
                first_token_ms   = round(first_token_ms, 1) if first_token_ms is not None else None,
                finish_reason    = finish_reason,
                cancelled        = cancel_event.is_set(),
                timeout_occurred = timeout_occurred,
                timeout_reason   = timeout_reason,
            )


# ==============================================================================
# Module-level singleton
# ==============================================================================

llm_engine = LLMEngine()
