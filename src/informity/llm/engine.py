# ==============================================================================
# Informity AI — LLM Engine
# Lazy-loads a GGUF model via llama-cpp-python with Metal acceleration.
# Provides synchronous generation and async streaming. Downloads the model
# from Hugging Face Hub if not present locally.
# ==============================================================================

from __future__ import annotations

import asyncio
import os
import re
import shutil
import threading
import time
from collections.abc import AsyncGenerator
from contextlib import redirect_stderr, suppress
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from informity.config import settings
from informity.exceptions import LLMError
from informity.llm.model_adapter import get_profile, get_profile_for_filename
from informity.utils.directory_utils import ensure_file_directory

# Import LogitsProcessorList at runtime (used in _make_min_tokens_processor)
try:
    from llama_cpp import LogitsProcessorList
except ImportError:
    # Fallback if llama_cpp not available yet (shouldn't happen in normal flow)
    LogitsProcessorList = None

if TYPE_CHECKING:
    from llama_cpp import Llama

# ==============================================================================
# Logger
# ==============================================================================

log = structlog.get_logger(__name__)
_PROMPT_RENDER_EXCEPTIONS = (ValueError, TypeError, AttributeError, RuntimeError)
_TOKENIZATION_EXCEPTIONS = (ValueError, TypeError, AttributeError, RuntimeError)

# ==============================================================================
# Constants
# ==============================================================================

# Default filename within the repo (saved as llm_model_filename in app)
# Note: DEFAULT_HF_REPO moved to config.py as llm_hf_repo setting (configurable via env var)
DEFAULT_HF_FILENAME = 'Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf'

# Sentinel put on the queue when the sync stream iterator is exhausted
_STREAM_END: object = object()


def _normalize_finish_reason(reason: str | None) -> str | None:
    """
    Normalize llama-cpp-python finish_reason values to standard values.

    P2 A9: Maps various finish reason strings to known values:
    - 'stop' - hit stop sequence or EOS token
    - 'length' - hit max_tokens limit
    - 'eos' - end of sequence (normalized to 'stop')
    - None - unknown (returns None)

    Args:
        reason: Raw finish_reason from llama-cpp-python

    Returns:
        Normalized finish reason string or None
    """
    if not reason:
        return None

    reason_lower = str(reason).lower().strip()

    # Map common variations to standard values
    if reason_lower in ('stop', 'eos', 'end_of_sequence'):
        return 'stop'
    if reason_lower in ('length', 'max_tokens', 'max_tokens_reached'):
        return 'length'
    if reason_lower == 'cancelled':
        return 'cancelled'

    # Unknown value - return as-is but log for debugging
    log.debug('llm_unknown_finish_reason', reason=reason)
    return reason_lower


def remove_models_dir_cache() -> None:
    # Remove any nested .cache directories left by huggingface_hub after download.
    # With unified HF cache, these shouldn't be created, but clean up if they exist.
    # Check llm/, query-classifier/, and diagnostics/ directories.
    for models_dir in [
        settings.models_dir,
        settings.query_classifier_models_dir,
        settings.diagnostics_models_dir,
    ]:
        if models_dir is None:
            continue
        cache_dir = models_dir / '.cache'
        if cache_dir.is_dir():
            try:
                shutil.rmtree(cache_dir)
                log.info('models_dir_cache_removed', path=str(cache_dir))
            except OSError as exc:
                log.warning('models_dir_cache_remove_failed', path=str(cache_dir), error=str(exc))


def _make_min_tokens_processor(
    min_tokens: int,
    suppress_ids: list[int],
) -> LogitsProcessorList:
    # Create a logits processor that suppresses EOS-related tokens for the first
    # `min_tokens` generated tokens. After that, the model can stop naturally.
    # This prevents premature EOS without affecting content quality — only the
    # stop decision is modified, not temperature, penalties, or sampling.
    if LogitsProcessorList is None:
        raise RuntimeError('LogitsProcessorList not available - llama_cpp not imported')

    _generated = 0

    def _processor(input_ids: list[int], logits: list[float]) -> list[float]:
        nonlocal _generated
        _generated += 1
        if _generated <= min_tokens:
            for token_id in suppress_ids:
                if 0 <= token_id < len(logits):
                    logits[token_id] = float('-inf')
        return logits

    return LogitsProcessorList([_processor])


def _messages_to_prompt(
    model: Llama,
    messages: list[dict[str, str]],
) -> str:
    # Convert chat messages to a prompt string using the GGUF's embedded
    # Jinja2 chat template. The model's native template is ALWAYS preferred
    # because it matches the format the model was trained on.
    #
    # Why this matters:
    #   - Some models use non-standard tokens (e.g. fullwidth Unicode) instead
    #     of standard ChatML <|im_start|> / <|im_end|>.
    #   - If we use a different format, the model doesn't recognize the prompt
    #     and may output reasoning WITHOUT <think> tags, causing reasoning to
    #     leak as plain text.
    #   - The model's native EOS token handles generation stopping at the
    #     token-ID level in llama-cpp-python's raw API, independent of our
    #     text-based stop sequences.
    #   - <think>/</ think> detection is literal string matching, works with
    #     any chat template format.
    #
    # Strategy:
    #   1. Try the GGUF template → always use it if it renders
    #   2. Fall back to ChatML ONLY if GGUF has no template or render fails
    template_str = ''
    try:
        metadata = model.metadata
        if metadata:
            template_str = metadata.get('tokenizer.chat_template', '')
    except (AttributeError, TypeError):
        pass

    if template_str:
        try:
            prompt = _render_gguf_template(template_str, messages)

            if prompt and len(prompt.strip()) > 0:
                # Strip trailing <think> if the template added it as a prefill
                # optimization. Some GGUF builds include <think> at the end so
                # the model starts reasoning immediately. This breaks our
                # streaming <think> block detection because we need to see
                # <think> in the generated output to enter the suppression
                # state. The model will generate <think> on its own.
                prompt = re.sub(r'\s*<think>\s*$', '', prompt)

                log.debug(
                    'gguf_template_used',
                    prompt_len  = len(prompt),
                    prompt_tail = repr(prompt[-80:]),
                )
                return prompt

            log.warning(
                'gguf_template_empty',
                msg = 'GGUF template rendered to empty string; using ChatML fallback',
            )
        except _PROMPT_RENDER_EXCEPTIONS as exc:
            log.warning('gguf_template_render_failed', error=str(exc))
    else:
        log.debug('gguf_template_not_found', msg='No chat template in GGUF metadata')

    # Fallback: build prompt using ChatML format.
    # Only used when the GGUF has no embedded template or rendering fails.
    # This is a reasonable default for ChatML-family models (Qwen, Phi, etc.)
    prompt = _fallback_chatml_prompt(messages)
    log.debug(
        'chatml_fallback_used',
        prompt_len  = len(prompt),
        prompt_tail = repr(prompt[-80:]),
    )
    return prompt


def _render_gguf_template(
    template_str: str,
    messages: list[dict[str, str]],
) -> str:
    # Render the Jinja2 chat template extracted from GGUF metadata.
    # Uses a sandboxed Jinja2 environment with raise_exception support
    # (used by some HuggingFace templates).
    from jinja2 import BaseLoader, Environment

    env = Environment(loader=BaseLoader())
    env.globals['raise_exception'] = lambda msg: (_ for _ in ()).throw(
        ValueError(msg),
    )
    template = env.from_string(template_str)
    return template.render(
        messages              = messages,
        add_generation_prompt = True,
    )


def _fallback_chatml_prompt(messages: list[dict[str, str]]) -> str:
    # Build a ChatML prompt string. This is the proven format for ChatML-family
    # models (Qwen3, Phi, etc.) that works correctly with:
    #   - <|im_end|> stop sequences
    #   - <think>...</think> reasoning block detection
    #   - All our streaming and post-processing code
    parts: list[str] = []
    for msg in messages:
        parts.append(f'<|im_start|>{msg["role"]}\n{msg["content"]}<|im_end|>\n')
    parts.append('<|im_start|>assistant\n')
    return ''.join(parts)


def _truncate_messages_to_fit(
    model: Llama,
    messages: list[dict[str, str]],
    context_length: int,
    max_tokens: int,
    force_chatml: bool = False,
) -> tuple[list[dict[str, str]], dict]:
    """
    Truncate messages to fit within context_length.

    Intelligent truncation strategy:
    1. Remove history messages from start (oldest first)
    2. Truncate system message content from end (remove chunks)
    3. Keep current question intact (never truncated)

    Args:
        model: Llama model instance (for token counting)
        messages: List of message dicts (role, content)
        context_length: Maximum context window size (from model profile)
        max_tokens: Maximum tokens to generate (reserved for output)
        force_chatml: Whether to use ChatML format for prompt conversion

    Returns:
        Tuple of (truncated_messages, truncation_info dict)
    """
    # Safety margin for prompt formatting overhead
    safety_margin = 100

    # Calculate available budget
    available_budget = context_length - max_tokens - safety_margin

    # Convert messages to prompt string for accurate token counting
    if force_chatml:
        prompt_string = _fallback_chatml_prompt(messages)
    else:
        prompt_string = _messages_to_prompt(model, messages)

    # Count tokens in full prompt
    total_tokens = len(model.tokenize(prompt_string.encode('utf-8'), add_bos=False, special=False))

    truncation_info = {
        'truncated': False,
        'original_tokens': total_tokens,
        'available_budget': available_budget,
        'history_messages_removed': 0,
        'system_content_truncated': False,
    }

    # If prompt fits, return as-is
    if total_tokens <= available_budget:
        return messages, truncation_info

    truncation_info['truncated'] = True

    # Create a copy to avoid mutating original
    truncated_messages = [msg.copy() for msg in messages]

    # Strategy 1: Remove history messages from start (oldest first)
    # History messages are between system (index 0) and current question (last)
    # Keep system message and current question intact
    if len(truncated_messages) > 2:  # More than just system + question
        history_start = 1  # After system message
        history_end = len(truncated_messages) - 1  # Before current question

        # Remove history messages from start until we fit
        for i in range(history_start, history_end):
            # Test removal: rebuild prompt without this message
            test_messages = [truncated_messages[0]] + truncated_messages[i+1:]
            if force_chatml:
                test_prompt = _fallback_chatml_prompt(test_messages)
            else:
                test_prompt = _messages_to_prompt(model, test_messages)
            test_tokens = len(model.tokenize(test_prompt.encode('utf-8'), add_bos=False, special=False))

            if test_tokens <= available_budget:
                # This removal fits - apply it
                truncated_messages = test_messages
                removed_count = i - history_start + 1
                truncation_info['history_messages_removed'] = removed_count

                # Re-check if we still need more truncation
                if force_chatml:
                    prompt_string = _fallback_chatml_prompt(truncated_messages)
                else:
                    prompt_string = _messages_to_prompt(model, truncated_messages)
                total_tokens = len(model.tokenize(prompt_string.encode('utf-8'), add_bos=False, special=False))

                if total_tokens <= available_budget:
                    # We fit now - return
                    truncation_info['final_tokens'] = total_tokens
                    return truncated_messages, truncation_info
                break

        # If we removed all history and still exceed, continue to Strategy 2

    # Strategy 2: Truncate system message content from end
    # Parse system message to extract context chunks and truncate from end
    # System message format: "{_SYSTEM_PROMPT}\n\nContext:\n{chunk1}\n\n{chunk2}\n\n..."
    system_msg = truncated_messages[0]
    system_content = system_msg['content']

    # Find "Context:\n" marker
    context_marker = 'Context:\n'
    marker_pos = system_content.find(context_marker)

    if marker_pos != -1:
        # Extract system prompt and context parts
        system_prompt_part = system_content[:marker_pos + len(context_marker)]
        context_part = system_content[marker_pos + len(context_marker):]

        # Split context into chunks (separated by "\n\n")
        # Each chunk starts with "[Source: N] ..."
        chunks = []
        if context_part:
            # Split by double newline, but preserve chunk structure
            parts = context_part.split('\n\n')
            current_chunk = []

            for part in parts:
                # Check if this part starts a new chunk (has [Source: N])
                if part.strip().startswith('[Source:'):
                    # Save previous chunk if exists
                    if current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [part]
                else:
                    # Continuation of current chunk
                    if current_chunk:
                        current_chunk.append(part)
                    else:
                        # First part without [Source:] - treat as standalone
                        chunks.append(part)

            # Add last chunk
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))

        # Remove chunks from end until we fit
        if chunks:
            for i in range(len(chunks) - 1, -1, -1):  # Iterate backwards
                # Test removal: rebuild system message without this chunk
                remaining_chunks = chunks[:i]
                new_context = '\n\n'.join(remaining_chunks) if remaining_chunks else ''

                new_system_content = system_prompt_part + new_context
                test_messages = [{'role': 'system', 'content': new_system_content}] + truncated_messages[1:]

                if force_chatml:
                    test_prompt = _fallback_chatml_prompt(test_messages)
                else:
                    test_prompt = _messages_to_prompt(model, test_messages)
                test_tokens = len(model.tokenize(test_prompt.encode('utf-8'), add_bos=False, special=False))

                if test_tokens <= available_budget:
                    # This removal fits - apply it
                    truncated_messages[0]['content'] = new_system_content
                    truncation_info['system_content_truncated'] = True
                    truncation_info['chunks_removed'] = len(chunks) - len(remaining_chunks)

                    # Final token count
                    if force_chatml:
                        prompt_string = _fallback_chatml_prompt(truncated_messages)
                    else:
                        prompt_string = _messages_to_prompt(model, truncated_messages)
                    total_tokens = len(model.tokenize(prompt_string.encode('utf-8'), add_bos=False, special=False))
                    truncation_info['final_tokens'] = total_tokens
                    return truncated_messages, truncation_info

    # If we still exceed (shouldn't happen, but handle gracefully)
    # Return truncated messages anyway - model will handle it
    if force_chatml:
        prompt_string = _fallback_chatml_prompt(truncated_messages)
    else:
        prompt_string = _messages_to_prompt(model, truncated_messages)
    total_tokens = len(model.tokenize(prompt_string.encode('utf-8'), add_bos=False, special=False))
    truncation_info['final_tokens'] = total_tokens
    truncation_info['warning'] = 'Prompt still exceeds budget after truncation'

    return truncated_messages, truncation_info


def _run_stream_worker(
    model: Llama,
    messages: list[dict[str, str]],
    max_tok: int,
    temp: float,
    top_p_val: float,
    stop_seqs: list[str],
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[str | object],
    exception_holder: list[BaseException],
    cancel_event: threading.Event,
    min_tokens: int = 0,
    force_chatml: bool = False,
    extra_eos_tokens: tuple[str, ...] = (),
) -> None:
    # Run the blocking llama-cpp stream loop in a background thread.
    # Converts messages to a prompt using the GGUF's embedded Jinja2 template,
    # then uses the raw completion API for generation. This gives us:
    #   - Automatic template support for any model (ChatML, Llama 3, etc.)
    #   - Reliable <think> block handling (raw API yields text, not delta)
    #   - Guaranteed logits_processor support
    #   - Proven streaming behavior
    #
    # Pushes each token onto the asyncio queue via call_soon_threadsafe so the
    # event loop is not blocked. On end or error, pushes _STREAM_END.
    #
    # The cancel_event is checked after each token: if the consumer abandons the
    # generator (e.g. client disconnect), the event is set so we stop generating
    # tokens early instead of wasting GPU/CPU on output nobody will read.
    #
    # min_tokens: when > 0, suppress EOS/im_end tokens for the first N tokens
    # to prevent premature generation cutoff (value from model profile).
    #
    # force_chatml: when True, use ChatML format instead of the GGUF's native
    # template. Driven by the model profile for query types that need a
    # different prompt format (e.g. ChatML to bypass reasoning on some models).
    try:
        # Convert messages to a prompt.
        # force_chatml bypasses the GGUF native template to prevent reasoning
        # models from entering <think> loops on large-context coverage queries.
        if force_chatml:
            prompt = _fallback_chatml_prompt(messages)
            log.debug(
                'chatml_forced',
                prompt_len  = len(prompt),
                prompt_tail = repr(prompt[-80:]),
            )
        else:
            prompt = _messages_to_prompt(model, messages)

        # Build logits_processor for min_tokens enforcement
        logits_processor = None
        if min_tokens > 0:
            # Collect EOS-related token IDs to suppress
            suppress_ids = []
            eos_id = model.token_eos()
            if eos_id is not None and eos_id >= 0:
                suppress_ids.append(eos_id)
            # Also suppress <|im_end|> if it's a different token (ChatML models)
            try:
                im_end_ids = model.tokenize(b'<|im_end|>', special=True)
                for tid in im_end_ids:
                    if tid not in suppress_ids and tid >= 0:
                        suppress_ids.append(tid)
            except _TOKENIZATION_EXCEPTIONS:
                pass  # Model doesn't support special tokenization
            # Suppress additional EOS tokens (e.g. Gemma's <<end_of_turn>>)
            for eos_str in extra_eos_tokens:
                try:
                    eos_ids = model.tokenize(eos_str.encode('utf-8'), special=True)
                    for tid in eos_ids:
                        if tid not in suppress_ids and tid >= 0:
                            suppress_ids.append(tid)
                except _TOKENIZATION_EXCEPTIONS:
                    pass  # Model doesn't support tokenization for this string
            if suppress_ids:
                logits_processor = _make_min_tokens_processor(min_tokens, suppress_ids)
                log.debug(
                    'llm_min_tokens_active',
                    min_tokens    = min_tokens,
                    suppress_ids  = suppress_ids,
                )

        call_kwargs: dict = {
            'max_tokens':  max_tok,
            'temperature': temp,
            'top_p':       top_p_val,
            'stop':        stop_seqs or None,
            'echo':        False,
            'stream':      True,
        }
        if logits_processor is not None:
            call_kwargs['logits_processor'] = logits_processor

        stream = model(prompt, **call_kwargs)
        finish_reason = None
        for chunk in stream:
            if cancel_event.is_set():
                finish_reason = 'cancelled'
                break
            token = ''
            if chunk and 'choices' in chunk and chunk['choices']:
                choice        = chunk['choices'][0]
                token         = choice.get('text', '')
                chunk_reason  = choice.get('finish_reason')
                if chunk_reason:
                    finish_reason = _normalize_finish_reason(chunk_reason)
            if token:
                loop.call_soon_threadsafe(queue.put_nowait, token)

        # Push finish reason so the consumer can log why generation stopped.
        # Values: 'stop' (hit stop sequence or EOS), 'length' (hit max_tokens),
        # 'cancelled' (consumer abandoned), None (unknown).
        loop.call_soon_threadsafe(
            queue.put_nowait,
            ('__finish_reason__', finish_reason),
        )
    except BaseException as exc:
        if not isinstance(exc, Exception):
            # KeyboardInterrupt / SystemExit — record but don't swallow
            exception_holder.append(exc)
        else:
            exception_holder.append(exc)
    finally:
        # Always push _STREAM_END so the consumer never blocks forever,
        # regardless of how the worker exits (normal, error, or cancellation).
        # Event loop already closed — nothing to do.
        with suppress(RuntimeError):
            loop.call_soon_threadsafe(queue.put_nowait, _STREAM_END)


# ==============================================================================
# LLMEngine — lazy-loading llama-cpp-python wrapper
# ==============================================================================

class LLMEngine:
    # Wraps a llama-cpp-python Llama model with lazy loading, automatic
    # download, and both synchronous and async streaming generation.
    # Configured for Apple Metal GPU acceleration by default.

    def __init__(self) -> None:
        self._model: Llama | None = None

    # -- Model loading --------------------------------------------------------

    @property
    def model(self) -> Llama:
        # Lazy-load the model on first access.
        # For large models (30B+), the Llama() constructor blocks until Metal GPU
        # initialization is complete, ensuring the model is fully ready before use.
        if self._model is None:
            self._load_model()
        return self._model  # type: ignore[return-value]

    @property
    def is_loaded(self) -> bool:
        # Check whether the model has been loaded without triggering a load.
        return self._model is not None

    def ensure_ready(self) -> None:
        """
        Ensure the model is fully loaded and ready for use.

        For large models (30B+), this verifies that Metal GPU initialization
        is complete by performing a test tokenization. This is useful when
        you want to explicitly wait for model readiness before starting generation.

        Raises:
            LLMError: If model loading or verification fails.
        """
        # Access the model property to trigger lazy loading if not yet loaded
        model = self.model

        # Perform a test tokenization to verify Metal GPU is ready
        # This ensures the model weights are fully loaded into GPU memory
        try:
            test_text = 'ready'
            _ = model.tokenize(test_text.encode('utf-8'), add_bos=False, special=False)
            log.debug('llm_model_ready_verified', msg='Model readiness verified via tokenization')
        except _TOKENIZATION_EXCEPTIONS as exc:
            raise LLMError(
                f'Model verification failed - model may not be fully ready: {exc}'
            ) from exc

    def unload(self) -> None:
        """
        Unload the model and free GPU memory.
        Useful when switching between models (e.g. during evaluation runs).
        """
        if self._model is not None:
            # Delete the model instance to free GPU memory
            del self._model
            self._model = None
            # Trigger Python GC to encourage immediate memory release
            import gc
            gc.collect()
            log.debug('llm_model_unloaded')

    def count_tokens(self, text: str) -> int:
        # Count tokens using the loaded model's tokenizer.
        # Used by the RAG pipeline for accurate context budget so prompt + context
        # fit within n_ctx. add_bos=False so we count only the segment's tokens.
        if not text:
            return 0
        return len(self.model.tokenize(text.encode('utf-8'), add_bos=False, special=False))

    def _get_model_path(self) -> Path:
        # Resolve the full path to the GGUF model file.
        return settings.models_dir / settings.llm_model_filename

    def _load_model(self, model_filename: str | None = None) -> None:
        """
        Load the GGUF model via llama-cpp-python.

        When llm_local_only is True, only load from models_dir; never download.
        When False, download from Hugging Face if the file is missing.

        Args:
            model_filename: Optional model filename for profile lookup. If None,
                          uses settings.llm_model_filename. Allows subclasses
                          (e.g. QualityLLMEngine) to specify a different model
                          for profile lookup without mutating global settings.
        """
        # Unload existing model first to free GPU memory before loading new one
        if self._model is not None:
            self.unload()

        model_path = self._get_model_path()

        # Use provided filename for profile lookup, or fall back to settings
        profile_filename = model_filename if model_filename is not None else settings.llm_model_filename

        if not model_path.exists():
            # When full_privacy is on, always local only; otherwise respect llm_local_only
            local_only = settings.full_privacy or settings.llm_local_only
            if local_only:
                raise LLMError(
                    f'LLM model not found at {model_path}. '
                    'Place your GGUF file in the models directory '
                    f'({settings.models_dir}) or turn off Full Privacy Mode (Settings) or set INFORMITY_FULL_PRIVACY=false to allow download.'
                )
            log.info(
                'model_not_found_locally',
                path     = str(model_path),
                filename = profile_filename,
            )
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
            from llama_cpp import Llama

            # Suppress C-layer messages (ggml_metal_init, n_ctx_per_seq, etc.) that
            # go to stderr; they are informational only (e.g. bf16 kernels not
            # supported on this GPU, context smaller than train size). verbose=False
            # does not affect them.
            with open(os.devnull, 'w') as devnull, redirect_stderr(devnull):
                self._model = Llama(
                    model_path   = str(model_path),
                    n_ctx        = ctx_len,
                    n_gpu_layers = -1,                        # Offload all layers to Metal GPU
                    n_batch      = 512,                       # Reduce peak CPU during prompt prefill
                    n_threads    = settings.llm_cpu_threads,  # Cap llama-cpp CPU threads
                    verbose      = False,
                )
        except ImportError as exc:
            raise LLMError(
                f'llama-cpp-python is not installed: {exc}'
            ) from exc
        except ValueError as exc:
            raise LLMError(
                f'Invalid model configuration: {exc}'
            ) from exc
        except RuntimeError as exc:
            raise LLMError(
                f'Failed to load LLM model "{model_path.name}": {exc}'
            ) from exc

        # Verify model is fully initialized by performing a test tokenization
        # This ensures Metal GPU is ready and model weights are loaded into memory
        # For large models (30B+), this is critical to ensure readiness before use
        try:
            test_text = 'test'
            _ = self._model.tokenize(test_text.encode('utf-8'), add_bos=False, special=False)
            log.debug(
                'llm_model_verified',
                model=model_path.name,
                msg='Model tokenization verified - Metal GPU initialization confirmed',
            )
        except _TOKENIZATION_EXCEPTIONS as exc:
            log.warning(
                'llm_model_verification_failed',
                model=model_path.name,
                error=str(exc),
                msg='Model loaded but verification failed - may not be fully ready',
            )
            # Don't fail loading - model might still work, but log the warning

        elapsed_ms = (time.perf_counter() - start) * 1000
        log.info(
            'llm_model_loaded',
            model      = model_path.name,
            elapsed_ms = round(elapsed_ms, 1),
        )

    # -- Model download -------------------------------------------------------

    def _download_model(
        self,
        target_path: Path,
        repo_id:     str | None = None,
        filename:    str | None = None,
    ) -> None:
        # Download the GGUF model from Hugging Face Hub to the local models dir.
        # Uses hf_hub_download which handles caching and resume.
        # Configured to use unified HF cache to prevent nested .cache directories.
        repo     = repo_id or settings.llm_hf_repo
        # Use the target filename (from settings.llm_model_filename) as the download filename
        # to ensure we download the correct quantization that matches the configured filename
        fname    = filename or target_path.name

        log.info(
            'downloading_llm_model',
            repo     = repo,
            filename = fname,
            target   = str(target_path),
        )

        start = time.perf_counter()

        try:
            from huggingface_hub import hf_hub_download

            from informity.config import DirNames, configure_hf_environment

            # Configure HF environment to use unified cache (prevents nested .cache directories)
            configure_hf_environment()
            hf_cache = settings.cache_dir / DirNames.HUGGINGFACE / DirNames.HUB if settings.cache_dir else None

            # Ensure the models directory exists
            ensure_file_directory(target_path)

            # Download to the models directory with the configured filename
            # Use unified HF cache to prevent creating nested .cache dirs under model directories.
            download_kwargs = {
                'repo_id':   repo,
                'filename':  fname,
                'local_dir': str(target_path.parent),
            }
            if hf_cache:
                download_kwargs['cache_dir'] = str(hf_cache)

            downloaded_path = hf_hub_download(**download_kwargs)

            # hf_hub_download saves with the original filename from the repo;
            # rename to the configured filename if different
            downloaded = Path(downloaded_path)
            if downloaded.name != target_path.name and downloaded.exists():
                downloaded.rename(target_path)
                log.debug(
                    'model_renamed',
                    from_name = downloaded.name,
                    to_name   = target_path.name,
                )

        except ImportError as exc:
            raise LLMError(
                f'huggingface-hub is not installed: {exc}'
            ) from exc
        except OSError as exc:
            raise LLMError(
                f'Failed to download model from {repo}/{fname}: {exc}'
            ) from exc

        elapsed_s = time.perf_counter() - start
        size_mb   = target_path.stat().st_size / (1024 * 1024) if target_path.exists() else 0

        log.info(
            'llm_model_downloaded',
            repo      = repo,
            filename  = fname,
            size_mb   = round(size_mb, 1),
            elapsed_s = round(elapsed_s, 1),
        )

        # Remove huggingface_hub .cache under models_dir; we only need the .gguf file.
        remove_models_dir_cache()

    # -- Streaming generation -------------------------------------------------

    async def generate_stream(
        self,
        messages:        list[dict[str, str]],
        max_tokens:      int | None        = None,
        temperature:     float | None      = None,
        top_p:           float | None      = None,
        stop:            list[str] | None  = None,
        min_tokens:      int                = 0,
        force_chatml:    bool               = False,
        extra_eos_tokens: tuple[str, ...]   = (),
        timeout_seconds: float | None       = None,
    ) -> AsyncGenerator[str | tuple[str, bool]]:
        # Stream generated tokens one at a time as an async generator.
        # Accepts a messages list, converts it to a prompt using the GGUF's
        # embedded Jinja2 template, then generates via the raw completion API.
        # This hybrid approach gives us:
        #   - Automatic template support (ChatML, Llama 3, etc.)
        #   - Reliable <think> block handling (raw API yields text tokens)
        #   - Guaranteed logits_processor support
        #   - Proven streaming behavior
        #
        # This runs the llama-cpp synchronous iterator in a thread executor
        # to avoid blocking the async event loop.
        #
        # Args:
        #   messages:       Chat messages (system, user, assistant turns).
        #   max_tokens:     Maximum tokens to generate. Defaults to config value.
        #   temperature:    Sampling temperature. Defaults to config value.
        #   stop:           Optional list of stop sequences to halt generation.
        #   min_tokens:     Suppress EOS for the first N tokens to prevent
        #                   premature cutoff (e.g. coverage queries). 0 = disabled.
        #   force_chatml:   When True, use ChatML format instead of the GGUF's
        #                   native template. Driven by the model profile for
        #                   query types that need a different prompt format.
        #   timeout_seconds: Wall-clock timeout for generation. None = 120s (chat default).
        #                   Callers (e.g. quality analysis) can pass a longer value.
        #
        # Yields:
        #   Individual token strings as they are generated.
        #
        # Raises:
        #   LLMError: If messages are empty or generation fails.
        if not messages:
            raise LLMError('Cannot generate from empty messages')

        max_tok    = max_tokens if max_tokens is not None else settings.llm_max_tokens
        temp       = temperature if temperature is not None else settings.llm_temperature
        top_p_val  = 1.0 if top_p is None else top_p
        stop_seqs  = stop if stop is not None else []
        wall_clock = 120.0 if timeout_seconds is None else float(timeout_seconds)

        # Get model profile for context_length
        profile = get_profile()
        context_len = profile.context_length

        # Truncate messages to fit within context_length (app-compliant, engine-level enforcement)
        model = self.model
        truncated_messages, truncation_info = _truncate_messages_to_fit(
            model=model,
            messages=messages,
            context_length=context_len,
            max_tokens=max_tok,
            force_chatml=force_chatml,
        )

        # Log truncation if it occurred
        if truncation_info['truncated']:
            log.warning(
                'prompt_truncated',
                original_tokens=truncation_info['original_tokens'],
                final_tokens=truncation_info.get('final_tokens', truncation_info['original_tokens']),
                available_budget=truncation_info['available_budget'],
                history_messages_removed=truncation_info.get('history_messages_removed', 0),
                system_content_truncated=truncation_info.get('system_content_truncated', False),
                chunks_removed=truncation_info.get('chunks_removed', 0),
                warning=truncation_info.get('warning'),
            )

        # Use truncated messages for generation
        messages = truncated_messages

        log.debug(
            'llm_streaming',
            messages_count   = len(messages),
            max_tokens       = max_tok,
            temperature      = temp,
            top_p            = top_p_val,
            min_tokens       = min_tokens,
            force_chatml     = force_chatml,
            timeout_seconds  = wall_clock,
            context_length   = context_len,
        )

        start           = time.perf_counter()
        token_count     = 0
        first_token_ms: float | None = None
        total_text      = ''
        model           = self.model
        loop            = asyncio.get_running_loop()
        queue: asyncio.Queue[str | object] = asyncio.Queue()
        exception_holder: list[BaseException] = []
        cancel_event = threading.Event()

        # Wall-clock timeout: default 120s for chat; diagnostics analysis uses diagnostics_llm_timeout_seconds
        wall_clock_timeout = wall_clock

        worker = threading.Thread(
            target   = _run_stream_worker,
            args     = (model, messages, max_tok, temp, top_p_val, stop_seqs, loop, queue, exception_holder, cancel_event,
                        min_tokens, force_chatml, extra_eos_tokens),
            name     = 'llm-stream-worker',
            daemon   = True,
        )
        worker.start()

        finish_reason = None
        timeout_occurred = False
        timeout_reason: str | None = None

        try:
            while True:
                # Check wall-clock timeout before waiting for next token
                elapsed = time.perf_counter() - start
                if elapsed >= wall_clock_timeout:
                    log.warning(
                        'llm_stream_wall_clock_timeout',
                        elapsed_seconds = round(elapsed, 1),
                        timeout_seconds  = wall_clock_timeout,
                        tokens_generated = token_count,
                        msg              = 'Hard timeout reached; stopping generation',
                    )
                    cancel_event.set()
                    timeout_occurred = True
                    timeout_reason = 'wall_clock_limit'
                    break

                # Wait for next token with remaining timeout
                remaining_timeout = wall_clock_timeout - elapsed
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=remaining_timeout)
                except TimeoutError:
                    log.warning(
                        'llm_stream_queue_timeout',
                        elapsed_seconds = round(time.perf_counter() - start, 1),
                        timeout_seconds  = wall_clock_timeout,
                        tokens_generated = token_count,
                        msg              = 'Queue timeout; stopping generation',
                    )
                    cancel_event.set()
                    timeout_occurred = True
                    timeout_reason = 'queue_wait_timeout'
                    break

                if item is _STREAM_END:
                    break
                # Capture finish reason metadata from worker thread
                if isinstance(item, tuple) and len(item) == 2 and item[0] == '__finish_reason__':
                    finish_reason = item[1]
                    continue
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - start) * 1000
                token_count += 1
                total_text  += str(item)
                yield str(item)

            worker.join(timeout=60.0)
            if worker.is_alive():
                log.warning('llm_stream_worker_timeout', msg='Worker thread did not exit within 60s')

            # A1: If timeout occurred, append notice to total_text and yield it + timeout marker
            if timeout_occurred:
                timeout_notice = f'\n\n[Response truncated: generation time limit ({int(wall_clock_timeout)}s) reached]'
                total_text += timeout_notice
                # Yield the notice text so user sees it in the stream
                yield timeout_notice
                # Yield timeout marker so caller can emit SSE event
                yield ('__timeout__', {
                    'reason': timeout_reason or 'unknown_timeout',
                    'elapsed_seconds': round(time.perf_counter() - start, 1),
                    'timeout_seconds': wall_clock_timeout,
                })

            if exception_holder:
                exc = exception_holder[0]
                # Re-raise BaseException subclasses (KeyboardInterrupt, SystemExit) directly
                # to allow clean process shutdown
                if not isinstance(exc, Exception):
                    raise exc
                raise LLMError(f'LLM streaming failed: {exc}') from exc

        except LLMError:
            raise  # Already wrapped — do not double-wrap

        except RuntimeError as exc:
            raise LLMError(f'LLM streaming failed: {exc}') from exc

        except GeneratorExit:
            # Consumer abandoned the generator (e.g. generator.close()).
            # Signal the worker thread to stop generating tokens early to free GPU/CPU.
            cancel_event.set()
            log.debug(
                'llm_stream_cancelled',
                tokens_generated = token_count,
                msg             = 'Stream cancelled (GeneratorExit); worker thread signaled to stop',
            )
            return

        except asyncio.CancelledError:
            # Task was cancelled (e.g. client disconnect via AbortController).
            # Signal the worker thread to stop generating tokens early to free GPU/CPU.
            cancel_event.set()
            log.debug(
                'llm_stream_cancelled',
                tokens_generated = token_count,
                msg             = 'Stream cancelled (CancelledError); worker thread signaled to stop',
            )
            # Re-raise to allow proper async cleanup
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
