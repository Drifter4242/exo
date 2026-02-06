"""
Batch generation support for concurrent agent conversations.

This module provides BatchGenerator integration for running multiple
inference requests simultaneously, enabling efficient multi-agent workloads.

Architecture: This module yields raw token events. The caller (runner.py)
handles event emission, GPT-OSS processing, and task management.
"""
import os
import time
from collections.abc import Generator
from dataclasses import dataclass, field

from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.common import CommandId
from exo.shared.types.tasks import TaskId
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.cache import encode_prompt
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    detect_thinking_prompt_suffix,
)
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.token_processing import TokenProcessor

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BATCH_ENABLED = os.environ.get("EXO_BATCH_ENABLED", "1").lower() in ("1", "true", "yes")
BATCH_MAX_SIZE = int(os.environ.get("EXO_BATCH_MAX_SIZE", "32"))


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class BatchRequestState:
    """Per-request state for batch generation."""
    # Unified token processor (handles GPT-OSS, Kimi filtering, thinking prepend, etc.)
    token_processor: TokenProcessor = field(default_factory=lambda: TokenProcessor.for_model("", False, False, None))  # type: ignore
    # Accumulated output
    output_text: str = ""
    tokens_generated: int = 0
    # Stop sequence handling
    stop_sequences: list[str] = field(default_factory=list)
    max_stop_len: int = 0
    # Timing for stats
    generation_start_time: float = 0.0
    first_token_time: float = 0.0


@dataclass
class BatchRequest:
    """Tracks a request in the batch generator."""
    uid: int
    command_id: CommandId
    task_id: TaskId
    task_params: TextGenerationTaskParams
    prompt_tokens: list[int]
    state: BatchRequestState = field(default_factory=BatchRequestState)
    created_at: float = field(default_factory=time.time)


@dataclass
class BatchState:
    """Manages batch generation state."""
    generator: BatchGenerator | None = None
    model: Model | None = None
    requests: dict[int, BatchRequest] = field(default_factory=dict)
    command_to_uid: dict[CommandId, int] = field(default_factory=dict)

    def add_request(self, uid: int, request: BatchRequest) -> None:
        self.requests[uid] = request
        self.command_to_uid[request.command_id] = uid

    def remove_request(self, uid: int) -> BatchRequest | None:
        request = self.requests.pop(uid, None)
        if request:
            self.command_to_uid.pop(request.command_id, None)
        return request

    def get_uid_for_command(self, command_id: CommandId) -> int | None:
        return self.command_to_uid.get(command_id)

    def is_active(self) -> bool:
        return self.generator is not None and len(self.requests) > 0


@dataclass
class BatchTokenEvent:
    """A token event yielded by the batch generator."""
    request: BatchRequest
    token_id: int
    token_text: str
    finish_reason: str | None


# -----------------------------------------------------------------------------
# State Initialization
# -----------------------------------------------------------------------------

def init_request_state(
    task_params: TextGenerationTaskParams,
    model: Model,
    model_id: str,
    prompt: str,
    tokenizer: TokenizerWrapper,
) -> BatchRequestState:
    """Initialize per-request state for batch generation."""
    stop_sequences: list[str] = []
    if task_params.stop is not None:
        if isinstance(task_params.stop, str):
            stop_sequences = [task_params.stop]
        else:
            stop_sequences = list(task_params.stop)

    needs_think_prepend = detect_thinking_prompt_suffix(prompt, tokenizer)
    token_processor = TokenProcessor.for_model(
        model=model,
        model_id=model_id,
        needs_think_prepend=needs_think_prepend,
        tokenizer=tokenizer,
    )

    return BatchRequestState(
        token_processor=token_processor,
        stop_sequences=stop_sequences,
        max_stop_len=max((len(s) for s in stop_sequences), default=0),
        generation_start_time=time.perf_counter(),
    )


# -----------------------------------------------------------------------------
# Batch Operations
# -----------------------------------------------------------------------------

def insert_batch_request(
    batch_state: BatchState,
    task_id: TaskId,
    command_id: CommandId,
    task_params: TextGenerationTaskParams,
    tokenizer: TokenizerWrapper,
    model_id: str,
) -> BatchRequest:
    """
    Insert a new request into the batch.
    
    Returns the created BatchRequest.
    """
    assert batch_state.generator is not None, "BatchGenerator must be initialized"
    
    prompt = apply_chat_template(tokenizer, task_params)
    prompt_tokens = encode_prompt(tokenizer, prompt).tolist()
    
    req_sampler = make_sampler(
        temp=task_params.temperature if task_params.temperature is not None else 0.7,
        top_p=task_params.top_p if task_params.top_p is not None else 1.0,
        top_k=task_params.top_k if task_params.top_k is not None else 0,
    )
    
    max_tokens = task_params.max_output_tokens or 4096
    uids = batch_state.generator.insert(
        [prompt_tokens],
        max_tokens=[max_tokens],
        samplers=[req_sampler],
    )
    uid = uids[0]
    
    request = BatchRequest(
        uid=uid,
        command_id=command_id,
        task_id=task_id,
        task_params=task_params,
        prompt_tokens=prompt_tokens,
        state=init_request_state(task_params, batch_state.model, model_id, prompt, tokenizer),
    )
    batch_state.add_request(uid, request)
    logger.info(
        f"[batch:{uid}] Starting generation: {len(prompt_tokens)} prompt tokens, "
        f"max_tokens={max_tokens}, batch_size={len(batch_state.requests)}"
    )
    
    return request


def cancel_batch_request(
    batch_state: BatchState,
    command_id: CommandId,
) -> BatchRequest | None:
    """
    Cancel and remove a request from the batch.
    
    Returns the removed BatchRequest, or None if not found.
    """
    uid = batch_state.get_uid_for_command(command_id)
    if uid is None:
        return None
    
    assert batch_state.generator is not None
    logger.info(f"Cancelling request {uid} (command {command_id})")
    batch_state.generator.remove([uid])
    return batch_state.remove_request(uid)


def finish_batch_request(
    batch_state: BatchState,
    request: BatchRequest,
    was_stopped_early: bool = False,
) -> None:
    """
    Mark a request as finished and remove from batch.
    
    Args:
        batch_state: The batch state
        request: The request to finish
        was_stopped_early: True if stopped by stop sequence (need to call generator.remove)
    """
    if was_stopped_early and batch_state.generator is not None:
        batch_state.generator.remove([request.uid])
    
    batch_state.remove_request(request.uid)
    logger.info(f"Request {request.uid} finished")


# -----------------------------------------------------------------------------
# Batch Generation
# -----------------------------------------------------------------------------

def batch_generate(
    batch_state: BatchState,
    tokenizer: TokenizerWrapper,
) -> Generator[BatchTokenEvent, None, None]:
    """
    Generate one step of tokens for all active requests.
    
    Yields a BatchTokenEvent for each token generated.
    The caller handles:
    - Event emission
    - GPT-OSS processing
    - Tool call processing
    - Stop sequence handling
    - Finish handling
    
    This is a generator that yields tokens for ONE generation step.
    Call it repeatedly in a loop while batch_state.is_active().
    """
    assert batch_state.generator is not None, "BatchGenerator must be initialized"
    
    gen = batch_state.generator
    responses = gen.next()
    
    if not responses:
        return
    
    for resp in responses:
        request = batch_state.requests.get(resp.uid)
        if not request:
            logger.warning(f"[batch] Response for unknown uid={resp.uid}")
            continue
        
        token_text = tokenizer.decode([resp.token])

        if resp.finish_reason == "stop" and resp.token in gen.stop_tokens:
            token_text = ""
        
        yield BatchTokenEvent(
            request=request,
            token_id=resp.token,
            token_text=token_text,
            finish_reason=resp.finish_reason,
        )


def init_batch_generator(
    batch_state: BatchState,
    model: Model,
    tokenizer: TokenizerWrapper,
) -> None:
    """Initialize the BatchGenerator if not already created."""
    if batch_state.generator is not None:
        return
    
    eos_tokens = set()
    if hasattr(tokenizer, 'eos_token_ids') and tokenizer.eos_token_ids:
        eos_tokens = set(tokenizer.eos_token_ids)
    
    batch_state.generator = BatchGenerator(
        model,
        stop_tokens=eos_tokens,
        completion_batch_size=BATCH_MAX_SIZE,
        prefill_batch_size=8,
        prefill_step_size=2048,
    )
    batch_state.model = model
    logger.info("Initialized BatchGenerator")
