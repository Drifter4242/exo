import time
from typing import Any, Callable, Generator, cast, get_args

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.models.cache import KVCache, trim_prompt_cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.api import (
    BenchChatCompletionTaskParams,
    ChatCompletionMessage,
    FinishReason,
    GenerationStats,
)
from exo.shared.types.memory import Memory
from exo.shared.types.mlx import KVCacheType
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
)
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.cache import KVPrefixCache, encode_prompt
from exo.worker.engines.mlx.constants import KV_BITS, KV_GROUP_SIZE, MAX_TOKENS
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    make_kv_cache,
    mx_barrier,
)
from exo.worker.runner.bootstrap import logger

generation_stream = mx.new_stream(mx.default_device())


def maybe_quantize_kv_cache(
    prompt_cache: list[KVCache | Any],
    quantized_kv_start: int,
    kv_group_size: int,
    kv_bits: int | None,
) -> None:
    if kv_bits is None:
        return
    for e, c in enumerate(prompt_cache):
        if hasattr(c, "to_quantized") and c.offset >= quantized_kv_start:
            prompt_cache[e] = c.to_quantized(group_size=kv_group_size, bits=kv_bits)


def prefill(
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
    prompt_tokens: mx.array,
    cache: KVCacheType,
) -> tuple[int, float]:
    """Prefill the KV cache with prompt tokens.

    This runs the model over the prompt tokens to populate the cache,
    then trims off the extra generated token.

    Returns:
        tuple of (tokens_prefilled, tokens_per_sec)
    """
    num_tokens = len(prompt_tokens)
    if num_tokens <= 1:
        # Nothing to prefill - stream_generate will handle single token
        return (0, 0.0)

    tokens_to_prefill = num_tokens - 1
    logger.info(f"Prefilling {tokens_to_prefill} tokens...")
    start_time = time.time()

    def progress_callback(processed: int, total: int) -> None:
        elapsed = time.time() - start_time
        tok_per_sec = processed / elapsed if elapsed > 0 else 0
        logger.info(
            f"Prefill progress: {processed}/{total} tokens ({tok_per_sec:.1f} tok/s)"
        )

    # Use max_tokens=1 because max_tokens=0 is buggy in some mlx_lm versions
    # We just throw away the generated token - we only care about filling the cache
    for _ in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_tokens[:-1],  # Prefill all but last token
        max_tokens=1,
        sampler=sampler,
        prompt_cache=cache,
        prefill_step_size=2048,
        kv_group_size=KV_GROUP_SIZE,
        kv_bits=KV_BITS,
        prompt_progress_callback=progress_callback,
    ):
        break  # Stop after first iteration - cache is now filled
    # Trim the extra token we generated (max_tokens=1 workaround)
    trim_prompt_cache(cache, 1)

    elapsed = time.time() - start_time
    tokens_per_sec = tokens_to_prefill / elapsed if elapsed > 0 else 0.0
    logger.info(
        f"Prefill complete: {tokens_to_prefill} tokens in {elapsed:.2f}s "
        f"({tokens_per_sec:.1f} tok/s)"
    )
    return (tokens_to_prefill, tokens_per_sec)


def warmup_inference(
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
) -> int:
    content = "Prompt to warm up the inference engine. Repeat this."

    warmup_prompt = apply_chat_template(
        tokenizer=tokenizer,
        chat_task_data=ChatCompletionTaskParams(
            model="",
            messages=[
                ChatCompletionMessage(
                    role="user",
                    content=content,
                )
            ],
        ),
    )

    tokens_generated = 0

    cache = make_kv_cache(
        model=model,
    )

    logger.info("Generating warmup tokens")
    for _r in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=warmup_prompt,
        max_tokens=50,
        sampler=sampler,
        prompt_cache=cache,
        prefill_step_size=2048,
        kv_group_size=KV_GROUP_SIZE,
        kv_bits=KV_BITS,
    ):
        logger.info("Generated warmup token: " + str(_r.text))
        tokens_generated += 1

    logger.info("Generated ALL warmup tokens")

    # TODO: Do we want an mx_barrier?
    #  At least this version is actively incorrect, as it should use mx_barrier(group)
    mx_barrier()

    return tokens_generated


def ban_token_ids(token_ids: list[int]) -> Callable[[mx.array, mx.array], mx.array]:
    token_ids = [int(t) for t in token_ids]

    def proc(_history: mx.array, logits: mx.array) -> mx.array:
        for tid in token_ids:
            logits[..., tid] = -1e9
        return logits

    return proc


def eos_ids_from_tokenizer(tokenizer: TokenizerWrapper) -> list[int]:
    eos: list[int] | None = getattr(tokenizer, "eos_token_ids", None)
    if eos is None:
        return []
    return eos


def make_sampler_from_task(
    task: ChatCompletionTaskParams,
) -> Callable[[mx.array], mx.array]:
    """Create a sampler configured from task parameters.

    Supports OpenAI-compatible parameters:
    - temperature: Controls randomness (0.0 = deterministic, higher = more random)
    - top_p: Nucleus sampling threshold (0.0 = disabled)
    """
    temp = task.temperature if task.temperature is not None else 0.7
    top_p = task.top_p if task.top_p is not None else 0.0
    logger.info(f"Creating sampler with temp={temp}, top_p={top_p}")
    return make_sampler(temp=temp, top_p=top_p)


def make_logits_processors_from_task(
    task: ChatCompletionTaskParams,
) -> list[Callable[[mx.array, mx.array], mx.array]]:
    """Create logits processors configured from task parameters.

    Supports OpenAI-compatible parameters:
    - logit_bias: Dict of token_id -> bias to add to logits
    - frequency_penalty: Mapped to repetition_penalty (similar but not identical)

    Note: presence_penalty is not directly supported by mlx_lm.
    frequency_penalty in OpenAI penalizes based on frequency in generated text,
    while repetition_penalty in mlx_lm penalizes any repetition in context window.
    """
    # Convert logit_bias from OpenAI format (str keys) to mlx_lm format (int keys)
    logit_bias: dict[int, float] | None = None
    if task.logit_bias:
        logit_bias = {int(k): float(v) for k, v in task.logit_bias.items()}

    # Map frequency_penalty to repetition_penalty
    # OpenAI range: -2.0 to 2.0, mlx_lm expects positive values where 1.0 = no penalty
    # penalty < 1.0 boosts repetition, penalty > 1.0 penalizes repetition
    repetition_penalty: float | None = None
    if task.frequency_penalty is not None and task.frequency_penalty != 0.0:
        # Convert: OpenAI 0.0 -> mlx 1.0, OpenAI 2.0 -> mlx 1.5, OpenAI -2.0 -> mlx 0.5
        # This is an approximation - the algorithms differ
        repetition_penalty = 1.0 + (task.frequency_penalty * 0.25)
        # Clamp to avoid division by zero or extreme values
        # Min 0.1 (strong boost), Max 2.0 (strong penalty)
        repetition_penalty = max(0.1, min(2.0, repetition_penalty))

    processors = make_logits_processors(
        logit_bias=logit_bias,
        repetition_penalty=repetition_penalty,
    )

    if processors:
        logger.info(
            f"Created logits processors: logit_bias={logit_bias is not None}, "
            f"repetition_penalty={repetition_penalty}"
        )

    return processors


def mlx_generate(
    model: Model,
    tokenizer: TokenizerWrapper,
    task: ChatCompletionTaskParams,
    kv_prefix_cache: KVPrefixCache | None = None,
) -> Generator[GenerationResponse]:
    # Ensure that generation stats only contains peak memory for this generation
    mx.reset_peak_memory()
    is_bench: bool = isinstance(task, BenchChatCompletionTaskParams)

    # Currently we support chat-completion tasks only.
    logger.info(f"task_params: {task}")

    sampler = make_sampler_from_task(task)
    logits_processors = make_logits_processors_from_task(task)

    prompt = apply_chat_template(
        tokenizer=tokenizer,
        chat_task_data=task,
    )

    # Use prefix cache if available, otherwise create fresh cache
    if kv_prefix_cache is not None:
        caches, prompt_tokens = kv_prefix_cache.get_kv_cache(model, tokenizer, prompt)
    else:
        caches = make_kv_cache(model=model)
        prompt_tokens = encode_prompt(tokenizer, prompt)

    logits_processors: list[Callable[[mx.array, mx.array], mx.array]] = []
    if is_bench:
        # Only sample length eos tokens
        eos_ids = eos_ids_from_tokenizer(tokenizer)
        logits_processors = [ban_token_ids(eos_ids)]

    # Prefill cache with all tokens except the last one
    prefill_tokens, prefill_tps = prefill(
        model, tokenizer, sampler, prompt_tokens, caches
    )

    # stream_generate starts from the last token
    last_token = prompt_tokens[-1:]

    max_tokens = task.max_tokens or MAX_TOKENS
    generated_text_parts: list[str] = []
    generation_start_time = time.time()
    for out in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=last_token,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors if logits_processors else None,
        prompt_cache=caches,
        # TODO: Dynamically change prefill step size to be the maximum possible without timing out.
        prefill_step_size=2048,
        kv_group_size=KV_GROUP_SIZE,
        kv_bits=KV_BITS,
    ):
        generated_text_parts.append(out.text)
        logger.info(out.text)

        stats: GenerationStats | None = None
        if out.finish_reason is not None:
            stats = GenerationStats(
                prompt_tps=float(out.prompt_tps),
                generation_tps=float(out.generation_tps),
                prompt_tokens=int(out.prompt_tokens),
                generation_tokens=int(out.generation_tokens),
                peak_memory_usage=Memory.from_gb(out.peak_memory),
            )

            if out.finish_reason not in get_args(FinishReason):
                # We don't throw here as this failure case is really not all that bad
                # Just log the error and move on
                logger.warning(
                    f"Model generated unexpected finish_reason: {out.finish_reason}"
                )

        yield GenerationResponse(
            text=out.text,
            token=out.token,
            finish_reason=cast(FinishReason | None, out.finish_reason),
            stats=stats,
        )

        if out.finish_reason is not None:
            # Log generation stats
            generation_elapsed = time.time() - generation_start_time
            generated_tokens = len(generated_text_parts)
            generation_tps = (
                generated_tokens / generation_elapsed if generation_elapsed > 0 else 0.0
            )
            logger.info(
                f"Generation complete: prefill {prefill_tokens} tokens @ "
                f"{prefill_tps:.1f} tok/s, generated {generated_tokens} tokens @ "
                f"{generation_tps:.1f} tok/s"
            )
            # Save cache for future prefix matching (clear first to keep only the last one)
            if kv_prefix_cache is not None:
                kv_prefix_cache.clear()
                full_prompt = prompt + "".join(generated_text_parts)
                kv_prefix_cache.add_kv_cache(tokenizer, full_prompt, caches)
            break

        # TODO: Do we want an mx_barrier?
