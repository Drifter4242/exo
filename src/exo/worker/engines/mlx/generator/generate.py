import time
from typing import Any, Callable, Generator, cast, get_args

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.models.cache import KVCache, trim_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.api import ChatCompletionMessage, FinishReason
from exo.shared.types.mlx import KVCacheType
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
)
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.cache import KVPrefixCache, encode_prompt
from exo.worker.engines.mlx.constants import (
    KV_BITS,
    KV_GROUP_SIZE,
    MAX_TOKENS,
    TEMPERATURE,
)
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
) -> None:
    """Prefill the KV cache with prompt tokens.

    This runs the model over the prompt tokens to populate the cache,
    then trims off the extra generated token.
    """
    num_tokens = len(prompt_tokens)
    if num_tokens <= 1:
        # Nothing to prefill - stream_generate will handle single token
        return

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
    tokens_per_sec = tokens_to_prefill / elapsed if elapsed > 0 else 0
    logger.info(
        f"Prefill complete: {tokens_to_prefill} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )


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
        prefill_step_size=65536,
        kv_group_size=KV_GROUP_SIZE,
        kv_bits=KV_BITS,
    ):
        logger.info("Generated warmup token: " + str(_r.text))
        tokens_generated += 1

    logger.info("Generated ALL warmup tokens")
    mx_barrier()

    return tokens_generated


def make_sampler_from_task(
    task: ChatCompletionTaskParams,
) -> Callable[[mx.array], mx.array]:
    """Create a sampler configured from task parameters.

    Supports OpenAI-compatible parameters:
    - temperature: Controls randomness (0.0 = deterministic, higher = more random)
    - top_p: Nucleus sampling threshold (0.0 = disabled)

    Note: Other OpenAI parameters like frequency_penalty, presence_penalty,
    and logit_bias would require logit processors, not samplers.
    """
    temp = task.temperature if task.temperature is not None else TEMPERATURE
    top_p = task.top_p if task.top_p is not None else 0.0
    logger.info(f"Creating sampler with temp={temp}, top_p={top_p}")
    return make_sampler(temp=temp, top_p=top_p)


def mlx_generate(
    model: Model,
    tokenizer: TokenizerWrapper,
    task: ChatCompletionTaskParams,
    kv_prefix_cache: KVPrefixCache | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> Generator[GenerationResponse]:
    # Currently we support chat-completion tasks only.
    logger.info(f"task_params: {task}")

    sampler = make_sampler_from_task(task)

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

    # Prefill cache with all tokens except the last one
    prefill(model, tokenizer, sampler, prompt_tokens, caches)

    # stream_generate starts from the last token
    last_token = prompt_tokens[-1:]

    max_tokens = task.max_tokens or MAX_TOKENS
    generated_text_parts: list[str] = []
    for out in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=last_token,
        max_tokens=max_tokens,
        sampler=sampler,
        prompt_cache=caches,
        prefill_step_size=65536,
        kv_group_size=KV_GROUP_SIZE,
        kv_bits=KV_BITS,
    ):
        if is_cancelled is not None and is_cancelled():
            logger.info("Generation cancelled by client")
            out.finish_reason = "stop"

        generated_text_parts.append(out.text)
        logger.info(out.text)
        if out.finish_reason is not None and out.finish_reason not in get_args(
            FinishReason
        ):
            # We don't throw here as this failure case is really not all that bad
            # Just log the error and move on
            logger.warning(
                f"Model generated unexpected finish_reason: {out.finish_reason}"
            )

        yield GenerationResponse(
            text=out.text,
            token=out.token,
            finish_reason=cast(FinishReason | None, out.finish_reason),
        )

        if out.finish_reason is not None:
            # Save cache for future prefix matching (clear first to keep only the last one)
            if kv_prefix_cache is not None:
                kv_prefix_cache.clear()
                full_prompt = prompt + "".join(generated_text_parts)
                kv_prefix_cache.add_kv_cache(tokenizer, full_prompt, caches)
            break
