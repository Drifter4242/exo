import concurrent.futures
import gc
import hashlib
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np
import psutil
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)
from mlx_lm.models.deepseek_v4 import (
    DeepseekV4Cache,
)
from mlx_lm.models.deepseek_v4 import (
    _CompressorBranch as CompressorBranch,  # type: ignore
)
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.memory import Memory
from exo.worker.engines.mlx.constants import CACHE_GROUP_SIZE, KV_CACHE_BITS
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from exo.worker.engines.mlx.vision import MediaRegion


# Fraction of device memory above which LRU eviction kicks in.
# Smaller machines need more aggressive eviction.
def _default_memory_threshold() -> float:
    total_gb = Memory.from_bytes(psutil.virtual_memory().total).in_gb
    if total_gb >= 128:
        return 0.85
    if total_gb >= 64:
        return 0.80
    if total_gb >= 32:
        return 0.75
    return 0.70


_MEMORY_THRESHOLD = float(
    os.environ.get("EXO_MEMORY_THRESHOLD", _default_memory_threshold())
)


class CacheSnapshot:
    """Snapshot of states at a known token position."""

    def __init__(
        self,
        states: list[
            RotatingKVCache | ArraysCache | CacheList | DeepseekV4Cache | None
        ],
        token_count: int,
    ):
        self.states = states
        self.token_count = token_count


def _detached_copy(a: mx.array) -> mx.array:
    dtype = a.dtype
    if dtype == mx.bfloat16:
        return mx.array(np.array(a.astype(mx.float32))).astype(mx.bfloat16)
    return mx.array(np.array(a))


def copy_cache_shallow(cache: KVCacheType) -> KVCacheType:
    """Fast cache copy: new wrapper objects with lazy slices of the underlying arrays.

    For each KVCache layer, nc.keys = c.keys[..., :offset, :] creates a NEW Python
    object (not an alias). MLX slice indexing does not share the Python identity of
    the original, so __setitem__ on the copy never mutates the stored cache entry.
    The slice is lazy — no data is copied until the first decode step writes new
    tokens, at which point update_and_fetch allocates a fresh buffer via concatenate.
    Because only the used portion (offset tokens) is eventually copied rather than
    the full padded buffer, this is ~100x faster than deepcopy for large caches.
    """
    new_cache: list[
        KVCache | RotatingKVCache | QuantizedKVCache | ArraysCache | CacheList
    ] = []
    for c in cache:
        if isinstance(c, KVCache):
            nc = KVCache.__new__(KVCache)
            # Slice creates a new Python object; __setitem__ on nc.keys cannot
            # reach stored.keys. Lazy: no data copy until first decode write.
            nc.keys = c.keys[..., : c.offset, :] if c.keys is not None else None
            nc.values = c.values[..., : c.offset, :] if c.values is not None else None
            nc.offset = c.offset
            new_cache.append(nc)
        elif isinstance(c, RotatingKVCache):
            nc = RotatingKVCache.__new__(RotatingKVCache)
            nc.keys = c.keys
            nc.values = c.values
            nc.offset = c.offset
            nc._idx = c._idx
            nc.max_size = c.max_size
            nc.keep = c.keep
            new_cache.append(nc)
        elif isinstance(c, QuantizedKVCache):
            nc = QuantizedKVCache.__new__(QuantizedKVCache)
            nc.keys = c.keys
            nc.values = c.values
            nc.offset = c.offset
            nc.group_size = c.group_size
            nc.bits = c.bits
            new_cache.append(nc)
        elif isinstance(c, ArraysCache):
            nc = ArraysCache.__new__(ArraysCache)
            nc.cache = list(c.cache)
            nc.left_padding = c.left_padding
            nc.lengths = c.lengths
            new_cache.append(nc)
        elif isinstance(c, CacheList):
            nc = CacheList.__new__(CacheList)
            nc.caches = tuple(copy_cache_shallow(list(c.caches)))
            new_cache.append(nc)
        else:
            # Fallback: unknown cache type — keep reference (may need deepcopy)
            new_cache.append(c)
    return new_cache


def copy_rotating_kv_cache(cache: RotatingKVCache) -> RotatingKVCache | None:
    """
    Deepcopy copies the metadata associated with an mx array.
    Specifically, it shares a shared_ptr to the underlying data and
    the mlx graph inputs of the array. This causes a memory leak for rotating
    kv cache. By creating an np array, no metadata is stored so the old cache
    can be cleaned up nicely.
    """
    if cache.keys is None or cache.values is None:
        return None
    n = min(cache.max_size, cache.keys.shape[2])
    k_slice = _detached_copy(cache.keys[..., -n:, :])
    v_slice = _detached_copy(cache.values[..., -n:, :])
    mx.eval(k_slice, v_slice)
    snap = RotatingKVCache.__new__(RotatingKVCache)
    snap.keys = k_slice
    snap.values = v_slice
    snap.offset = cache.offset
    snap._idx = n
    snap.keep = cache.keep
    snap.max_size = cache.max_size
    return snap


def _copy_arrays_cache(ac: ArraysCache) -> ArraysCache:
    entries: list[mx.array | None] = []
    for entry in ac.cache:  # type: ignore[reportUnknownMemberType]
        if entry is None:
            entries.append(None)
            continue
        assert isinstance(entry, mx.array)
        entries.append(_detached_copy(entry))
    copy = ArraysCache(len(entries))
    copy.cache = entries  # type: ignore[reportUnknownMemberType]
    return copy


def _copy_cache_list(cl: CacheList) -> CacheList:
    inners: list[object] = list(cl)  # type: ignore[reportUnknownArgumentType]
    copied: list[object] = []
    for inner in inners:
        if isinstance(inner, RotatingKVCache):
            snap = copy_rotating_kv_cache(inner)
            copied.append(snap if snap is not None else deepcopy(inner))
        elif isinstance(inner, ArraysCache):
            copied.append(_copy_arrays_cache(inner))
        else:
            copied.append(deepcopy(inner))
    return CacheList(*copied)


def _detached_copy_or_none(a: mx.array | None) -> mx.array | None:
    if a is None:
        return None
    out = _detached_copy(a)
    mx.eval(out)
    return out


def _copy_compressor_branch(b: CompressorBranch) -> CompressorBranch:
    out = CompressorBranch.__new__(CompressorBranch)
    out.buffer_kv = _detached_copy_or_none(b.buffer_kv)
    out.buffer_gate = _detached_copy_or_none(b.buffer_gate)
    out.prev_kv = _detached_copy_or_none(b.prev_kv)
    out.prev_gate = _detached_copy_or_none(b.prev_gate)
    out.pool = _detached_copy_or_none(b.pool)
    out.buffer_lengths = deepcopy(b.buffer_lengths)
    out.pool_lengths = deepcopy(b.pool_lengths)
    out.buffer_count = deepcopy(b.buffer_count)
    out._new_pool_lengths = deepcopy(b._new_pool_lengths)
    return out


def _copy_v4_cache(c: DeepseekV4Cache) -> DeepseekV4Cache:
    snap = DeepseekV4Cache.__new__(DeepseekV4Cache)

    local: RotatingKVCache = c.local
    local_snap = copy_rotating_kv_cache(local)
    if local_snap is None:
        local_snap = RotatingKVCache.__new__(RotatingKVCache)
        local_snap.keys = None
        local_snap.values = None
        local_snap.offset = local.offset
        local_snap._idx = 0
        local_snap.keep = local.keep
        local_snap.max_size = local.max_size
    snap.local = local_snap

    snap._branches = {
        key: _copy_compressor_branch(branch) for key, branch in c._branches.items()
    }
    snap._pending_lengths = deepcopy(c._pending_lengths)
    return snap


def copy_snapshot_entry(
    entry: ArraysCache | RotatingKVCache | CacheList | DeepseekV4Cache | None,
) -> ArraysCache | RotatingKVCache | CacheList | DeepseekV4Cache | None:
    match entry:
        case None:
            return None
        case RotatingKVCache():
            snap = copy_rotating_kv_cache(entry)
            return snap if snap is not None else deepcopy(entry)
        case ArraysCache():
            return _copy_arrays_cache(entry)
        case CacheList():
            return _copy_cache_list(entry)
        case DeepseekV4Cache():
            return _copy_v4_cache(entry)


def snapshot_ssm_states(cache: KVCacheType) -> CacheSnapshot:
    states: list[
        RotatingKVCache | ArraysCache | CacheList | DeepseekV4Cache | None
    ] = []
    for c in cache:
        if isinstance(c, ArraysCache):
            states.append(_copy_arrays_cache(c))
        elif isinstance(c, RotatingKVCache):
            states.append(copy_rotating_kv_cache(c))
        elif isinstance(c, CacheList) and not bool(c.is_trimmable()):  # type: ignore[reportUnknownMemberType]
            states.append(_copy_cache_list(c))
        elif isinstance(c, DeepseekV4Cache):
            states.append(_copy_v4_cache(c))
        else:
            states.append(None)
    token_count = cache_length(cache)
    return CacheSnapshot(states=states, token_count=token_count)


def _find_nearest_snapshot(
    snapshots: list[CacheSnapshot],
    target_token_count: int,
) -> CacheSnapshot | None:
    best: CacheSnapshot | None = None
    for snap in snapshots:
        if snap.token_count <= target_token_count and (
            best is None or snap.token_count > best.token_count
        ):
            best = snap
    return best


def is_non_trimmable_cache_entry(c: object) -> bool:
    """A cache entry is non-trimmable if `trim(n)` can't roll back its full
    state — meaning the prefill +2 rollback must snapshot+restore it instead.
    """
    if isinstance(c, (ArraysCache, RotatingKVCache)):
        return True
    if isinstance(c, CacheList):
        return not bool(c.is_trimmable())  # type: ignore[reportUnknownMemberType]
    return isinstance(c, DeepseekV4Cache)


def has_non_kv_caches(cache: KVCacheType) -> bool:
    """Check if a cache contains any ArraysCache (SSM) entries."""
    return any(is_non_trimmable_cache_entry(c) for c in cache)


# ---------------------------------------------------------------------------
# Context-shift helpers: rotate cached k_pe by -delta positions so a cache
# built at absolute positions [0..N-1] can be reused at positions [0..N-delta-1]
# after the front delta tokens are dropped from the context window.
# ---------------------------------------------------------------------------


def _apply_reverse_rope(
    x: mx.array,
    cos_d: mx.array,
    sin_d: mx.array,
    traditional: bool,
) -> mx.array:
    """Apply R(-delta) to every token in x.

    For traditional=True (GPT-J / DeepSeekV3 interleaved pairs):
        R^-1 pairs:  new[2i]   =  x[2i]*cos + x[2i+1]*sin
                     new[2i+1] = -x[2i]*sin + x[2i+1]*cos

    For traditional=False (GPT-NeoX split-half):
        new_left  = x_left*cos + x_right*sin
        new_right = -x_left*sin + x_right*cos

    cos_d / sin_d have shape (D//2,) — the per-frequency values for delta.
    """
    if traditional:
        x_even = x[..., 0::2]  # (..., N, D//2)
        x_odd = x[..., 1::2]
        new_even = x_even * cos_d + x_odd * sin_d
        new_odd = -x_even * sin_d + x_odd * cos_d
        # Interleave back: stack on last axis then flatten
        stacked = mx.stack([new_even, new_odd], axis=-1)  # (..., N, D//2, 2)
        return stacked.reshape(*x.shape[:-1], x.shape[-1])
    else:
        d2 = x.shape[-1] // 2
        x_left = x[..., :d2]
        x_right = x[..., d2:]
        new_left = x_left * cos_d + x_right * sin_d
        new_right = -x_left * sin_d + x_right * cos_d
        return mx.concatenate([new_left, new_right], axis=-1)


def _get_rope_config(model: "Model") -> tuple[mx.array, bool] | None:
    """Extract (rope_freqs, traditional) from the first attention layer.

    Walks model → language_model → model → layers[i].self_attn.rope
    and returns the first rope module that has a ``_freqs`` attribute.
    Returns None if no suitable rope is found.
    """
    candidate = model
    for attr in ("language_model", "model"):
        sub = getattr(candidate, attr, None)
        if sub is not None:
            candidate = sub
    layers = getattr(candidate, "layers", None) or []
    for layer in layers:
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if attn is None:
            continue
        rope = getattr(attn, "rope", None)
        if rope is None:
            continue
        freqs = getattr(rope, "_freqs", None)
        if freqs is not None:
            traditional = bool(getattr(rope, "traditional", True))
            return freqs, traditional
    return None


def shift_kv_cache(
    cache: KVCacheType,
    delta: int,
    rope_freqs: mx.array,
    traditional: bool,
) -> None:
    """Shift the logical start of every KVCache entry forward by *delta* tokens.

    After this call each KVCache holds tokens that were previously at absolute
    positions [delta .. offset-1]; they are now presented as positions
    [0 .. offset-delta-1].  The position-encoded component is re-rotated by
    R(-delta) so that attention scores remain correct.

    Cache-type semantics
    --------------------
    MLA models (DeepSeekV3 / Kimi): keys = kv_latent (no RoPE, dim 512),
        values = k_pe (RoPE baked in at absolute pos, dim 64).
        → only values need rotation; keys are sliced as-is.

    Standard models (LLaMA, Qwen …): keys = R(pos)*K (head_dim),
        values = V (no RoPE, head_dim).
        → only keys need rotation; values are sliced as-is.

    Detection: MLA when keys_dim > values_dim.

    Non-KVCache entries (RotatingKVCache, ArraysCache) are left untouched
    because they handle positions differently (sliding window / SSM state).
    """
    cos_d = mx.cos(delta * rope_freqs)  # (rope_dims//2,)
    sin_d = mx.sin(delta * rope_freqs)

    for c in cache:
        if not isinstance(c, KVCache):
            continue
        if c.keys is None or c.values is None:
            continue
        n = c.offset
        if n <= delta:
            # Nothing useful remains after shift — leave cache valid but empty.
            c.offset = 0
            continue

        k = c.keys  # physical buffer, may be larger than [0:n]
        v = c.values
        dim_k = k.shape[-1]
        dim_v = v.shape[-1]
        is_mla = dim_k > dim_v

        if is_mla:
            # keys = kv_latent: position-free, slice only
            c.keys = k[:, :, delta:n, :]
            # values = k_pe: rotate to compensate for position shift
            c.values = _apply_reverse_rope(
                v[:, :, delta:n, :], cos_d, sin_d, traditional
            )
        else:
            # keys = K with RoPE: rotate
            c.keys = _apply_reverse_rope(k[:, :, delta:n, :], cos_d, sin_d, traditional)
            # values = V: no position encoding, slice only
            c.values = v[:, :, delta:n, :]

        c.offset = n - delta

    # Force evaluation so the sliced arrays are materialised before generation
    # starts; otherwise the compute graph may hold references to the full buffers.
    to_eval = []
    for c in cache:
        if isinstance(c, KVCache) and c.keys is not None:
            to_eval.extend([c.keys, c.values])
    if to_eval:
        mx.eval(*to_eval)


def _find_suffix_shift(
    prompt_tokens: mx.array,
    cached_tokens: mx.array,
    min_match: int = 64,
) -> tuple[int, int]:
    """Find the smallest delta such that cached_tokens[delta:] is a prefix of prompt_tokens.

    Returns (delta, match_length) where match_length is the number of tokens
    from the cache that are reusable.  Returns (0, 0) if no match is found.

    Uses a two-point fingerprint to guard against pathological repetitive prefixes
    (e.g. all-spaces tokens): requires both pt[:min_match] and pt[min_match:2*min_match]
    to match before doing the full overlap check.  Falls back to single-point if the
    prompt is shorter than 2*min_match.  min_match controls the minimum overlap
    required to accept a match.
    """
    pt = np.asarray(prompt_tokens)
    ct = np.asarray(cached_tokens)

    n_cached = len(ct)
    n_prompt = len(pt)

    if n_cached == 0 or n_prompt < min_match:
        return 0, 0

    key1 = pt[:min_match]
    # Second fingerprint from offset min_match; absent when prompt is short.
    use_two_keys = n_prompt >= 2 * min_match
    key2 = pt[min_match : 2 * min_match] if use_two_keys else None

    limit = n_cached - min_match

    for delta in range(1, limit + 1):
        if not np.array_equal(ct[delta : delta + min_match], key1):
            continue
        # First fingerprint matched — check second to filter repetitive sequences.
        if use_two_keys and not np.array_equal(
            ct[delta + min_match : delta + 2 * min_match], key2
        ):
            continue
        # Both fingerprints match — measure full overlap.
        overlap = min(n_cached - delta, n_prompt)
        if np.array_equal(ct[delta : delta + overlap], pt[:overlap]):
            return delta, overlap

    return 0, 0


class KVPrefixCache:
    def __init__(self, group: mx.distributed.Group | None, model_id: str = ""):
        self.prompts: list[mx.array] = []  # mx array of tokens (ints)
        self.caches: list[KVCacheType] = []
        self._snapshots: list[list[CacheSnapshot] | None] = []
        self._media_regions: list[list["MediaRegion"]] = []
        self._last_used: list[int] = []  # monotonic counter of last access per entry
        self.prefill_tps: list[float] = []
        self._access_counter: int = 0
        self._group = group
        self._model_id = model_id
        self._rank = 0 if group is None else group.rank()
        self._save_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._disk_dir: Path | None = self._init_disk_dir()
        logger.info("KV prefix cache initialized (fresh start — all entries cleared)")
        if self._disk_dir is not None:
            self._save_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="kv_disk_save"
            )
            self._load_from_disk()

    def clear(self):
        """Clear all cached prompts and caches."""
        self.prompts.clear()
        self.caches.clear()
        self._snapshots.clear()
        self._media_regions.clear()
        self._last_used.clear()
        self.prefill_tps.clear()

    # ------------------------------------------------------------------
    # Disk-cache helpers
    # ------------------------------------------------------------------

    def _init_disk_dir(self) -> Path | None:
        """Called once at construction — creates and caches the disk cache directory."""
        d = os.environ.get("EXO_DISK_CACHE_DIR")
        if not d:
            return None
        p = Path(d).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _get_disk_dir(self) -> Path | None:
        return self._disk_dir

    def _entry_path(self, tokens: mx.array) -> Path:
        h = hashlib.sha256(np.asarray(tokens).tobytes()).hexdigest()[:16]
        return self._get_disk_dir() / f"{h}_r{self._rank}.safetensors"  # type: ignore[operator]

    def _touch_disk_entry(self, index: int) -> None:
        """Update the disk file's mtime on a cache hit so eviction is true LRU."""
        if self._disk_dir is None:
            return
        try:
            path = self._entry_path(self.prompts[index])
            if path.exists():
                os.utime(path, None)
        except OSError:
            pass

    def _save_entry_async(self, index: int) -> None:
        """Build + evaluate arrays on the main thread, then write to disk in background."""
        if self._save_executor is None:
            return
        tokens = self.prompts[index]
        cache = self.caches[index]
        prefill_tps = self.prefill_tps[index]

        if has_non_kv_caches(cache):
            return

        cdir = self._get_disk_dir()
        if cdir is None:
            return

        # Build arrays dict here (main thread) so we can eval on the GPU stream
        #
        # Cache layout:
        #   Standard models:  [KVCache, KVCache, ...]
        #   GLM-5.2 (DSA):    [CacheList(KVCache, KVCache), ...]
        # We handle both by checking for CacheList and iterating inner caches.
        # Key scheme:
        #   Flat:    layer_{i}_k / layer_{i}_v
        #   Nested:  layer_{i}_sub_{j}_k / layer_{i}_sub_{j}_v
        arrays: dict[str, mx.array] = {}
        offsets: dict[str, int] = {}
        sub_counts: dict[str, int] = {}  # number of inner caches per layer
        # Shape + dtype of every k/v we serialize, INCLUDING zero-length ones.
        # safetensors refuses to serialize an empty array, and GLM-5.2 DSA layers
        # have asymmetric inner caches — e.g. the indexer sub-cache holds keys
        # (offset > 0) but a zero-length values tensor. We therefore record a spec
        # for every array, persist only the non-empty ones, and recreate the empty
        # ones from their spec on load. (Previously this raised "Cannot serialize an
        # empty array (layer_..._sub_1_v)" and left a 0-byte file → no persistence.)
        array_specs: dict[str, list] = {}  # name -> [shape, dtype_name]

        def _record(name: str, t: mx.array) -> None:
            array_specs[name] = [list(t.shape), str(t.dtype).split(".")[-1]]
            if t.size > 0:
                arrays[name] = t

        for i, c in enumerate(cache):
            inner_caches: list[KVCache] = []
            if isinstance(c, CacheList):
                inner_caches = [ic for ic in c.caches if isinstance(ic, KVCache)]
                sub_counts[str(i)] = len(inner_caches)
            elif isinstance(c, KVCache):
                inner_caches = [c]
                sub_counts[str(i)] = 0  # 0 = flat (single KVCache, not CacheList)
            else:
                continue

            for j, ic in enumerate(inner_caches):
                if ic.keys is None:
                    continue
                n = ic.offset
                if sub_counts[str(i)] > 0:
                    kname, vname = f"layer_{i}_sub_{j}_k", f"layer_{i}_sub_{j}_v"
                else:
                    kname, vname = f"layer_{i}_k", f"layer_{i}_v"
                _record(kname, ic.keys[..., :n, :])
                if ic.values is not None:
                    _record(vname, ic.values[..., :n, :])
                offsets[str(i)] = n
        if not arrays:
            return
        arrays["tokens"] = tokens

        # Evaluate on main thread — background thread has no GPU stream
        mx.eval(*list(arrays.values()))

        meta = {
            "model_id": self._model_id,
            "rank": str(self._rank),
            "offsets": json.dumps(offsets),
            "prefill_tps": str(prefill_tps),
            "saved_at": str(time.time()),
            "n_layers": str(len(cache)),
            "sub_counts": json.dumps(sub_counts),
            "array_specs": json.dumps(array_specs),
        }
        path = self._entry_path(tokens)
        self._save_executor.submit(self._write_to_disk, arrays, meta, path)

    def _write_to_disk(
        self, arrays: dict[str, mx.array], meta: dict[str, str], path: "Path"
    ) -> None:
        """Write already-evaluated arrays to disk (background thread)."""
        try:
            t0 = time.monotonic()
            mx.save_safetensors(str(path), arrays, metadata=meta)
            dt = time.monotonic() - t0
            sz_gb = path.stat().st_size / 1e9
            n_tokens = len(arrays["tokens"])
            logger.info(
                f"KV disk cache saved: {n_tokens} tokens → {path.name} "
                f"({sz_gb:.3f} GB in {dt:.2f}s)"
            )
            self._evict_disk_if_needed()
        except Exception:
            logger.exception("KV disk cache: write failed")

    def _load_from_disk(self) -> None:
        """Load saved cache entries for this model from disk on startup.

        Entries are loaded most-recently-used first (file mtime) and only up
        to EXO_DISK_CACHE_LOAD_GB (default 50) of file bytes are materialized
        into memory. Remaining entries stay on disk untouched.
        """
        cdir = self._get_disk_dir()
        if cdir is None:
            return
        files = sorted(
            cdir.glob(f"*_r{self._rank}.safetensors"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not files:
            logger.info(f"KV disk cache: no saved entries found (dir={cdir})")
            return
        load_budget = float(os.environ.get("EXO_DISK_CACHE_LOAD_GB", "50")) * 1e9
        loaded_bytes = 0
        loaded = 0
        skipped = 0
        for path in files:
            try:
                size = path.stat().st_size
                if loaded_bytes + size > load_budget:
                    skipped += 1
                    continue
                arrays, meta = mx.load(str(path), return_metadata=True)
                if meta.get("model_id", "") != self._model_id:
                    logger.debug(
                        f"KV disk cache: skipping {path.name} "
                        f"(model_id {meta.get('model_id')!r} != {self._model_id!r})"
                    )
                    skipped += 1
                    continue
                n_layers = int(meta["n_layers"])
                prefill_tps = float(meta.get("prefill_tps", 0.0))
                tokens = arrays["tokens"]

                # sub_counts: {layer_index_str: num_inner_caches}
                # 0 = flat KVCache (standard models), >0 = CacheList (GLM-5.2 DSA)
                sub_counts: dict[str, int] = {}
                if "sub_counts" in meta:
                    try:
                        sub_counts = json.loads(meta["sub_counts"])
                    except (json.JSONDecodeError, TypeError):
                        pass

                # array_specs lets us recreate zero-length arrays that were not
                # serialized (safetensors can't store them). Maps name -> [shape, dtype].
                array_specs: dict[str, list] = {}
                if "array_specs" in meta:
                    try:
                        array_specs = json.loads(meta["array_specs"])
                    except (json.JSONDecodeError, TypeError):
                        pass

                def _materialize(name: str):
                    """Return the saved array, or recreate an empty one from its spec."""
                    if name in arrays:
                        return arrays[name]
                    spec = array_specs.get(name)
                    if spec is None:
                        return None
                    shape, dtype_name = spec
                    dt = getattr(mx, dtype_name, mx.float16)
                    return mx.zeros(tuple(shape), dtype=dt)

                def _rebuild_kv(k_key: str, v_key: str) -> "KVCache | None":
                    k = _materialize(k_key)
                    if k is None:
                        return None
                    v = _materialize(v_key)
                    if v is None:
                        # keys present, values absent (no spec): empty values matching k.
                        v = mx.zeros((*k.shape[:-2], 0, k.shape[-1]), dtype=k.dtype)
                    c = KVCache()
                    c.state = (k, v)
                    return c

                rebuilt: list[KVCache | CacheList] = []
                for i in range(n_layers):
                    sc = sub_counts.get(str(i), 0)

                    if sc > 0:
                        # CacheList layer (GLM-5.2 DSA): reconstruct from
                        # layer_{i}_sub_{j}_k / layer_{i}_sub_{j}_v
                        inner: list[KVCache] = []
                        for j in range(sc):
                            kv = _rebuild_kv(
                                f"layer_{i}_sub_{j}_k", f"layer_{i}_sub_{j}_v"
                            )
                            inner.append(kv if kv is not None else KVCache())
                        rebuilt.append(CacheList(*inner))
                    else:
                        # Flat KVCache layer (standard models)
                        kv = _rebuild_kv(f"layer_{i}_k", f"layer_{i}_v")
                        rebuilt.append(kv if kv is not None else KVCache())

                # Trigger eval so tensors are resident before first use.
                # For CacheList layers, pull keys/values from inner caches.
                to_eval = []
                for c in rebuilt:
                    if isinstance(c, CacheList):
                        for ic in c.caches:
                            if ic.keys is not None:
                                to_eval.append(ic.keys)
                            if ic.values is not None:
                                to_eval.append(ic.values)
                    elif c.keys is not None:
                        to_eval.append(c.keys)
                        to_eval.append(c.values)
                if to_eval:
                    mx.eval(*to_eval)

                self.prompts.append(tokens)
                self.caches.append(rebuilt)  # type: ignore[arg-type]
                self._snapshots.append(None)
                self._media_regions.append([])
                self.prefill_tps.append(prefill_tps)
                self._access_counter += 1
                self._last_used.append(self._access_counter)
                loaded += 1
                loaded_bytes += size
            except Exception:
                logger.exception(f"KV disk cache: failed to load {path.name}, skipping")
                skipped += 1
        logger.info(
            f"KV disk cache: loaded {loaded} entries "
            f"({loaded_bytes / 1e9:.2f} GB), skipped {skipped} (dir={cdir})"
        )

    def _evict_disk_if_needed(self) -> None:
        """Remove oldest disk cache files if total size exceeds EXO_DISK_CACHE_GB."""
        cdir = self._get_disk_dir()
        if cdir is None:
            return
        max_bytes = float(os.environ.get("EXO_DISK_CACHE_GB", "50")) * 1e9
        files = sorted(
            cdir.glob(f"*_r{self._rank}.safetensors"),
            key=lambda p: p.stat().st_mtime,
        )
        total = sum(p.stat().st_size for p in files)
        while total > max_bytes and files:
            oldest = files.pop(0)
            sz = oldest.stat().st_size
            total -= sz
            oldest.unlink(missing_ok=True)
            logger.info(
                f"KV disk cache evicted (size limit): {oldest.name} ({sz / 1e9:.3f} GB)"
            )

    def close(self) -> None:
        """Shut down the background save executor gracefully."""
        if self._save_executor is not None:
            self._save_executor.shutdown(wait=True)
            self._save_executor = None

    # ------------------------------------------------------------------
    # End disk-cache helpers
    # ------------------------------------------------------------------

    def add_kv_cache(
        self,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        ssm_snapshots: list[CacheSnapshot] | None = None,
        media_regions: list["MediaRegion"] | None = None,
        prefill_tps: float = 0.0,
    ):
        """Add a new cache entry. Evicts LRU entries if memory is high."""
        self._evict_if_needed()
        self.prompts.append(prompt_tokens)
        self.caches.append(deepcopy(cache))
        self._snapshots.append(ssm_snapshots)
        self._media_regions.append(media_regions or [])
        self.prefill_tps.append(prefill_tps)
        self._access_counter += 1
        self._last_used.append(self._access_counter)
        logger.info(f"KV cache added: {len(prompt_tokens)} tokens")
        self._save_entry_async(len(self.caches) - 1)

    def update_kv_cache(
        self,
        index: int,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        snapshots: list[CacheSnapshot] | None,
        restore_pos: int,
        media_regions: list["MediaRegion"] | None = None,
        prefill_tps: float = 0.0,
    ):
        """Update an existing cache entry in-place."""
        old_snapshots = self._snapshots[index]
        merged: list[CacheSnapshot] = []
        if old_snapshots:
            merged = [s for s in old_snapshots if s.token_count <= restore_pos]
        if snapshots:
            merged.extend(snapshots)

        self.prompts[index] = prompt_tokens
        self.caches[index] = deepcopy(cache)
        self._snapshots[index] = merged or None
        self._media_regions[index] = media_regions or []
        self.prefill_tps[index] = prefill_tps
        self._access_counter += 1
        self._last_used[index] = self._access_counter
        logger.info(f"KV cache updated (index {index}): {len(prompt_tokens)} tokens")
        self._save_entry_async(index)

    def _get_snapshot(
        self, entry_index: int, target_token_count: int
    ) -> tuple[int, CacheSnapshot | None]:
        if not has_non_kv_caches(self.caches[entry_index]):
            return target_token_count, None

        snapshots = self._snapshots[entry_index]
        if not snapshots:
            return 0, None

        snap = _find_nearest_snapshot(snapshots, target_token_count)
        if snap is not None:
            return snap.token_count, snap

        return 0, None

    def get_kv_cache(
        self,
        model: Model,
        prompt_tokens: mx.array,
        media_regions: list["MediaRegion"] | None = None,
    ) -> tuple[KVCacheType, mx.array, int | None, bool]:
        """Get KV cache for prompt, returning remaining tokens to prefill.

        Returns:
            Tuple of (cache, remaining_tokens, matched_index, is_exact) where:
            - cache: KV cache to use for generation
            - remaining_tokens: tokens that still need prefilling
            - matched_index: index of the matched entry (None if no match)
            - is_exact: True if the full prompt matched the cached entry

        For models with SSM layers (which are ArraysCache in mlx), the cache is trimmed to the
        nearest SSM snapshot position at or before the match point for correctness.
        Same for rotating KV Cache.

        Media region validation: if the token-level prefix match extends into
        a cached media region whose content_hash differs from the query's, the
        match is truncated to the start of that region.
        """
        max_length = len(prompt_tokens)
        query_regions = media_regions or []

        best_index: int | None = None
        best_length = 0
        is_exact = False

        # Find best cache match
        for i, cached_prompt in enumerate(self.prompts):
            length = get_prefix_length(prompt_tokens, cached_prompt)
            if length > 0:
                length = self._validate_media_match(
                    length,
                    self._media_regions[i],
                    query_regions,
                )
            if length >= max_length - 1:
                best_index, best_length = i, length
                is_exact = True
                break
            if length > best_length:
                best_index, best_length = i, length

        if best_index is None:
            # ---------------------------------------------------------------
            # Suffix-alignment fallback: the new prompt may be a suffix of a
            # cached prompt (e.g. Copilot dropped the oldest message so the
            # shared context starts delta tokens into the cached sequence).
            # If we find such a shift we can reuse the cache after applying
            # R(-delta) to the RoPE-encoded component.
            # Only attempt this for pure-KVCache entries (no SSM / sliding window).
            # ---------------------------------------------------------------
            rope_cfg = _get_rope_config(model)
            if rope_cfg is not None:
                for i, cached_prompt in enumerate(self.prompts):
                    if has_non_kv_caches(self.caches[i]):
                        continue
                    delta, match_len = _find_suffix_shift(prompt_tokens, cached_prompt)
                    if match_len < 1:
                        continue
                    logger.info(
                        f"KV cache suffix shift: delta={delta}, reusing {match_len} tokens "
                        f"(saves {match_len} prefill tokens)"
                    )
                    _t0 = time.monotonic()
                    prompt_cache = copy_cache_shallow(self.caches[i])
                    logger.debug(
                        f"copy_cache_shallow: {(time.monotonic() - _t0) * 1000:.2f}ms"
                    )
                    shift_kv_cache(prompt_cache, delta, *rope_cfg)
                    self._access_counter += 1
                    self._last_used[i] = self._access_counter
                    self._touch_disk_entry(i)
                    remaining = prompt_tokens[match_len:]
                    return prompt_cache, remaining, i, False

            return make_kv_cache(model), prompt_tokens, None, False

        # For exact match: trim to max_length-1 so remaining has the last token
        # For partial match: trim to best_length, remaining has suffix to prefill
        # This ensures stream_generate always has at least one token to start with
        has_ssm = has_non_kv_caches(self.caches[best_index])
        cached_length = cache_length(self.caches[best_index])
        if has_ssm:
            target = best_length
        else:
            desired = (max_length - 1) if is_exact else best_length
            target = min(cached_length, desired)
        restore_pos, restore_snap = self._get_snapshot(best_index, target)

        # No usable snapshot — need fresh cache
        if restore_snap is None and has_ssm:
            return make_kv_cache(model), prompt_tokens, None, False

        # Use shallow copy for fast cache duplication. MLX arrays use
        # copy-on-write, so the first write during generation will trigger
        # a lazy copy of only affected pages rather than copying the entire
        # cache upfront. This reduces cache-hit TTFT from ~200ms to <10ms.
        _t0 = time.monotonic()
        prompt_cache = copy_cache_shallow(self.caches[best_index])
        logger.debug(f"copy_cache_shallow: {(time.monotonic() - _t0) * 1000:.2f}ms")
        tokens_to_trim = cached_length - restore_pos
        if tokens_to_trim > 0:
            trim_cache(prompt_cache, tokens_to_trim, restore_snap)
            # Reset cache offset to match trimmed length
            for c in prompt_cache:
                if isinstance(c, (ArraysCache, RotatingKVCache)):
                    continue
                if isinstance(c, DeepseekV4Cache):
                    continue
                if hasattr(c, "offset"):
                    c.offset = restore_pos

        self._access_counter += 1
        self._last_used[best_index] = self._access_counter
        self._touch_disk_entry(best_index)
        remaining = prompt_tokens[restore_pos:]

        return prompt_cache, remaining, best_index, is_exact

    @staticmethod
    def _validate_media_match(
        match_length: int,
        cached_regions: list["MediaRegion"],
        query_regions: list["MediaRegion"],
    ) -> int:
        if not cached_regions:
            return match_length

        query_by_start: dict[int, "MediaRegion"] = {
            r.start_pos: r for r in query_regions
        }

        for cached_r in cached_regions:
            if cached_r.start_pos >= match_length:
                break
            query_r = query_by_start.get(cached_r.start_pos)
            if query_r is None:
                continue
            if query_r.content_hash != cached_r.content_hash:
                logger.info(
                    f"Media region mismatch at pos {cached_r.start_pos}: "
                    f"cached={cached_r.content_hash[:12]}... "
                    f"query={query_r.content_hash[:12]}... — "
                    f"truncating match from {match_length} to {cached_r.start_pos}"
                )
                match_length = cached_r.start_pos
                break

        return match_length

    def _evict_if_needed(self):
        """Evict least recently used entries while memory usage is high."""
        if len(self.caches) == 0:
            return

        evicted_any = False
        # Evict LRU entries until below threshold
        while (
            len(self.caches) > 0
            and self.get_memory_used_percentage() > _MEMORY_THRESHOLD
        ):
            lru_index = self._last_used.index(min(self._last_used))
            evicted_tokens = len(self.prompts[lru_index])
            self.prompts.pop(lru_index)
            self.caches.pop(lru_index)
            self._snapshots.pop(lru_index)
            self._media_regions.pop(lru_index)
            self._last_used.pop(lru_index)
            self.prefill_tps.pop(lru_index)

            evicted_any = True
            logger.info(
                f"KV cache evicted LRU entry ({evicted_tokens} tokens) due to memory usage"
            )

        if evicted_any:
            gc.collect()
            mx.clear_cache()

    def get_memory_used_percentage(self) -> float:
        local_pressure: float = get_memory_used_percentage()

        if self._group is None:
            return local_pressure

        all_pressure = mx.distributed.all_gather(
            mx.array([local_pressure], dtype=mx.float32),
            group=self._group,
        )
        # .item() evals.
        max_pressure = float(mx.max(all_pressure).item())
        return max_pressure


def trim_cache(
    cache: KVCacheType,
    num_tokens: int,
    snapshot: CacheSnapshot | None = None,
) -> None:
    for i, c in enumerate(cache):
        non_trimmable = isinstance(c, (ArraysCache, RotatingKVCache)) or (
            isinstance(c, CacheList) and not bool(c.is_trimmable())  # type: ignore[reportUnknownMemberType]
        )
        if non_trimmable:
            if snapshot is not None and snapshot.states[i] is not None:
                restored = copy_snapshot_entry(snapshot.states[i])
                if restored is not None:
                    cache[i] = restored  # type: ignore
            elif isinstance(c, (ArraysCache, RotatingKVCache)):
                c.state = [None] * len(c.state)
                if isinstance(c, RotatingKVCache):
                    c.offset = 0
                    c._idx = 0
            else:
                # CacheList without a snapshot — zero each inner cache's state
                for inner in c:  # type: ignore[reportUnknownVariableType]
                    if isinstance(inner, (ArraysCache, RotatingKVCache)):
                        inner.state = [None] * len(inner.state)
                        if isinstance(inner, RotatingKVCache):
                            inner.offset = 0
                            inner._idx = 0
        else:
            c.trim(num_tokens)


def encode_prompt(tokenizer: TokenizerWrapper, prompt: str) -> mx.array:
    """Encode a prompt string to token array.

    For chat-templated prompts (which have their own structure markers like
    <|im_user|>, <|im_middle|>, etc.), we should NOT add BOS/EOS tokens as
    that would corrupt the prompt structure.
    """
    # Chat templates define their own structure - don't add BOS/EOS
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    return mx.array(prompt_tokens)


def _entry_length(
    c: KVCache
    | RotatingKVCache
    | QuantizedKVCache
    | ArraysCache
    | CacheList
    | DeepseekV4Cache,
) -> int:
    # Use .offset attribute which KVCache types have (len() not implemented in older QuantizedKVCache).
    if hasattr(c, "offset"):
        return c.offset
    # For CacheList
    if hasattr(c, "size"):
        return int(c.size())  # type: ignore
    return 0


def cache_length(cache: KVCacheType) -> int:
    """Get the number of tokens in a KV cache."""
    return max((_entry_length(c) for c in cache), default=0)


def get_prefix_length(prompt: mx.array, cached_prompt: mx.array) -> int:
    """Find the length of the common prefix between two token arrays."""
    n = min(int(prompt.shape[0]), int(cached_prompt.shape[0]))
    if n == 0:
        return 0

    equal = mx.equal(prompt[:n], cached_prompt[:n]).astype(mx.int32)
    prefix_mask = mx.cumprod(equal)  # stays 1 until first mismatch, then 0 forever
    return int(mx.sum(prefix_mask).item())


def get_available_memory() -> Memory:
    mem: int = psutil.virtual_memory().available
    return Memory.from_bytes(mem)


def get_memory_used_percentage() -> float:
    mem = psutil.virtual_memory()
    # percent is 0-100
    return float(mem.percent / 100)


def make_kv_cache(
    model: Model, max_kv_size: int | None = None, keep: int = 0
) -> KVCacheType:
    assert hasattr(model, "layers")

    if hasattr(model, "make_cache"):
        logger.info("Using MLX LM's make cache")
        return model.make_cache()  # type: ignore

    if max_kv_size is None:
        if KV_CACHE_BITS is None:
            logger.info("Using default KV cache")
            return [KVCache() for _ in model.layers]
        else:
            logger.info("Using quantized KV cache")
            return [
                QuantizedKVCache(group_size=CACHE_GROUP_SIZE, bits=KV_CACHE_BITS)
                for _ in model.layers
            ]
    else:
        logger.info(f"Using rotating KV cache with {max_kv_size=} with {keep=}")
        return [RotatingKVCache(max_size=max_kv_size, keep=keep) for _ in model.layers]
