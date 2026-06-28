# type: ignore
"""Tests for KV disk cache save/load with CacheList layers (GLM-5.2 DSA).

Verifies that _save_entry_async correctly serializes CacheList-wrapped KVCache
entries and _load_from_disk correctly reconstructs them.
"""
import hashlib
import json
import os
import tempfile

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import CacheList, KVCache

from exo.worker.engines.mlx.cache import KVPrefixCache


def _make_glmlayer_cache(n_layers: int = 3, n_tokens: int = 8, n_heads: int = 2, head_dim: int = 4):
    """Build a cache mimicking GLM-5.2 DSA: list of CacheList(KVCache, KVCache)."""
    cache = []
    for _ in range(n_layers):
        k1 = mx.random.uniform(shape=(1, n_heads, n_tokens, head_dim)).astype(mx.float16)
        v1 = mx.random.uniform(shape=(1, n_heads, n_tokens, head_dim)).astype(mx.float16)
        kv1 = KVCache()
        kv1.state = (k1, v1)

        k2 = mx.random.uniform(shape=(1, n_heads, n_tokens, head_dim)).astype(mx.float16)
        v2 = mx.random.uniform(shape=(1, n_heads, n_tokens, head_dim)).astype(mx.float16)
        kv2 = KVCache()
        kv2.state = (k2, v2)

        cache.append(CacheList(kv1, kv2))
    return cache


def _make_flat_cache(n_layers: int = 3, n_tokens: int = 8, n_heads: int = 2, head_dim: int = 4):
    """Build a standard flat cache: list of KVCache."""
    cache = []
    for _ in range(n_layers):
        k = mx.random.uniform(shape=(1, n_heads, n_tokens, head_dim)).astype(mx.float16)
        v = mx.random.uniform(shape=(1, n_heads, n_tokens, head_dim)).astype(mx.float16)
        kv = KVCache()
        kv.state = (k, v)
        cache.append(kv)
    return cache


class TestDiskCacheCacheList:
    """Test disk cache save/load with CacheList (GLM-5.2 DSA layout)."""

    def test_save_writes_file_with_cachelist(self, tmp_path):
        """_save_entry_async should write a .safetensors file when cache has CacheList layers."""
        cache_dir = tmp_path / "kv_disk"
        os.environ["EXO_DISK_CACHE_DIR"] = str(cache_dir)
        os.environ["EXO_DISK_CACHE_GB"] = "200"

        kvc = KVPrefixCache(group=None, model_id="test/glm-dsa")
        tokens = mx.array([1, 2, 3, 4, 5, 6, 7, 8])
        cache = _make_glmlayer_cache(n_layers=3, n_tokens=8)

        kvc.prompts.append(tokens)
        kvc.caches.append(cache)
        kvc.prefill_tps.append(100.0)
        kvc._save_entry_async(0)

        # Wait for background write
        kvc.close()

        files = list(cache_dir.glob("*.safetensors"))
        assert len(files) == 1, f"Expected 1 disk cache file, got {len(files)}"
        assert files[0].stat().st_size > 0, "Disk cache file is empty"

    def test_save_then_load_roundtrip_cachelist(self, tmp_path):
        """Saved CacheList cache should load back with matching keys/values."""
        cache_dir = tmp_path / "kv_disk"
        os.environ["EXO_DISK_CACHE_DIR"] = str(cache_dir)
        os.environ["EXO_DISK_CACHE_GB"] = "200"
        os.environ["EXO_DISK_CACHE_LOAD_GB"] = "200"

        # Save
        kvc = KVPrefixCache(group=None, model_id="test/glm-dsa")
        tokens = mx.array([1, 2, 3, 4, 5, 6, 7, 8])
        cache = _make_glmlayer_cache(n_layers=3, n_tokens=8, n_heads=2, head_dim=4)
        original_keys = [
            (mx.array(ic.keys), mx.array(ic.values))
            for c in cache
            for ic in c.caches
        ]

        kvc.prompts.append(tokens)
        kvc.caches.append(cache)
        kvc.prefill_tps.append(100.0)
        kvc._save_entry_async(0)
        kvc.close()

        # Load in a fresh instance
        kvc2 = KVPrefixCache(group=None, model_id="test/glm-dsa")
        assert len(kvc2.caches) == 1, f"Expected 1 loaded entry, got {len(kvc2.caches)}"
        loaded = kvc2.caches[0]
        assert len(loaded) == 3, f"Expected 3 layers, got {len(loaded)}"

        for i, layer in enumerate(loaded):
            assert isinstance(layer, CacheList), f"Layer {i} should be CacheList"
            assert len(layer.caches) == 2, f"Layer {i} should have 2 inner caches"

        # Verify keys/values match
        loaded_keys = [
            (ic.keys, ic.values)
            for c in loaded
            for ic in c.caches
        ]
        assert len(loaded_keys) == len(original_keys)
        for (ok, ov), (lk, lv) in zip(original_keys, loaded_keys):
            assert mx.allclose(ok, lk, equal_nan=True), "Loaded keys don't match saved keys"
            assert mx.allclose(ov, lv, equal_nan=True), "Loaded values don't match saved values"

        kvc2.close()

    def test_save_then_load_roundtrip_flat(self, tmp_path):
        """Standard flat KVCache layout should still work (backward compat)."""
        cache_dir = tmp_path / "kv_disk"
        os.environ["EXO_DISK_CACHE_DIR"] = str(cache_dir)
        os.environ["EXO_DISK_CACHE_GB"] = "200"
        os.environ["EXO_DISK_CACHE_LOAD_GB"] = "200"

        kvc = KVPrefixCache(group=None, model_id="test/standard")
        tokens = mx.array([10, 20, 30, 40])
        cache = _make_flat_cache(n_layers=2, n_tokens=4, n_heads=2, head_dim=4)
        original_keys = [(mx.array(c.keys), mx.array(c.values)) for c in cache]

        kvc.prompts.append(tokens)
        kvc.caches.append(cache)
        kvc.prefill_tps.append(50.0)
        kvc._save_entry_async(0)
        kvc.close()

        kvc2 = KVPrefixCache(group=None, model_id="test/standard")
        assert len(kvc2.caches) == 1
        loaded = kvc2.caches[0]
        assert len(loaded) == 2
        for i, layer in enumerate(loaded):
            assert isinstance(layer, KVCache), f"Layer {i} should be KVCache, got {type(layer).__name__}"

        for (ok, ov), lc in zip(original_keys, loaded):
            assert mx.allclose(ok, lc.keys, equal_nan=True)
            assert mx.allclose(ov, lc.values, equal_nan=True)

        kvc2.close()

    def test_save_skips_when_no_disk_dir(self, tmp_path):
        """Without EXO_DISK_CACHE_DIR, save should be a no-op."""
        os.environ.pop("EXO_DISK_CACHE_DIR", None)
        kvc = KVPrefixCache(group=None, model_id="test/no-disk")
        tokens = mx.array([1, 2, 3])
        cache = _make_glmlayer_cache(n_layers=2, n_tokens=3)

        kvc.prompts.append(tokens)
        kvc.caches.append(cache)
        kvc.prefill_tps.append(0.0)
        # Should not raise
        kvc._save_entry_async(0)
        kvc.close()


class TestEmptyInnerCache:
    """Regression: GLM-5.2 DSA layers can have an empty inner sub-cache (offset 0).

    safetensors refuses to serialize a zero-length array, so an unguarded save
    fails for the WHOLE entry and leaves a 0-byte file on disk (observed live as
    'Cannot serialize an empty array (layer_66_sub_1_v)'). The save path must skip
    empty inner caches and the load path must reconstruct them as empty.
    """

    def _make_cache_with_empty_sub(self, n_layers=3, n_tokens=8, n_heads=2, head_dim=4):
        """CacheList(KVCache[populated], KVCache[indexer: keys present, values empty]).

        Mirrors GLM-5.2 DSA, where the indexer sub-cache holds keys (offset > 0) but
        a zero-length values tensor — the exact shape that produced the live
        'Cannot serialize an empty array (layer_66_sub_1_v)' failure.
        """
        cache = []
        for _ in range(n_layers):
            k = mx.random.uniform(shape=(1, n_heads, n_tokens, head_dim)).astype(mx.float16)
            v = mx.random.uniform(shape=(1, n_heads, n_tokens, head_dim)).astype(mx.float16)
            full = KVCache()
            full.state = (k, v)

            indexer = KVCache()
            # Keys populated, values zero-length along the token axis.
            ik = mx.random.uniform(shape=(1, n_heads, n_tokens, head_dim)).astype(mx.float16)
            iv = mx.zeros((1, n_heads, 0, head_dim), dtype=mx.float16)
            indexer.state = (ik, iv)
            assert indexer.offset == n_tokens

            cache.append(CacheList(full, indexer))
        return cache

    def test_save_with_empty_sub_writes_nonzero_file(self, tmp_path):
        cache_dir = tmp_path / "kv_disk"
        os.environ["EXO_DISK_CACHE_DIR"] = str(cache_dir)
        os.environ["EXO_DISK_CACHE_GB"] = "200"
        os.environ["EXO_DISK_CACHE_LOAD_GB"] = "200"

        kvc = KVPrefixCache(group=None, model_id="test/glm-dsa")
        tokens = mx.array([1, 2, 3, 4, 5, 6, 7, 8])
        cache = self._make_cache_with_empty_sub(n_layers=3, n_tokens=8)
        kvc.prompts.append(tokens)
        kvc.caches.append(cache)
        kvc.prefill_tps.append(100.0)
        kvc._save_entry_async(0)
        kvc.close()

        files = list(cache_dir.glob("*.safetensors"))
        assert len(files) == 1
        # The bug left a 0-byte file; the fix must produce a real, non-empty file.
        assert files[0].stat().st_size > 0, "Empty inner cache broke the write (0-byte file)"

        # And it must reload with the empty sub reconstructed (empty / offset 0).
        kvc2 = KVPrefixCache(group=None, model_id="test/glm-dsa")
        assert len(kvc2.caches) == 1
        loaded = kvc2.caches[0]
        assert len(loaded) == 3
        for layer in loaded:
            assert isinstance(layer, CacheList)
            assert len(layer.caches) == 2
            full, indexer = layer.caches
            assert full.keys is not None and full.offset == 8
            # Indexer reconstructed: keys length 8, values zero-length (asymmetric).
            assert indexer.keys is not None and indexer.keys.shape[2] == 8
            assert indexer.values is not None and indexer.values.shape[2] == 0
        kvc2.close()


class TestEntryPathAndModelIdSkip:
    """Non-slow coverage for _entry_path and the model_id-mismatch load skip."""

    def test_entry_path_deterministic_and_rank_suffix(self, tmp_path):
        """_entry_path is a deterministic sha256[:16] of the token bytes with a _r{rank} suffix."""
        cache_dir = tmp_path / "kv_disk"
        os.environ["EXO_DISK_CACHE_DIR"] = str(cache_dir)

        kvc = KVPrefixCache(group=None, model_id="test/model")
        try:
            tokens = mx.array([1, 2, 3, 4, 5])
            p1 = kvc._entry_path(tokens)
            p2 = kvc._entry_path(mx.array([1, 2, 3, 4, 5]))

            # Deterministic: same token values -> identical path.
            assert p1 == p2

            # rank 0 (group=None) -> _r0 suffix, .safetensors extension.
            assert p1.name.endswith("_r0.safetensors")
            assert p1.parent == cache_dir

            # Hash matches the documented scheme (sha256 of raw bytes, first 16 hex chars).
            expected_hash = hashlib.sha256(np.asarray(tokens).tobytes()).hexdigest()[:16]
            assert p1.name == f"{expected_hash}_r0.safetensors"

            # Different tokens -> different path.
            p3 = kvc._entry_path(mx.array([9, 9, 9, 9, 9]))
            assert p3 != p1
        finally:
            kvc.close()

    def test_load_skips_mismatched_model_id(self, tmp_path):
        """Entries saved under a different model_id must be skipped on load (no OOM/garbage)."""
        cache_dir = tmp_path / "kv_disk"
        os.environ["EXO_DISK_CACHE_DIR"] = str(cache_dir)
        os.environ["EXO_DISK_CACHE_GB"] = "200"
        os.environ["EXO_DISK_CACHE_LOAD_GB"] = "200"

        # Save one entry under model "A".
        saver = KVPrefixCache(group=None, model_id="org/model-A")
        tokens = mx.array([1, 2, 3, 4, 5, 6, 7, 8])
        cache = _make_glmlayer_cache(n_layers=3, n_tokens=8)
        saver.prompts.append(tokens)
        saver.caches.append(cache)
        saver.prefill_tps.append(100.0)
        saver._save_entry_async(0)
        saver.close()

        files = list(cache_dir.glob("*.safetensors"))
        assert len(files) == 1, "Setup failed: entry not written"

        # A loader for a DIFFERENT model must skip the on-disk entry.
        mismatched = KVPrefixCache(group=None, model_id="org/model-B")
        assert len(mismatched.caches) == 0, "Entry from a different model_id was not skipped"
        mismatched.close()

        # Positive control: the SAME model_id loads it (proves the skip was model_id-driven,
        # not a file/path problem).
        matched = KVPrefixCache(group=None, model_id="org/model-A")
        assert len(matched.caches) == 1, "Matching model_id entry should load"
        matched.close()