"""Tests for the shallow copy cache optimization.

These tests verify that copy_cache_shallow() correctly creates new wrapper
objects while sharing underlying mx.array memory, ensuring cache isolation
via MLX's copy-on-write behavior.
"""

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)

from exo.worker.engines.mlx.cache import copy_cache_shallow


class TestCopyCacheShallow:
    """Test the fast cache copy function."""

    def test_kv_cache_shallow_copy(self):
        """KVCache: new wrapper, shared arrays, COW on write."""
        original = KVCache()
        # Simulate a pre-filled cache
        original.keys = mx.zeros((1, 4, 256, 64))
        original.values = mx.zeros((1, 4, 256, 64))
        original.offset = 128

        copied = copy_cache_shallow([original])[0]

        # Should be a new wrapper object
        assert copied is not original
        assert isinstance(copied, KVCache)

        # Arrays are new Python objects (slices), NOT the same object.
        # This is intentional: __setitem__ on a slice does not mutate the
        # original array, preventing corruption of the stored cache entry.
        assert copied.keys is not original.keys
        assert copied.values is not original.values

        # But they hold the same data values
        import numpy as np

        assert np.array_equal(
            np.array(copied.keys), np.array(original.keys[..., : original.offset, :])
        )

        # Shape reflects only the used portion (offset tokens, not padded buffer)
        assert copied.keys.shape[2] == original.offset

        # Metadata should match
        assert copied.offset == original.offset

        # Modifying copied metadata should NOT affect original
        copied.offset = 200
        assert original.offset == 128
        assert copied.offset == 200

    def test_rotating_kv_cache_shallow_copy(self):
        """RotatingKVCache: new wrapper, shared arrays, COW on write."""
        original = RotatingKVCache(max_size=1024, keep=4)
        original.keys = mx.zeros((1, 4, 1024, 64))
        original.values = mx.zeros((1, 4, 1024, 64))
        original.offset = 500
        original._idx = 500

        copied = copy_cache_shallow([original])[0]

        assert copied is not original
        assert isinstance(copied, RotatingKVCache)
        assert copied.keys is original.keys
        assert copied.values is original.values
        assert copied.offset == 500
        assert copied._idx == 500
        assert copied.max_size == 1024
        assert copied.keep == 4

        # Modify metadata
        copied.offset = 600
        copied._idx = 600
        assert original.offset == 500
        assert original._idx == 500

    def test_quantized_kv_cache_shallow_copy(self):
        """QuantizedKVCache: new wrapper, shared arrays, COW on write."""
        original = QuantizedKVCache(group_size=64, bits=4)
        # Quantized cache stores tuples of (quantized_data, scales, biases)
        original.keys = (
            mx.zeros((1, 4, 256, 8), dtype=mx.uint32),
            mx.zeros((1, 4, 256, 1)),
            mx.zeros((1, 4, 256, 1)),
        )
        original.values = (
            mx.zeros((1, 4, 256, 8), dtype=mx.uint32),
            mx.zeros((1, 4, 256, 1)),
            mx.zeros((1, 4, 256, 1)),
        )
        original.offset = 100
        original.group_size = 64
        original.bits = 4

        copied = copy_cache_shallow([original])[0]

        assert copied is not original
        assert isinstance(copied, QuantizedKVCache)
        assert copied.keys is original.keys
        assert copied.values is original.values
        assert copied.offset == 100
        assert copied.group_size == 64
        assert copied.bits == 4

        copied.offset = 200
        assert original.offset == 100

    def test_arrays_cache_shallow_copy(self):
        """ArraysCache: new wrapper, list copy, COW on write."""
        original = ArraysCache(3)
        original.cache = [mx.zeros((1, 64)), mx.ones((1, 64)), mx.zeros((1, 64))]
        original.left_padding = mx.array([0])
        original.lengths = mx.array([10])

        copied = copy_cache_shallow([original])[0]

        assert copied is not original
        assert isinstance(copied, ArraysCache)

        # The list should be a copy (different list object)
        assert copied.cache is not original.cache

        # But the array elements should be shared
        assert copied.cache[0] is original.cache[0]
        assert copied.cache[1] is original.cache[1]

        # Modifying the list should not affect original
        copied.cache.append(mx.zeros((1, 64)))
        assert len(original.cache) == 3
        assert len(copied.cache) == 4

    def test_cache_list_shallow_copy(self):
        """CacheList: recursively shallow-copies nested caches."""
        inner1 = KVCache()
        inner1.keys = mx.zeros((1, 4, 256, 64))
        inner1.values = mx.zeros((1, 4, 256, 64))
        inner1.offset = 50

        inner2 = KVCache()
        inner2.keys = mx.zeros((1, 4, 256, 64))
        inner2.values = mx.zeros((1, 4, 256, 64))
        inner2.offset = 75

        original = CacheList(inner1, inner2)

        copied = copy_cache_shallow([original])[0]

        assert copied is not original
        assert isinstance(copied, CacheList)

        # The caches tuple should be new
        assert copied.caches is not original.caches

        # Inner KVCache arrays are slices (new objects), not shared references
        assert copied.caches[0].keys is not original.caches[0].keys
        assert copied.caches[1].keys is not original.caches[1].keys

        # Metadata should be independent
        copied.caches[0].offset = 999
        assert original.caches[0].offset == 50

    def test_mixed_cache_types(self):
        """A cache with multiple types is handled correctly."""
        kv = KVCache()
        kv.keys = mx.zeros((1, 4, 256, 64))
        kv.values = mx.zeros((1, 4, 256, 64))
        kv.offset = 10

        rotating = RotatingKVCache(max_size=512)
        rotating.keys = mx.zeros((1, 4, 512, 64))
        rotating.values = mx.zeros((1, 4, 512, 64))
        rotating.offset = 20

        arrays = ArraysCache(2)
        arrays.cache = [mx.zeros((1, 64)), mx.ones((1, 64))]

        original = [kv, rotating, arrays]
        copied = copy_cache_shallow(original)

        assert len(copied) == 3
        assert copied[0] is not kv
        assert copied[1] is not rotating
        assert copied[2] is not arrays

        # KVCache arrays are slices (new objects, not shared)
        assert copied[0].keys is not kv.keys
        # RotatingKVCache and ArraysCache still share arrays
        assert copied[1].keys is rotating.keys
        assert copied[2].cache[0] is arrays.cache[0]

    def test_cache_isolation_after_metadata_modification(self):
        """Verify that modifying copied cache metadata doesn't affect original."""
        original = KVCache()
        original.keys = mx.zeros((1, 4, 256, 64))
        original.values = mx.zeros((1, 4, 256, 64))
        original.offset = 100

        copied = copy_cache_shallow([original])[0]

        # Modify all metadata fields
        copied.offset = 200

        # Original should be unchanged
        assert original.offset == 100

    def test_empty_cache(self):
        """An empty cache (no prefill) should be handled gracefully."""
        kv = KVCache()
        # keys and values are None by default
        assert kv.keys is None
        assert kv.values is None
        assert kv.offset == 0

        copied = copy_cache_shallow([kv])[0]

        assert copied is not kv
        assert copied.keys is None
        assert copied.values is None
        assert copied.offset == 0


class TestCacheIsolationScenario:
    """Test the real-world scenario: two requests sharing a prefix cache."""

    def test_two_requests_same_prefix_no_corruption(self):
        """Simulate two requests using the same cached prefix.

        This verifies that after Request A generates tokens (modifying its cache),
        Request B's cache still has the original prefix state.
        """
        # Create a "stored" prefix cache (simulating KVPrefixCache entry)
        stored_cache = KVCache()
        stored_cache.keys = mx.zeros((1, 4, 256, 64))
        stored_cache.values = mx.zeros((1, 4, 256, 64))
        stored_cache.offset = 100  # 100 tokens prefilled

        # Request A gets a shallow copy
        cache_a = copy_cache_shallow([stored_cache])[0]

        # Request B gets another shallow copy
        cache_b = copy_cache_shallow([stored_cache])[0]

        # Simulate Request A generating 5 tokens
        # (In real generation, this would call model() which updates the cache)
        cache_a.offset = 105

        # Request B should still see the original 100 tokens
        assert cache_b.offset == 100
        assert stored_cache.offset == 100

        # cache_b.keys is a slice (new Python object), not the same reference,
        # but it represents the same underlying data.
        assert cache_b.keys is not stored_cache.keys

    def test_update_and_fetch_does_not_corrupt_stored_cache(self):
        """Verify that calling update_and_fetch on a shallow copy never writes
        into the stored cache's buffer.

        This is the critical COW property: copy_cache_shallow() slices the keys/values
        to exactly [offset] tokens.  KVCache.update_and_fetch() sees a buffer that
        is already full (shape[-2] == offset) and therefore takes the realloc path
        (mx.concatenate), producing a brand-new Python object.  The stored cache's
        original data at positions [:offset] must remain untouched.
        """
        FILL_VALUE = 1.0
        NEW_VALUE = 99.0

        stored_cache = KVCache()
        # Pre-fill the first 100 token slots with FILL_VALUE
        stored_cache.keys = mx.full((1, 4, 256, 64), FILL_VALUE)
        stored_cache.values = mx.full((1, 4, 256, 64), FILL_VALUE)
        stored_cache.offset = 100

        # Take a shallow copy
        copy = copy_cache_shallow([stored_cache])[0]
        assert copy.keys is not stored_cache.keys, (
            "slice must produce new Python object"
        )
        assert copy.keys.shape[2] == 100, "slice size must equal offset"

        # Call update_and_fetch on the copy — writes NEW_VALUE at token position 100
        new_k = mx.full((1, 4, 1, 64), NEW_VALUE)
        new_v = mx.full((1, 4, 1, 64), NEW_VALUE)
        copy.update_and_fetch(new_k, new_v)
        mx.eval(copy.keys)

        # Stored cache's first 100 positions must be untouched
        stored_data = np.array(stored_cache.keys[0, 0, :100, 0])
        assert np.all(stored_data == FILL_VALUE), (
            f"stored cache corrupted after update_and_fetch on copy: "
            f"min={stored_data.min()}, max={stored_data.max()}"
        )

    def test_trim_after_shallow_copy(self):
        """Verify that trim_cache works correctly on a shallow-copied cache."""
        from exo.worker.engines.mlx.cache import trim_cache

        original = KVCache()
        original.keys = mx.zeros((1, 4, 256, 64))
        original.values = mx.zeros((1, 4, 256, 64))
        original.offset = 200

        copied = copy_cache_shallow([original])[0]

        # Trim 50 tokens from the copied cache
        trim_cache([copied], 50)

        # Copied should be trimmed
        assert copied.offset == 150

        # Original should be unchanged
        assert original.offset == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
