import pytest
import torch
from transformers import DynamicCache

from memorydraft.backends.buffered_cache import BufferedDynamicCache
from memorydraft.backends.draft_tree import commit_tree_cache


def test_growth_crop_rewrite_and_tree_compaction_match_dynamic():
    caches = [DynamicCache(), BufferedDynamicCache(block_tokens=4)]
    generator = torch.Generator().manual_seed(30)
    for added, crop, path in [(3, None, None), (4, 3, [0, 2]), (3, 6, None), (2, -1, None)]:
        for layer in range(2):
            key, value = [torch.randn(1, 2, added, 4, generator=generator) for _ in range(2)]
            results = [c.update(key, value, layer) for c in caches]
            assert all(torch.equal(a, b) for a, b in zip(*results))
        if path is not None:
            for c in caches:
                commit_tree_cache(c, crop, path)
        elif crop is not None:
            for c in caches:
                c.crop(crop)
        assert caches[0].get_seq_length() == caches[1].get_seq_length()
    stats = caches[1].summary()
    assert stats['allocations'] < stats['updates']
    assert stats['capacity_bytes'] > 0


def test_storage_is_reused_and_cropped_tail_is_not_exposed():
    cache = BufferedDynamicCache(block_tokens=8)
    key = torch.arange(12.).reshape(1, 1, 3, 4)
    first, _ = cache.update(key, key, 0)
    ptr = first.data_ptr()
    cache.crop(1)
    second, _ = cache.update(torch.full((1, 1, 2, 4), 99.), torch.zeros(1, 1, 2, 4), 0)
    assert second.data_ptr() == ptr and second.shape[-2] == 3
    assert second[0, 0, 1:].tolist() == [[99.]*4]*2
    assert torch.equal(first[..., :1, :], second[..., :1, :])
    with pytest.raises(ValueError, match='layout'):
        cache.update(torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 1, 4), 0)


@pytest.mark.parametrize('bad', [0, -1, True])
def test_invalid_capacity(bad):
    with pytest.raises(ValueError, match='positive'):
        BufferedDynamicCache(block_tokens=bad)
