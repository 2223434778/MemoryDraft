"""Request-local, growable KV storage with the DynamicCache logical frontier.

Only full-attention, single-request inference is supported. Views expose exactly
the valid prefix; unused capacity never participates in attention. Native crop
and the existing tree suffix compaction continue to operate on these views.
"""
from collections import Counter

import torch
from transformers import DynamicCache


class BufferedDynamicCache(DynamicCache):
    def __init__(self, *, block_tokens=256):
        if type(block_tokens) is not int or block_tokens < 1:
            raise ValueError('Cache block size must be a positive integer')
        super().__init__()
        self.block_tokens = block_tokens
        self._buffers = {}
        self.stats = Counter()

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if (type(layer_idx) is not int or layer_idx < 0 or key_states.ndim != 4
                or key_states.shape != value_states.shape or key_states.shape[0] != 1
                or key_states.dtype != value_states.dtype or key_states.device != value_states.device):
            raise ValueError('Buffered cache requires matching single-request KV tensors')
        length = self.get_seq_length(layer_idx)
        added = key_states.shape[-2]
        needed = length+added
        previous = self._buffers.get(layer_idx)
        if previous is not None:
            old = previous[0]
            if (old.shape[:2] != key_states.shape[:2] or old.shape[-1] != key_states.shape[-1]
                    or old.dtype != key_states.dtype or old.device != key_states.device):
                raise ValueError('Buffered cache KV layout changed within a request')
        if previous is None:
            # Let the installed HF version initialize its native layer/list API.
            super().update(key_states[..., :0, :], value_states[..., :0, :], layer_idx, cache_kwargs)
        if previous is None or needed > previous[0].shape[-2]:
            capacity = max(self.block_tokens, needed, 2*previous[0].shape[-2] if previous else 0)
            capacity = (capacity+self.block_tokens-1)//self.block_tokens*self.block_tokens
            shape = (*key_states.shape[:2], capacity, key_states.shape[-1])
            buffers = tuple(torch.empty(shape, dtype=key_states.dtype, device=key_states.device) for _ in range(2))
            if previous is not None and length:
                for new, old in zip(buffers, previous):
                    new[..., :length, :].copy_(old[..., :length, :])
                self.stats['history_copy_bytes'] += 2*key_states.element_size()*key_states.shape[1]*length*key_states.shape[-1]
            self._buffers[layer_idx] = previous = buffers
            self.stats['allocations'] += 1
        for buffer, new in zip(previous, (key_states, value_states)):
            buffer[..., length:needed, :].copy_(new)
        keys, values = (buffer[..., :needed, :] for buffer in previous)
        if hasattr(self, 'layers'):
            self.layers[layer_idx].keys, self.layers[layer_idx].values = keys, values
        else:
            self.key_cache[layer_idx], self.value_cache[layer_idx] = keys, values
            if layer_idx == 0:
                self._seen_tokens += added
        self.stats['updates'] += 1
        return keys, values

    def summary(self):
        return dict(self.stats, block_tokens=self.block_tokens,
            capacity_bytes=sum(t.numel()*t.element_size() for pair in self._buffers.values() for t in pair))
