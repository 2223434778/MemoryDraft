"""Bounded single-edit trees from parallel draft marginals; strict greedy only.

Research implementation, inspired by BASTION/OPT-Tree, without upstream code.
Sibling nodes share a position, never attention or committed KV entries.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import DynamicCache


@dataclass(frozen=True)
class TreeConfig:
    forks: int = 4
    branch_tokens: int = 4

    def __post_init__(self):
        if type(self.forks) is not int or type(self.branch_tokens) is not int or self.forks < 0 or self.branch_tokens < 1:
            raise ValueError("Invalid draft tree budget")


@dataclass
class DraftTree:
    tokens: torch.Tensor
    parents: tuple[int, ...]
    depths: tuple[int, ...]
    native_nodes: int

    def mask(self, prefix: int, dtype: torch.dtype):
        n = len(self.parents)
        rows = []
        for node, parent in enumerate(self.parents):
            row = rows[parent].copy() if parent >= 0 else [False]*n
            row[node] = True
            rows.append(row)
        allowed = torch.tensor(rows, dtype=torch.bool, device=self.tokens.device)
        mask = torch.zeros((1, 1, n, prefix+n), dtype=dtype, device=self.tokens.device)
        mask[..., prefix:].masked_fill_(~allowed, torch.finfo(dtype).min)
        return mask

    def select(self, posterior: torch.Tensor):
        predictions, tokens = posterior[0].tolist(), self.tokens[0].tolist()
        children = [[] for _ in tokens]
        for node, parent in enumerate(self.parents[1:], 1):
            children[parent].append(node)
        path = [0]
        while True:
            next_node = next((i for i in children[path[-1]] if tokens[i] == predictions[path[-1]]), None)
            if next_node is None:
                break
            path.append(next_node)
        native = 0
        while native+1 < self.native_nodes and tokens[native+1] == predictions[native]:
            native += 1
        return path, predictions[path[-1]], native


def build_tree(block: torch.Tensor, logits: torch.Tensor, config: TreeConfig) -> DraftTree:
    if block.ndim != 2 or block.shape[0] != 1 or not block.shape[1]:
        raise ValueError("Expected a single anchored block")
    n = block.shape[1]
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] < n-1 or logits.shape[2] < 2:
        raise ValueError("Missing draft position distributions")
    native = block[0].tolist()
    tokens = native.copy()
    parents, depths = list(range(-1, n-1)), list(range(n))
    # Use the same argmax tie convention as native drafting, then remove it
    # before finding the alternative. topk ordering at ties need not agree.
    count = min(config.forks, n-1)
    if count:
        scores = logits[0, :count].clone()
        scores.scatter_(1, block[0, 1:count+1, None], -torch.inf)
        alternatives = scores.argmax(-1).tolist()
        for depth, token in enumerate(alternatives, 1):
            parent = depth-1
            for offset in range(min(config.branch_tokens, n-depth)):
                tokens.append(token if offset == 0 else native[depth+offset])
                parents.append(parent)
                depths.append(depth+offset)
                parent = len(tokens)-1
    return DraftTree(block.new_tensor([tokens]), tuple(parents), tuple(depths), n)


def commit_tree_cache(cache: DynamicCache, prefix: int, path: list[int]) -> None:
    if (not path or path[0] != 0 or len(set(path)) != len(path) or min(path) < 0
            or prefix+max(path) >= cache.get_seq_length()):
        raise ValueError("Invalid committed tree path")
    if path != list(range(len(path))):
        # Copy only the selected suffix, preserving the old prefix in place.
        # DynamicCache's storage API changed between the two pinned TF stacks.
        indices = None
        if hasattr(cache, "layers"):
            entries = [(layer.keys, layer.values) for layer in cache.layers]
        else:
            entries = list(zip(cache.key_cache, cache.value_cache))
        for keys, values in entries:
            if indices is None:
                indices = torch.tensor(path, device=keys.device)+prefix
            keys[..., prefix:prefix+len(path), :].copy_(keys.index_select(-2, indices))
            values[..., prefix:prefix+len(path), :].copy_(values.index_select(-2, indices))
    cache.crop(prefix+len(path))
