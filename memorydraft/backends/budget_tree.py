"""BASTION-style best-first trees and request-budgeted ASD path selection.

Independently implements the path-sum construction in arXiv:2605.29727,
equations 1/3/6. Uses a measured latency table, not the paper's roofline model.
No upstream implementation is vendored. See docs/ASD_TREE_PROTOCOL.md.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import torch

from memorydraft.backends.asd import ASDConfig, ASDDecision, ASDVerifier
from memorydraft.backends.draft_tree import DraftTree


@dataclass(frozen=True)
class TreeCosts:
    contexts: tuple[int, ...]
    nodes: tuple[int, ...]
    seconds: tuple[tuple[float, ...], ...]
    draft_seconds: float
    auxiliary_seconds: float = 0.

    def __post_init__(self):
        if (len(self.contexts) < 2 or len(self.nodes) < 2
                or tuple(sorted(set(self.contexts))) != self.contexts
                or tuple(sorted(set(self.nodes))) != self.nodes
                or min(self.contexts) < 1 or min(self.nodes) < 1
                or len(self.seconds) != len(self.contexts)
                or any(len(row) != len(self.nodes) for row in self.seconds)
                or any(not math.isfinite(v) or v <= 0 for row in self.seconds for v in row)
                or not math.isfinite(self.draft_seconds) or self.draft_seconds <= 0
                or not math.isfinite(self.auxiliary_seconds) or self.auxiliary_seconds < 0):
            raise ValueError("Invalid measured tree latency table")

    def cycle(self, context: int, nodes: int) -> float:
        def interpolate(xs, ys, value):
            i = next((i for i in range(len(xs)-1) if value <= xs[i+1]), len(xs)-2)
            return ys[i] + (ys[i+1]-ys[i]) * (value-xs[i])/(xs[i+1]-xs[i])
        values = [interpolate(self.nodes, row, nodes) for row in self.seconds]
        verification = max(min(values), interpolate(self.contexts, values, context))
        return self.draft_seconds + self.auxiliary_seconds + verification


@dataclass(frozen=True)
class BudgetTreeConfig:
    costs: TreeCosts
    min_nodes: int = 8
    max_nodes: int = 32
    top_k: int = 4

    def __post_init__(self):
        if (not isinstance(self.costs, TreeCosts)
                or any(type(v) is not int for v in (self.min_nodes, self.max_nodes, self.top_k))
                or not 2 <= self.min_nodes <= self.max_nodes <= 64 or not 1 <= self.top_k <= 8):
            raise ValueError("Invalid cost-controlled tree bounds")


def build_budget_tree(block: torch.Tensor, logits: torch.Tensor,
                      config: BudgetTreeConfig, context: int, *, stop_token_ids=()) -> DraftTree:
    if (block.ndim != 2 or block.shape[0] != 1 or not block.shape[1]
            or logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] < block.shape[1]-1
            or logits.shape[2] < config.top_k):
        raise ValueError("Missing anchored block or aligned draft distributions")
    horizon = block.shape[1]-1
    tokens, parents, depths, masses = [int(block[0, 0])], [-1], [0], [1.]
    if not horizon:
        return DraftTree(block.clone(), (-1,), (0,), 1)
    scores = logits[0, :horizon].float()
    values, ids = scores.topk(config.top_k, dim=-1)
    probabilities = (values-scores.logsumexp(-1, keepdim=True)).exp()
    candidate_ids, weights = ids.tolist(), probabilities.tolist()
    if any(not math.isfinite(p) or p < 0 or p > 1 for row in weights for p in row):
        raise ValueError("Non-finite draft path probabilities")
    queue = []
    def expand(node):
        depth = depths[node]
        if depth < horizon and tokens[node] not in stop_token_ids:
            for token, probability in zip(candidate_ids[depth], weights[depth]):
                mass = masses[node]*probability
                heapq.heappush(queue, (-mass, depth+1, node, token))
    expand(0)
    prefix_mass = 1.
    best, best_score, best_mass = None, -math.inf, None
    while queue and len(tokens) < config.max_nodes:
        negative_mass, depth, parent, token = heapq.heappop(queue)
        tokens.append(token)
        parents.append(parent)
        depths.append(depth)
        masses.append(-negative_mass)
        prefix_mass -= negative_mass
        count = len(tokens)
        if count >= config.min_nodes:
            score = prefix_mass/config.costs.cycle(context, count)
            # Prefer fewer nodes at an exact objective tie.
            if score > best_score:
                best, best_score, best_mass = count, score, prefix_mass
        expand(count-1)
    if best is None:
        best, best_mass = len(tokens), prefix_mass
    tree = DraftTree(block.new_tensor([tokens[:best]]), tuple(parents[:best]), tuple(depths[:best]), 1)
    tree.budget_metadata = dict(nodes=best, built_nodes=len(tokens), max_depth=max(depths[:best]),
        expected_advance=best_mass, estimated_cycle_seconds=config.costs.cycle(context, best),
        horizon=horizon, top_k=config.top_k)
    return tree


class TreeASDVerifier(ASDVerifier):
    """Choose longest feasible root path, breaking length ties by least regret.

Every edge uses its parent's Target row; sibling states are never mixed.
The local denominator is the original proposal horizon, independent of width.
"""
    def __init__(self, config=ASDConfig(budget=8), *, record=False):
        if config.value != "remaining" or config.selection != "longest":
            raise ValueError("Tree ASD currently supports the original remaining/longest rule")
        super().__init__(config, record=record)

    def decide_tree(self, tree, logits, *, horizon, stop_token_ids=()):
        n = len(tree.parents)
        if (logits.ndim != 2 or logits.shape[0] != n or tree.tokens.shape != (1, n)
                or tree.parents[0] != -1 or tree.depths[0] != 0 or max(tree.depths) > horizon):
            raise ValueError("Invalid tree verification scores or horizon")
        if any(not 0 <= p < i or tree.depths[i] != tree.depths[p]+1
               for i, p in enumerate(tree.parents) if i):
            raise ValueError("Tree must be ordered and ancestor closed")
        top, top_ids = logits.max(-1)
        parent_ids = torch.tensor(tree.parents[1:], device=logits.device)
        proposed = logits[parent_ids, tree.tokens[0, 1:]]
        maxima, predictions = top.tolist(), top_ids.tolist()
        candidates, tokens = proposed.tolist(), tree.tokens[0].tolist()
        if not all(math.isfinite(v) for v in (*maxima, *candidates)):
            raise ValueError("Non-finite Target tree scores")
        paths, costs, counts, feasible = [[0]], [0.], [0], [True]
        strict_path, _, _ = tree.select(top_ids.unsqueeze(0))
        stop_at = next((j+1 for j, i in enumerate(strict_path) if tokens[i] in stop_token_ids), len(strict_path))
        strict_path = strict_path[:stop_at]
        for i, parent in enumerate(tree.parents[1:], 1):
            paths.append(paths[parent]+[i])
            mismatch = tokens[i] != predictions[parent]
            regret = max(0., maxima[parent]-candidates[i-1]) if mismatch else 0.
            costs.append(costs[parent]+regret)
            counts.append(counts[parent]+int(mismatch))
            allowed = not mismatch or (self.config.budget > 0 and self.config.gate > 0
                and self.config.cap > 0 and regret/(horizon-tree.depths[i]+1) <= self.config.gate)
            feasible.append(feasible[parent] and tokens[parent] not in stop_token_ids and allowed
                and counts[i] <= self.config.cap and costs[i] <= max(0., self.config.budget-self.spent))
        chosen = max((i for i in range(n) if feasible[i]),
                     key=lambda i: (tree.depths[i], -costs[i], -i))
        path, amount = paths[chosen], costs[chosen]
        if self.spent+amount > self.config.budget or not math.isfinite(amount):
            raise ValueError("Exceeded tree ASD request budget")
        self.spent += amount
        self.relaxed += counts[chosen]
        self.blocks += 1
        decision = ASDDecision(len(path)-1, len(strict_path)-1, counts[chosen], amount)
        if self.record:
            self.events.append(dict(path=path, regret=amount, relaxed=counts[chosen], spent=self.spent,
                                    strict_path=strict_path, feasible_nodes=sum(feasible)))
        return path, predictions[chosen], decision
