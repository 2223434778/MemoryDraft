"""Target sibling counterfactuals constrain approximate tree commitments.

Distances are local next-token diagnostics, not bounds on sequence divergence.
Both sibling rows come from the same ancestor-masked Target verification pass.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import heapq
import math

import torch

from memorydraft.backends.asd import ASDConfig
from memorydraft.backends.budget_tree import BudgetTreeConfig, TreeASDVerifier
from memorydraft.backends.draft_tree import DraftTree


class PairedTreeConfig(BudgetTreeConfig):
    """Reserve Draft top-2 siblings together inside the existing node cap."""


def build_paired_tree(block, logits, config, context, *, stop_token_ids=(), regret_budget=None):
    if (not isinstance(config, PairedTreeConfig) or config.spine_mode != "none" or config.top_k < 2
            or block.ndim != 2 or block.shape[0] != 1 or not block.shape[1]
            or logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] < block.shape[1]-1
            or logits.shape[2] < config.top_k):
        raise ValueError("Paired tree requires native marginals, top-k >= 2 and no reserved spine")
    horizon = block.shape[1]-1
    scores = logits[0, :horizon].float()
    values, ids = scores.topk(config.top_k, dim=-1)
    probabilities = (values-scores.logsumexp(-1, keepdim=True)).exp()
    candidate_ids, weights = ids.tolist(), probabilities.tolist()
    if any(not math.isfinite(p) or not 0 <= p <= 1 for row in weights for p in row):
        raise ValueError("Non-finite Draft probabilities")
    tokens, parents, depths, masses = [int(block[0, 0])], [-1], [0], [1.]
    queue = []

    def expand(node):
        depth = depths[node]
        if depth >= horizon or tokens[node] in stop_token_ids:
            return
        # The first pair is indivisible; lower-ranked siblings remain singleton groups.
        for ranks in ((0, 1), *((rank,) for rank in range(2, config.top_k))):
            mass = sum(masses[node]*weights[depth][r] for r in ranks)
            heapq.heappush(queue, (-mass, depth+1, node, ranks))

    expand(0)
    prefix_mass, best, best_score, best_mass = 1., None, -math.inf, None
    while queue:
        _, depth, parent, ranks = heapq.heappop(queue)
        if len(tokens)+len(ranks) > config.max_nodes:
            continue
        added = []
        for rank in ranks:
            added.append(len(tokens))
            tokens.append(candidate_ids[depth-1][rank])
            parents.append(parent)
            depths.append(depth)
            mass = masses[parent]*weights[depth-1][rank]
            masses.append(mass)
            prefix_mass += mass
        count = len(tokens)
        if count >= config.min_nodes:
            objective = prefix_mass/config.costs.cycle(context, count)
            if objective > best_score:
                best, best_score, best_mass = count, objective, prefix_mass
        for node in added:
            expand(node)
    if best is None:
        best, best_mass = len(tokens), prefix_mass
    tree = DraftTree(block.new_tensor([tokens[:best]]), tuple(parents[:best]), tuple(depths[:best]), 1)
    tree.budget_metadata = dict(nodes=best, built_nodes=len(tokens), max_depth=max(depths[:best]),
        expected_advance=best_mass, estimated_cycle_seconds=config.costs.cycle(context, best),
        horizon=horizon, top_k=config.top_k, sibling_pairs=True)
    return tree


@dataclass(frozen=True)
class DriftConfig:
    max_js: float = .10
    protect_stop: bool = True

    def __post_init__(self):
        if (type(self.max_js) not in (int, float) or not math.isfinite(self.max_js)
                or not 0 <= self.max_js <= 1 or type(self.protect_stop) is not bool):
            raise ValueError("Invalid normalized JS threshold or stop policy")


def sibling_js(logits, pairs):
    """Full vocabulary JS/ln(2), T=1, batched on the existing logits device."""
    if not pairs:
        return []
    indices = torch.tensor(pairs, device=logits.device, dtype=torch.long)
    rows = logits.index_select(0, indices.flatten()).float()
    if not bool(torch.isfinite(rows).all()):
        raise ValueError("Non-finite Target counterfactual distributions")
    logp = rows.log_softmax(-1).reshape(len(pairs), 2, -1)
    middle = torch.logaddexp(logp[:, 0], logp[:, 1])-math.log(2)
    distance = (.5*(logp.exp()*(logp-middle[:, None])).sum((1, 2))/math.log(2)).clamp(0, 1)
    return distance.tolist()


class BranchDriftVerifier(TreeASDVerifier):
    def __init__(self, config=ASDConfig(budget=8), *, drift=DriftConfig(), record=False):
        super().__init__(config, record=record)
        self.drift = drift
        self.counters = Counter()
        self.last_edges = []

    def edge_permissions(self, tree, logits, predictions, tokens, *, horizon, stop_token_ids):
        permissions = [True]*len(tokens)
        children = {(parent, tokens[i]): i for i, parent in enumerate(tree.parents) if i}
        edges, pairs = [], []
        self.last_edges = []
        remaining = max(0., self.config.budget-self.spent)
        parent_rows = torch.tensor(tree.parents[1:], device=logits.device, dtype=torch.long)
        candidate_ids = tree.tokens[0, 1:]
        best_ids = torch.tensor([predictions[p] for p in tree.parents[1:]], device=logits.device)
        best_scores = logits[parent_rows, best_ids].tolist()
        proposed_scores = logits[parent_rows, candidate_ids].tolist()
        for i, parent in enumerate(tree.parents[1:], 1):
            if tokens[i] == predictions[parent]:
                continue
            regret = max(0., best_scores[i-1]-proposed_scores[i-1])
            if (not self.config.cap or not self.config.gate or not self.config.budget
                    or regret > remaining or regret/(horizon-tree.depths[i]+1) > self.config.gate):
                continue
            self.counters['eligible_edges'] += 1
            if self.drift.protect_stop and (tokens[i] in stop_token_ids or predictions[parent] in stop_token_ids):
                permissions[i] = False
                self.counters['stop_blocked'] += 1
                self.last_edges.append(dict(node=i, reason='stop'))
                continue
            reference = children.get((parent, predictions[parent]))
            if reference is None:
                permissions[i] = False
                self.counters['missing_reference'] += 1
                self.last_edges.append(dict(node=i, reason='missing_reference'))
                continue
            edges.append((i, reference))
            pairs.append((i, reference))
        for (node, reference), distance in zip(edges, sibling_js(logits, pairs)):
            allowed = distance <= self.drift.max_js
            permissions[node] = allowed
            self.counters['compared_edges'] += 1
            self.counters['compared_js_sum'] += distance
            self.counters['compared_js_max'] = max(self.counters['compared_js_max'], distance)
            self.counters['permitted_edges' if allowed else 'drift_blocked'] += 1
            self.last_edges.append(dict(node=node, reference=reference, js=distance,
                                        reason='permitted' if allowed else 'drift'))
        return permissions

    def decide_tree(self, tree, logits, **kwargs):
        before = self.relaxed
        result = super().decide_tree(tree, logits, **kwargs)
        selected = set(result[0])
        self.counters['selected_exceptions'] += self.relaxed-before
        self.counters['selected_compared_edges'] += sum(
            e['node'] in selected and e['reason'] == 'permitted' for e in self.last_edges)
        self.counters['selected_js_sum'] += sum(e['js'] for e in self.last_edges
            if e['node'] in selected and e['reason'] == 'permitted')
        if self.record:
            self.events[-1]['drift_edges'] = self.last_edges
        return result

    def summary(self):
        return dict(super().summary(), branch_drift=dict(max_js=self.drift.max_js,
            protect_stop=self.drift.protect_stop, temperature=1., vocabulary='full',
            missing_reference_policy='reject_approximation', **self.counters))
