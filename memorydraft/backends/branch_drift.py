"""Target sibling counterfactuals constrain approximate tree commitments.

Distances are local next-token diagnostics, not bounds on sequence divergence.
Both sibling rows come from the same ancestor-masked Target verification pass.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import heapq
import math
import time

import torch

from memorydraft.backends.asd import ASDConfig
from memorydraft.backends.budget_tree import BudgetTreeConfig, RegretBudget, TreeASDVerifier
from memorydraft.backends.draft_tree import DraftTree


class PairedTreeConfig(BudgetTreeConfig):
    """Reserve Draft top-2 siblings together inside the existing node cap."""


@dataclass(frozen=True)
class GateTreeConfig(PairedTreeConfig):
    """Conservative Draft-only proxy; never a prediction of Target gate safety."""
    gate_penalty: float = 1.

    def __post_init__(self):
        super().__post_init__()
        if (type(self.gate_penalty) not in (int, float)
                or not math.isfinite(self.gate_penalty) or self.gate_penalty < 0):
            raise ValueError('Invalid gate allocation penalty')


class FeedbackTreeConfig(GateTreeConfig):
    """Use request-local width feedback carried by the verifier snapshot."""


@dataclass(frozen=True)
class WidthBudget(RegretBudget):
    gate_penalty: float = 1.

    def __post_init__(self):
        super().__post_init__()
        if type(self.gate_penalty) not in (int, float) or self.gate_penalty not in (0., .5, 1.):
            raise ValueError('Invalid feedback width snapshot')


def build_paired_tree(block, logits, config, context, *, stop_token_ids=(), regret_budget=None):
    started = time.perf_counter()
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
    gate_allocation = isinstance(config, GateTreeConfig)
    feedback = isinstance(config, FeedbackTreeConfig)
    if feedback and not isinstance(regret_budget, WidthBudget):
        raise ValueError('Feedback tree requires a width budget snapshot')
    penalty = regret_budget.gate_penalty if feedback else config.gate_penalty if gate_allocation else 0.

    def weight(depth, rank):
        probability = weights[depth][rank]
        if gate_allocation and rank:
            # Near ties keep alternatives; large Draft disagreements favor depth.
            probability *= (probability/max(weights[depth][0], 1e-30))**penalty
        return probability

    def expand(node):
        depth = depths[node]
        if depth >= horizon or tokens[node] in stop_token_ids:
            return
        # The first pair is indivisible; lower-ranked siblings remain singleton groups.
        groups = (tuple((rank,) for rank in range(config.top_k)) if gate_allocation
                  else ((0, 1), *((rank,) for rank in range(2, config.top_k))))
        for ranks in groups:
            mass = sum(masses[node]*weight(depth, r) for r in ranks)
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
            mass = masses[parent]*weight(depth-1, rank)
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
        horizon=horizon, top_k=config.top_k, sibling_pairs=not gate_allocation,
        allocation='draft-ratio-survival-proxy' if gate_allocation else 'paired',
        allocation_seconds=time.perf_counter()-started)
    if gate_allocation:
        tree.budget_metadata['gate_penalty'] = penalty
    if feedback:
        tree.width_candidates = tuple(tuple(row) for row in candidate_ids)
        tree.budget_metadata['width_feedback'] = True
    return tree


@dataclass(frozen=True)
class DriftConfig:
    max_js: float = .10
    protect_stop: bool = True
    evaluation: str = 'eager'
    max_token_regret: float | None = None

    def __post_init__(self):
        if (type(self.max_js) not in (int, float) or not math.isfinite(self.max_js)
                or not 0 <= self.max_js <= 1 or type(self.protect_stop) is not bool
                or self.evaluation not in ('eager', 'lazy')
                or (self.max_token_regret is not None and
                    (type(self.max_token_regret) not in (int, float)
                     or not math.isfinite(self.max_token_regret) or self.max_token_regret < 0))):
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

    def _prepare_edges(self, tree, logits, predictions, tokens, *, horizon, stop_token_ids):
        permissions = [True]*len(tokens)
        children = {(parent, tokens[i]): i for i, parent in enumerate(tree.parents) if i}
        pairs = []
        self.last_edges = []
        remaining = max(0., self.config.budget-self.spent)
        parent_rows = torch.tensor(tree.parents[1:], device=logits.device, dtype=torch.long)
        candidate_ids = tree.tokens[0, 1:]
        best_ids = torch.tensor([predictions[p] for p in tree.parents[1:]], device=logits.device,dtype=torch.long)
        best_scores = logits[parent_rows, best_ids].tolist()
        proposed_scores = logits[parent_rows, candidate_ids].tolist()
        regrets = [max(0., a-b) for a, b in zip(best_scores, proposed_scores)]
        for i, parent in enumerate(tree.parents[1:], 1):
            if tokens[i] == predictions[parent]:
                continue
            regret = regrets[i-1]
            if (not self.config.cap or not self.config.gate or not self.config.budget
                    or regret > remaining or regret/(horizon-tree.depths[i]+1) > self.config.gate):
                continue
            self.counters['eligible_edges'] += 1
            if self.drift.max_token_regret is not None and regret > self.drift.max_token_regret:
                permissions[i] = False
                self.counters['token_regret_blocked'] += 1
                self.last_edges.append(dict(node=i, reason='token_regret'))
                continue
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
            pairs.append((i, reference))
        return permissions, pairs, regrets

    def _compare(self, logits, pairs, permissions):
        started = time.perf_counter()
        distances = sibling_js(logits, pairs)
        if pairs:
            self.counters['js_batches'] += 1
            self.counters['js_seconds'] += time.perf_counter()-started
        for (node, reference), distance in zip(pairs, distances):
            allowed = distance <= self.drift.max_js
            permissions[node] = allowed
            self.counters['compared_edges'] += 1
            self.counters['compared_js_sum'] += distance
            self.counters['compared_js_max'] = max(self.counters['compared_js_max'], distance)
            self.counters['permitted_edges' if allowed else 'drift_blocked'] += 1
            self.last_edges.append(dict(node=node, reference=reference, js=distance,
                                        reason='permitted' if allowed else 'drift'))
    def edge_permissions(self, tree, logits, predictions, tokens, *, horizon, stop_token_ids):
        started = time.perf_counter()
        permissions, pairs, regrets = self._prepare_edges(tree, logits, predictions, tokens,
            horizon=horizon, stop_token_ids=stop_token_ids)
        if self.drift.evaluation == 'eager':
            self._compare(logits, pairs, permissions)
        else:
            self._lazy_permissions(tree, logits, predictions, tokens, horizon,
                                   stop_token_ids, permissions, pairs, regrets)
        self.counters['gate_seconds'] += time.perf_counter()-started
        return permissions

    def _lazy_permissions(self, tree, logits, predictions, tokens, horizon, stops, permissions, pairs, regrets):
        # Rank every cheaply feasible endpoint with exactly the base verifier's
        # tie-breaking. Unknown JS edges are optimistic until explicitly tested.
        references = dict(pairs)
        paths, costs, counts, feasible = [[0]], [0.], [0], [True]
        remaining = max(0., self.config.budget-self.spent)
        for i, parent in enumerate(tree.parents[1:], 1):
            mismatch = tokens[i] != predictions[parent]
            regret = regrets[i-1] if mismatch else 0.
            paths.append(paths[parent]+[i])
            costs.append(costs[parent]+regret)
            counts.append(counts[parent]+int(mismatch))
            allowed = not mismatch or (self.config.budget > 0 and self.config.gate > 0
                and self.config.cap > 0 and regret/(horizon-tree.depths[i]+1) <= self.config.gate)
            feasible.append(feasible[parent] and tokens[parent] not in stops and allowed
                and permissions[i] and counts[i] <= self.config.cap and costs[i] <= remaining)
        ordered = sorted((i for i in range(len(tokens)) if feasible[i]),
                         key=lambda i: (-tree.depths[i], costs[i], i))
        tested = set()
        for endpoint in ordered:
            path = paths[endpoint]
            if any(not permissions[i] for i in path):
                continue
            pending = [(i, references[i]) for i in path if i in references and i not in tested]
            self._compare(logits, pending, permissions)
            tested.update(i for i, _ in pending)
            if all(permissions[i] for i in path):
                break
        self.counters['skipped_comparisons'] += len(pairs)-len(tested)

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
            evaluation=self.drift.evaluation, max_token_regret=self.drift.max_token_regret,
            protect_stop=self.drift.protect_stop, temperature=1., vocabulary='full',
            missing_reference_policy='reject_approximation', **self.counters))
