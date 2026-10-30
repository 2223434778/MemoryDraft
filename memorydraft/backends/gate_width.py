"""Request-local tree width from observed, recoverable coverage misses."""
from collections import Counter, deque
import time

from memorydraft.backends.branch_drift import BranchDriftVerifier, WidthBudget


class FeedbackWidthVerifier(BranchDriftVerifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.width_history = deque(maxlen=4)
        self.width_counts = Counter()

    @property
    def width_penalty(self):
        misses = sum(self.width_history)
        return 1. if not misses else .5 if misses == 1 else 0.

    def construction_budget(self):
        base = super().construction_budget()
        return WidthBudget(base.total, base.remaining, base.gate, base.cap, self.width_penalty)

    def decide_tree(self, tree, logits, *, horizon, stop_token_ids=()):
        if not hasattr(tree, 'width_candidates') or len(tree.width_candidates) < horizon:
            raise ValueError('Feedback verifier requires draft rank metadata')
        # The returned recovery token is already the Target argmax at the
        # committed terminal node; no additional Target work is needed.
        result = super().decide_tree(tree, logits, horizon=horizon, stop_token_ids=stop_token_ids)
        started = time.perf_counter()
        path, prediction, _ = result
        terminal = path[-1]
        depth = tree.depths[terminal]
        tokens = tree.tokens[0].tolist()
        miss = False
        if depth < horizon and prediction not in stop_token_ids and tokens[terminal] not in stop_token_ids:
            ranks = tree.width_candidates[depth]
            rank = ranks.index(prediction) if prediction in ranks else -1
            covered = any(parent == terminal and tokens[i] == prediction
                          for i, parent in enumerate(tree.parents) if i)
            miss = rank > 0 and not covered
            self.width_counts['rank_outside_topk' if rank < 0 else 'rank_'+str(rank+1)] += 1
        self.width_history.append(miss)
        self.width_counts['recoverable_misses'] += int(miss)
        self.width_counts['blocks'] += 1
        used = tree.budget_metadata['gate_penalty']
        self.width_counts['penalty_'+str(used)] += 1
        tree.budget_metadata.update(width_recoverable_miss=miss, width_next_penalty=self.width_penalty)
        self.width_counts['feedback_seconds'] += time.perf_counter()-started
        return result

    def summary(self):
        return dict(super().summary(), width_feedback=dict(window=4, next_penalty=self.width_penalty,
            signal='uncovered Target argmax in Draft ranks 2-4 at committed boundary', **self.width_counts))
