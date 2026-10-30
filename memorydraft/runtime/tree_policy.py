"""Explicit deployment policy for the evaluated high-fidelity tree candidate."""
from dataclasses import asdict, dataclass
import math

from memorydraft.backends.branch_drift import DriftConfig, FeedbackTreeConfig
from memorydraft.backends.gate_width import MinGainFeedbackVerifier


@dataclass(frozen=True)
class TreePolicySpec:
    mode: str = 'min-gain-feedback'
    max_js: float = .1
    max_token_regret: float = math.log(4)
    min_extra_tokens: int = 2
    reuse_scores: bool = True
    buffered_kv: bool = True
    cache_block_tokens: int = 256

    def __post_init__(self):
        if self.mode != 'min-gain-feedback':
            raise ValueError('Unknown tree policy mode')
        DriftConfig(max_js=self.max_js, max_token_regret=self.max_token_regret)
        if (self.max_token_regret is None or type(self.min_extra_tokens) is not int or self.min_extra_tokens < 1
                or type(self.reuse_scores) is not bool or type(self.buffered_kv) is not bool
                or type(self.cache_block_tokens) is not int or not 1 <= self.cache_block_tokens <= 4096):
            raise ValueError('Invalid tree policy execution or minimum gain settings')

    def bind(self, tree):
        if tree.top_k < 2:
            raise ValueError('High-fidelity tree requires top_k >= 2')
        return FeedbackTreeConfig(tree.costs, tree.min_nodes, tree.max_nodes, tree.top_k, tree.spine_mode)

    def verifier(self, asd):
        return MinGainFeedbackVerifier(asd, min_extra_tokens=self.min_extra_tokens, reuse_scores=self.reuse_scores,
            drift=DriftConfig(max_js=self.max_js, max_token_regret=self.max_token_regret, evaluation='lazy'))

    def cache_factory(self):
        if self.buffered_kv:
            from memorydraft.backends.buffered_cache import BufferedDynamicCache
            return lambda: BufferedDynamicCache(block_tokens=self.cache_block_tokens)
        return None

    def describe(self):
        return asdict(self)
