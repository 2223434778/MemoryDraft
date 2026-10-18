"""Official linear SuffixDecoding candidates with the same strict DFlash verifier.

This is a candidate-component comparison, not the full vLLM/tree-speculation system.
The pinned, unmodified Apache-2.0 component lives in models/arctic-suffix-audit.
"""
from __future__ import annotations

from pathlib import Path
import sys

from memorydraft.retrieval.reuse import ReuseProposal


class ArcticMatcher:
    def __init__(self, snapshot, tokenizer, prompt_text, *, directory: Path, depth=64):
        sys.path.insert(0, str(directory))
        from arctic_inference.suffix_decoding.cache import SuffixDecodingCache

        self.cache = SuffixDecodingCache(max_tree_depth=depth)
        self.lookups = dict(arctic=0)
        special = set(tokenizer.all_special_ids)
        # Separate tree sequence IDs preserve record/special-token boundaries.
        for record in snapshot.records:
            segment = []
            for token in [*tokenizer.encode(record.text, add_special_tokens=False), None]:
                if token is None or token in special:
                    if segment:
                        identifier = (record.record_id, len(self.cache.cached_requests))
                        self.cache.start_request(identifier, [])
                        self.cache.add_active_response(identifier, segment)
                        self.cache.stop_request(identifier)
                    segment = []
                else:
                    segment.append(token)
        # Request text is not formatted chat control tokens, matching the Global resource.
        prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
        if special.intersection(prompt):
            raise ValueError("Public CI text unexpectedly contains model special tokens")
        self.cache.start_request("current", prompt)
        self.bytes_used = (self.cache._global_tree.estimate_memory()
                           + self.cache._local_trees["current"].estimate_memory())

    def propose(self, confirmed, max_tokens):
        self.lookups["arctic"] += 1
        draft = self.cache.speculate("current", confirmed, max_spec_tokens=max_tokens,
                                     max_spec_factor=1.0, max_spec_offset=0.0,
                                     min_token_prob=0.1, use_tree_spec=False)
        if not draft.token_ids:
            return None
        return ReuseProposal(tuple(draft.token_ids), "arctic:legal-resources", draft.match_len,
                             -1, -1, "arctic")

    def observe(self, proposal, accepted, confirmed):
        if not 0 <= accepted <= len(proposal.tokens):
            raise ValueError("Invalid accepted prefix length")
        # Both candidate methods use only the same frozen past files/answers and prompt.
        # No current-response tree extension or hidden extra candidate resource is used.
