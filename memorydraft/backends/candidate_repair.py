"""One-shot old-suffix proposals after a strictly verified recovery anchor.

Only token proposals are reused. Rejected Target KV, logits and features are
discarded; the changed context is verified again. No history or answers enter.
"""
from memorydraft.retrieval.reuse import ReuseProposal


class SuffixRepair:
    def __init__(self, max_tokens: int):
        if type(max_tokens) is not int or max_tokens < 2:
            raise ValueError("Suffix repair requires at least two proposal tokens")
        self.max_tokens = max_tokens
        self.pending: tuple[int, ...] = ()

    @staticmethod
    def _proposal(tokens, mode):
        return ReuseProposal(tuple(tokens), "request-draft", 0, 0, len(tokens), mode, True)

    def propose(self, confirmed, max_tokens):
        tokens, self.pending = self.pending[:max_tokens], ()
        return self._proposal(tokens, "suffix-repair") if len(tokens) >= 2 else None

    def propose_with_draft(self, confirmed, draft_tokens, max_tokens):
        return self._proposal(draft_tokens, "fresh-draft")

    def observe(self, proposal, accepted, confirmed):
        if not 0 <= accepted <= len(proposal.tokens):
            raise ValueError("Invalid accepted repair prefix")
        self.pending = ()
        if proposal.lookup == "fresh-draft" and accepted < len(proposal.tokens):
            # The rejected token is replaced by the confirmed recovery anchor.
            # Align the remaining tokens AFTER it; never retry the rejected one.
            self.pending = proposal.tokens[accepted+1:accepted+1+self.max_tokens]
