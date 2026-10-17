import numpy as np
import pytest
from types import SimpleNamespace

from memorydraft.retrieval.pool import CachedSemanticEncoder


def test_embedding_cache_queries_live_and_evicts_without_scoring_other_chunks():
    class Model:
        calls = []

        def encode(self, texts, **kwargs):
            self.calls.append(texts)
            table = {"q": [1., 0.], "changed": [0., 1.], "a": [1., 0.], "b": [0., 1.], "other": [-1., 0.]}
            return np.array([table[text] for text in texts], dtype=np.float32)

    model = Model()

    dense = SimpleNamespace(model=model, name="pinned-stub")
    encoder = CachedSemanticEncoder(dense, capacity=2)
    assert encoder.scores("q", ["a", "b"]) == [1., 0.]
    assert encoder.scores("changed", ["a", "b"]) == [0., 1.]
    assert model.calls == [["q", "a", "b"], ["changed"]]
    assert encoder.scores("q", ["other"]) == [-1.]
    assert list(encoder.cache) == ["b", "other"]
    assert encoder.scores("q", ["a"]) == [1.]
    assert model.calls[-1] == ["q", "a"]
    assert encoder.stats() == dict(hits=2, misses=4, chunks=2, capacity=2, vector_bytes=16)
    assert encoder.scores("q", []) == []
    with pytest.raises(ValueError, match="positive"):
        CachedSemanticEncoder(dense, capacity=0)
