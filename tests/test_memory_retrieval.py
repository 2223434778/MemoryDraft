import pytest

from memorydraft.memory.store import MemoryEvent, MemoryStore, Snapshot
from memorydraft.retrieval.pool import Chunk, ContinuationIndex, MemoryBundle
from memorydraft.routing.policy import Router


def test_snapshot_future_correction_and_user_isolation(tmp_path):
    with MemoryStore(tmp_path / "memory.db") as store:
        store.append([MemoryEvent("alice", "r", 1, "old"), MemoryEvent("bob", "r", 1, "private"),
                      MemoryEvent("alice", "r", 5, "old", "superseded"), MemoryEvent("alice", "new", 5, "new")])
        assert [r.text for r in store.snapshot("alice", 5).records] == ["old"]
        assert [r.text for r in store.snapshot("alice", 6).records] == ["new"]
        store.append([MemoryEvent("alice", "new", 8, "", "deleted")])
        assert store.snapshot("alice", 9).records == ()
        assert [r.text for r in store.snapshot("alice", 8).records] == ["new"]
        with pytest.raises(ValueError, match="overwrite"):
            store.append([MemoryEvent("alice", "r", 1, "changed")])
        store.purge("alice", "r")
        assert store.snapshot("alice", 3).records == ()
        assert len(store.snapshot("bob", 3).records) == 1


def test_persistence_and_atomic_import(tmp_path):
    path = tmp_path / "memory.db"
    with MemoryStore(path) as store:
        store.append([MemoryEvent("u", "r", 1, "hello")])
    with MemoryStore(path) as store:
        assert store.snapshot("u", 2).records[0].text == "hello"
        with pytest.raises(ValueError):
            store.append([MemoryEvent("u", "s", 1, "rolled back"), MemoryEvent("u", "r", 1, "collision")])
        assert len(store.snapshot("u", 2).records) == 1


def test_continuation_boundaries_and_confirmed_prefix():
    bundle = MemoryBundle("s", (Chunk("a", "", (1, 2, 3, 4, 5), 1), Chunk("b", "", (6, 7), 1)), "test")
    index = ContinuationIndex(bundle, min_match=2)
    assert index.match([], 8) == ([], None, 0)
    assert index.match([9, 1, 2, 3], 8) == ([4, 5], "a", 3)
    assert index.match([4, 5], 8) == ([], None, 0)
    assert index.match([1, 2], 1)[0] == [3]


def test_router_probes_and_cooldown():
    router = Router("RHD", True)
    for expected in ("H", "H", "D", "D"):
        assert router.choose(False) == expected
        router.update(expected, 4, 2, 3, 0.03 if expected == "H" else 0.06)
    for _ in range(2):
        assert router.choose(False) == "AR"
        router.update("AR", 0, 0, 1, 0.02)
    assert router.choose(False) == "H"
    assert router.choose(True) == "R"
    router.update("R", 4, 0, 1, 0.01)
    assert router.choose(True) != "R"


def test_empty_history_can_fall_back_to_ar():
    router = Router("RHD", False)
    for route in ("D", "D", "AR", "AR"):
        assert router.choose(False) == route
        router.update(route, 4 if route == "D" else 0, 0, 1, 0.2 if route == "D" else 0.01)
    assert router.choose(False) == "AR"
