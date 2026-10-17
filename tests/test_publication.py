import gzip
import hashlib

from scripts.publish_research_artifacts import publish


def test_publication_is_lossless_and_excludes_database_weights_and_other_roots(tmp_path):
    source = tmp_path / "source"
    stage = source / "qwen3-pilot" / "smoke"
    stage.mkdir(parents=True)
    raw = b'{"correct":true,"token_ids":[1,2]}\n'
    (stage / "replay.jsonl").write_bytes(raw)
    (stage / "history.db").write_bytes(b"not published")
    (stage / "model.safetensors").write_bytes(b"not published")
    other = source / "unrelated"
    other.mkdir()
    (other / "private.json").write_text("{}")
    destination = tmp_path / "published"
    entries = publish(source, destination)
    assert len(entries) == 1
    assert gzip.decompress((destination / entries[0]["published"]).read_bytes()) == raw
    assert entries[0]["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert not list(destination.rglob("*.db"))
    assert not list(destination.rglob("*.safetensors"))
    assert not (destination / "unrelated").exists()
