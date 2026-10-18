"""Publish allowlisted public/synthetic results with lossless trace compression."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


NAMES = ("qwen3-pilot", "qwen3-inherited-defaults", "research-old-bounded", "research-old-controls", "h-causal", "h-teacher-router", "h-training", "dflash-baseline", "reuse-pilot", "reuse-pilot-lookup-failure", "reuse-confirmation", "config-revisions-development", "config-revisions-confirmation", "config-revisions-contract-probe", "release-triage-feasibility")


def publish(source: Path, destination: Path):
    entries = []
    for name in NAMES:
        for path in sorted((source / name).rglob("*")):
            if not path.is_file() or path.suffix not in {".json", ".jsonl", ".txt", ".log"}:
                continue
            data = path.read_bytes()
            relative = path.relative_to(source)
            target = destination / relative
            encoded = data
            if path.suffix == ".jsonl":
                target = target.with_suffix(".jsonl.gz")
                encoded = gzip.compress(data, mtime=0)
                if gzip.decompress(encoded) != data:
                    raise RuntimeError("Lossless compression check failed")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(encoded)
            entries.append(dict(source=relative.as_posix(), raw_bytes=len(data),
                                raw_sha256=hashlib.sha256(data).hexdigest(),
                                published=target.relative_to(destination).as_posix(),
                                published_bytes=len(encoded), published_sha256=hashlib.sha256(encoded).hexdigest()))
    (destination / "research-v2-publication.json").write_bytes((json.dumps(entries, indent=2) + "\n").encode("utf-8"))
    return entries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", default="results", type=Path)
    args = parser.parse_args()
    entries = publish(args.source, args.destination)
    print("Published", len(entries), "allowlisted artifacts; JSONL content was losslessly compressed.")
