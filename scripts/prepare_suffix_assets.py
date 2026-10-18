"""Fetch the pinned Apache-2.0 SuffixDecoding component without importing vLLM."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
REVISION = "aca5d9a8a62474035c15d114d40a01abc8c94b51"
FILES = ["LICENSE", "arctic_inference/suffix_decoding/cache.py", "csrc/suffix_decoding/CMakeLists.txt",
         "csrc/suffix_decoding/bindings.cc", "csrc/suffix_decoding/int32_map.h",
         "csrc/suffix_decoding/suffix_tree.cc", "csrc/suffix_decoding/suffix_tree.h"]


def main():
    directory = ROOT / "models/arctic-suffix-audit"
    manifest_path = ROOT / "configs/suffix-manifest.json"
    frozen = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    entries = []
    for filename in FILES:
        path = directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            data = urlopen(f"https://raw.githubusercontent.com/snowflakedb/ArcticInference/{REVISION}/{filename}",
                           timeout=20).read()
            path.write_bytes(data)
        data = path.read_bytes()
        entry = dict(filename=filename, sha256=hashlib.sha256(data).hexdigest())
        if frozen and entry not in frozen["files"]:
            raise ValueError("Pinned SuffixDecoding source mismatch: " + filename)
        entries.append(entry)
    # Empty package initializers avoid the full plugin's unrelated vLLM imports.
    for filename in ("arctic_inference/__init__.py", "arctic_inference/suffix_decoding/__init__.py"):
        path = directory / filename
        if path.exists() and path.read_bytes():
            raise ValueError("Unexpected plugin initializer")
        path.write_bytes(b"")
    result = dict(repository="https://github.com/snowflakedb/ArcticInference", revision=REVISION,
                  license="Apache-2.0", files=entries, adaptation="Unmodified C++/cache.py; empty package initializers; standalone CMake build")
    manifest_path.write_bytes((json.dumps(result, indent=2) + "\n").encode())
    notice = ROOT / "docs/third_party/ARCTIC_SUFFIX_LICENSE.txt"
    notice.write_bytes((directory / "LICENSE").read_bytes())
    print("Verified", len(entries), "official files at", REVISION)


if __name__ == "__main__":
    main()
