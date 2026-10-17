"""Download only pinned, public DFlash assets with recorded SHA-256 hashes."""
import hashlib
import json
import urllib.request
from pathlib import Path

from scripts.download_verified_weights import download


root = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    manifest = json.loads((root / "configs/dflash-manifest.json").read_text())
    destination = root / "models/dflash-qwen3-8b"
    destination.mkdir(parents=True, exist_ok=True)
    for item in manifest["code"]:
        path = destination / item["filename"]
        if not path.exists():
            with urllib.request.urlopen(item["url"], timeout=40) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("Code checksum mismatch: " + item["filename"])
            path.write_bytes(data)
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Existing code checksum mismatch")
    download(manifest["weight"])
    (destination / "VERIFIED.json").write_text(json.dumps(manifest, indent=2) + "\n")
