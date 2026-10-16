"""Download public mirror assets only when their pinned HF SHA-256 is known."""
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path


root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "configs/qwen3-download-manifest.json").read_text())


def download(item):
    path = root / "models" / item["destination"] / item["filename"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if hashlib.file_digest(path.open("rb"), "sha256").hexdigest() != item["sha256"]:
            raise ValueError("Existing weight checksum mismatch: " + path.name)
        return path.name
    partial = path.with_suffix(path.suffix + ".partial")
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            def fetch_range(first):
                last = min(first + 16 * 1024 * 1024, item["size"]) - 1
                url = item["url"] + f"?download=true&range_start={first}"
                request = urllib.request.Request(url, headers={"Range": f"bytes={first}-{last}", "User-Agent": "MemoryDraft"})
                with urllib.request.urlopen(request, timeout=40) as response:
                    if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {first}-"):
                        raise RuntimeError("Mirror did not honor resume range")
                    data = response.read(last - first + 2)
                if len(data) != last - first + 1:
                    raise ValueError("Invalid range response size")
                return data
            while offset < item["size"]:
                starts = range(offset, min(offset + 4 * 16 * 1024 * 1024, item["size"]), 16 * 1024 * 1024)
                # Bounded batches cap buffering at 64 MiB per file and preserve a contiguous resume prefix.
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as parts:
                    chunks = list(parts.map(fetch_range, starts))
                with partial.open("ab") as stream:
                    for data in chunks:
                        stream.write(data)
                        offset += len(data)
            if partial.stat().st_size != item["size"]:
                raise ValueError("Downloaded size mismatch")
            with partial.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != item["sha256"]:
                raise ValueError("Downloaded checksum mismatch")
            partial.replace(path)
            print("verified", item["destination"], path.name, item["sha256"], flush=True)
            return path.name
        except (OSError, RuntimeError, ValueError) as error:
            print("retry", path.name, attempt + 1, type(error).__name__, flush=True)
            if attempt == 2:
                raise
            time.sleep(2)


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(download, manifest))
    stamp = dict(manifest_sha256=hashlib.sha256((root / "configs/qwen3-download-manifest.json").read_bytes()).hexdigest(),
                 verified=[{key: x[key] for key in ("model", "hf_revision", "filename", "sha256", "size")} for x in manifest])
    marker = root / "models/qwen3-weights-verified.json"
    marker.with_suffix(".tmp").write_text(json.dumps(stamp, indent=2) + "\n")
    marker.with_suffix(".tmp").replace(marker)
    print("all pinned weight files verified", flush=True)
