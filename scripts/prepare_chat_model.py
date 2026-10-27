"""Prepare a pinned optional chat model without touching the DFlash environment."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess


REPO = "ggml-org/Qwen3.8-27B-GGUF"
REVISION = "a316ea527fde32c12487be7574e2db5ac2ad2a7e"
FILENAME = "Qwen3.8-27B-Q8_0.gguf"
WEIGHT_SHA = "aab65c67ef0dad127960efef9247f1832bca105faa1c7a052cc039b223cf86a1"
WEIGHT_SIZE = 28595763648
LLAMA_TAG = "b11907"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def prepare(root, proxy, patch):
    root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, HTTP_PROXY=proxy, HTTPS_PROXY=proxy,
        HF_HUB_DISABLE_XET="1", HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DOWNLOAD_TIMEOUT="60")
    environment.pop("HF_HUB_OFFLINE", None)

    def model():
        # Reuse the installed official downloader's bounded resume behavior.
        code = ("from huggingface_hub import hf_hub_download; "
                "hf_hub_download(repo_id=" + repr(REPO) + ",revision=" + repr(REVISION) +
                ",filename=" + repr(FILENAME) + ",local_dir=" + repr(str(root / "model")) + ",token=False)")
        subprocess.run([os.sys.executable, "-c", code], env=environment, check=True)
        path = root / "model" / FILENAME
        if path.stat().st_size != WEIGHT_SIZE or digest(path) != WEIGHT_SHA:
            raise ValueError("Pinned chat model checksum mismatch")
        return {"repo": REPO, "revision": REVISION, "filename": FILENAME, "sha256": WEIGHT_SHA,
                "bytes": WEIGHT_SIZE, "license": "Apache-2.0"}

    def build():
        source = root / "llama.cpp"
        if not source.exists():
            subprocess.run(["git", "-c", "http.proxy=" + proxy, "clone", "--depth", "1", "--branch", LLAMA_TAG,
                            "https://github.com/ggml-org/llama.cpp.git", str(source)], check=True)
        tag = subprocess.check_output(["git", "-C", str(source), "describe", "--tags", "--exact-match"], text=True).strip()
        if tag != LLAMA_TAG:
            raise ValueError("Preserve existing llama.cpp checkout; expected the pinned tag")
        commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if subprocess.run(["git", "-C", str(source), "apply", "--reverse", "--check", str(patch)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            subprocess.run(["git", "-C", str(source), "apply", "--check", str(patch)], check=True)
            subprocess.run(["git", "-C", str(source), "apply", str(patch)], check=True)
        subprocess.run(["cmake", "-S", str(source), "-B", str(root / "build"), "-DGGML_CUDA=ON",
            "-UCUDA_*", "-UCUDAToolkit_*", "-DCUDAToolkit_ROOT=/usr/local/cuda-12.1",
            "-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc", "-DCMAKE_CUDA_ARCHITECTURES=86",
            "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_UI=OFF", "-DLLAMA_USE_PREBUILT_UI=OFF",
            "-DCMAKE_BUILD_TYPE=Release"], check=True)
        subprocess.run(["cmake", "--build", str(root / "build"), "--target", "llama-server", "-j", "12"], check=True)
        return {"repo": "https://github.com/ggml-org/llama.cpp", "tag": tag, "commit": commit,
                "license": "MIT", "binary_sha256": digest(root / "build/bin/llama-server"),
                "stream_token_patch_sha256": digest(patch)}

    with ThreadPoolExecutor(max_workers=2) as pool:
        weight, binary = pool.submit(model), pool.submit(build)
        result = {"model": weight.result(), "runtime": binary.result()}
    (root / "prepared.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prepared": True, "model_sha256": WEIGHT_SHA}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--patch", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.proxy, args.patch.resolve())
