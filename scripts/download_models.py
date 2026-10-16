"""Download only pinned public assets into this project's models directory."""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


parser = argparse.ArgumentParser()
assets = parser.add_mutually_exclusive_group()
assets.add_argument("--embedding", action="store_true")
assets.add_argument("--qwen3", choices=["target", "draft"])
parser.add_argument("--metadata-only", action="store_true")
args = parser.parse_args()
model, revision, destination = (
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "e8f8c211226b894fcb81acc59f3b34ba3efd5f42", "multilingual-minilm")
    if args.embedding else
    ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775", "qwen2.5-0.5b")
)
if args.qwen3:
    model, revision, destination = {
        "target": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", "qwen3-8b"),
        "draft": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", "qwen3-0.6b"),
    }[args.qwen3]
root = Path(__file__).resolve().parents[1] / "models" / destination
print(snapshot_download(model, revision=revision, local_dir=root,
                        allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "LICENSE", "README.md"],
                        ignore_patterns=["onnx/*", "openvino/*", *(["*.safetensors"] if args.metadata_only else [])], max_workers=2))
