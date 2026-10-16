"""Download only pinned public assets into this project's models directory."""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


parser = argparse.ArgumentParser()
parser.add_argument("--embedding", action="store_true")
args = parser.parse_args()
model, revision, destination = (
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "e8f8c211226b894fcb81acc59f3b34ba3efd5f42", "multilingual-minilm")
    if args.embedding else
    ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775", "qwen2.5-0.5b")
)
root = Path(__file__).resolve().parents[1] / "models" / destination
print(snapshot_download(model, revision=revision, local_dir=root,
                        allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "LICENSE", "README.md"],
                        ignore_patterns=["onnx/*", "openvino/*"], max_workers=2))
