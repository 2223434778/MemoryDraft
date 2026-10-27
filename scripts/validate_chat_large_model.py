"""Compare real native token IDs with the protected AR adapter, including UTF-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import httpx

from memorydraft.serving.llama import LlamaEngine


def validate(env, url, output):
    runtime = LlamaEngine()
    report = {"real_gpu": True, "checks": [], "purpose": "adapter correctness; no comparative quality claim"}
    try:
        with httpx.Client(base_url=url, headers={"Authorization": "Bearer " + env["MEMORYDRAFT_API_KEY"]},
                          timeout=650, trust_env=False) as client:
            report["health"] = client.get("/healthz").json()
            assert report["health"]["methods"] == ["AR"]
            for name, content in (("chinese", "请用一句中文说明9乘9等于多少。"),
                    ("utf8", "只输出这一行，不加解释：你好，世界！🦄🧑🏽‍🚀𠮷。")):
                messages = [{"role": "user", "content": content}]
                prompt = runtime.tokenizer.apply_chat_template(messages)
                response = runtime.client.post("/completion", json={"prompt": prompt, "temperature": 0,
                    "n_predict": 96, "stream": False, "return_tokens": True, "cache_prompt": False})
                response.raise_for_status(); native = response.json()
                response = client.post("/v1/generate", json={"id": "large-model-" + name,
                    "user_id": "synthetic-large-model-check", "order": 1, "messages": messages,
                    "max_new_tokens": 96, "method": "AR", "timeout_seconds": 600})
                response.raise_for_status(); actual = response.json()
                assert actual["token_ids"] == native["tokens"], (name, actual["token_ids"], native["tokens"])
                expected = runtime.tokenizer.decode(native["tokens"], skip_special_tokens=True)
                assert actual["text"] == expected and actual["text"] and "\ufffd" not in actual["text"]
                assert "<|im_end|>" not in actual["text"]
                report["checks"].append({"case": name, "identical_native_ids": True, "identical_decoding": True,
                    "no_unicode_replacement": True, "tokens": len(actual["token_ids"]), "characters": len(actual["text"]),
                    "text_sha256": hashlib.sha256(actual["text"].encode()).hexdigest(),
                    "finish_reason": actual["finish_reason"], "server_seconds": actual["server_seconds"]})
    finally:
        runtime.close()
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"large_model_checks": len(report["checks"]), "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    environment = dict(line.split("=", 1) for line in args.env.read_text().splitlines() if line and not line.startswith("#"))
    validate(environment, args.url, args.output)
