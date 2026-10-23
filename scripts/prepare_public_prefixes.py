"""Register public worked examples without caching the final test question."""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from memorydraft.evaluation.long_context import load_data, sha
from memorydraft.runtime.config import RuntimeConfig
from memorydraft.runtime.prefix_cache import tokenizer_sha256

ROOT = Path(__file__).resolve().parents[1]


def build(items, tokenizer, config):
    groups = {}
    for item in items:
        text = tokenizer.apply_chat_template(item["request"]["messages"], tokenize=False,
                    add_generation_prompt=True, **config.template_options)
        marker = "\n\nFinal question:\n"
        if text.count(marker) != 1:
            raise ValueError("Cannot identify the public demonstration boundary")
        public = text.split(marker)[0]+marker
        groups.setdefault(public, []).append(item)
    prefixes = []
    for public, group in groups.items():
        tokens = tokenizer.encode(public, add_special_tokens=False)
        full = [tokenizer.apply_chat_template(r["request"]["messages"], add_generation_prompt=True,
                    **config.template_options) for r in group]
        # A BPE token at the question boundary may merge across that boundary.
        while tokens and any(ids[:len(tokens)] != tokens for ids in full):
            tokens.pop()
        if not tokens or any(len(tokens) >= len(ids) for ids in full):
            raise ValueError("Public prefix must leave a nonempty question suffix")
        prefixes.append(dict(id="gsm-public-"+sha(public.encode())[:16], token_ids=tokens,
            source="Pinned public GSM8K train worked examples; data/workloads/asd-long-context/manifest.json",
            demonstration_ids=group[0]["demonstration_ids"], levels=sorted({r["level"] for r in group})))
    return dict(schema=1, visibility="public", target_revision=config.target_revision,
                tokenizer_sha256=tokenizer_sha256(tokenizer),template_options=config.template_options,
                prefixes=prefixes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"data/workloads/asd-prefix-cache/public-prefixes.json")
    args = parser.parse_args()
    _,items,_ = load_data(ROOT)
    config = RuntimeConfig.load(ROOT/"configs/runtime-dflash-asd.json")
    tokenizer = AutoTokenizer.from_pretrained(config.target,local_files_only=True,trust_remote_code=False)
    value = build(items,tokenizer,config)
    data = (json.dumps(value,ensure_ascii=False,indent=2)+"\n").encode()
    if args.output.exists():
        if args.output.read_bytes() != data:raise ValueError("Existing registry differs")
    else:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_bytes(data)
    print(json.dumps(dict(path=str(args.output),sha256=sha(data),tokens=[len(r["token_ids"]) for r in value["prefixes"]])))


if __name__ == "__main__":
    main()
