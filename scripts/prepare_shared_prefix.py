"""Coalesce registered public 4K prefixes at their exact token intersection."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = "9bfda318cd757144b64961e8cc05d6a29f95c00e9c65229b2ab8ca2a365bae82"


def build(registry, parent_sha):
    if registry.get("schema") != 1 or registry.get("visibility") != "public":
        raise ValueError("Only an existing public registry may be coalesced")
    selected = [p for p in registry["prefixes"] if p.get("levels") == [4096]]
    if len(selected) < 2:raise ValueError("At least two public 4K prefixes are required")
    sequences = [p["token_ids"] for p in selected]
    if any(not s or any(type(t) is not int or t < 0 for t in s) for s in sequences):
        raise ValueError("Invalid registered prefix tokens")
    common = []
    for tokens in zip(*sequences):
        if len(set(tokens)) != 1:break
        common.append(tokens[0])
    if not common:raise ValueError("No nonempty shared public prefix")
    digest = hashlib.sha256(json.dumps(common).encode()).hexdigest()
    entry = dict(id="gsm-shared-"+digest[:16],token_ids=common,levels=[4096],
        source="Exact common token prefix of already registered public worked examples",
        parent_registry_sha256=parent_sha,parent_prefix_ids=[p["id"] for p in selected],
        parent_prefix_lengths=[len(p["token_ids"]) for p in selected])
    result = deepcopy(registry)
    result["prefixes"] = [p for p in result["prefixes"] if p.get("levels") != [4096]]+[entry]
    result["derivation"] = dict(parent_registry_sha256=parent_sha,rule="Longest exact common prefix of public 4K entries; no request answers used")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"data/workloads/asd-shared-prefix/public-prefixes.json")
    args = parser.parse_args()
    raw = (ROOT/"data/workloads/asd-prefix-cache/public-prefixes.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != PARENT_SHA:raise ValueError("Changed parent registry")
    value = build(json.loads(raw),PARENT_SHA)
    data = (json.dumps(value,ensure_ascii=False,indent=2)+"\n").encode()
    if args.output.exists():
        if args.output.read_bytes() != data:raise ValueError("Existing shared registry differs")
    else:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_bytes(data)
    print(json.dumps(dict(path=str(args.output),sha256=hashlib.sha256(data).hexdigest(),
                         tokens=[len(p["token_ids"]) for p in value["prefixes"]])))


if __name__ == "__main__":
    main()
