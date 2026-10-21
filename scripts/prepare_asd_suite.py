"""Freeze broader tasks before generation, with separate development/validation/reserve."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import random
import shutil

from memorydraft.evaluation.asd_suite import BENCHMARKS, SPLITS, sha

ROOT = Path(__file__).resolve().parents[1]
SEED = "MemoryDraft-asd-suite-v1"


def rank(record):
    return sha((SEED + ":" + record["id"]).encode())


def balanced(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    names = sorted(groups, key=lambda n: sha((SEED + str(n)).encode()))
    for name in names:
        groups[name].sort(key=rank)
    result = []
    for i in range(max(map(len, groups.values()))):
        result.extend(groups[name][i] for name in names if i < len(groups[name]))
    return result


def fixtures():
    rows = []
    for i in range(32):
        rng = random.Random(SEED + ":json:" + str(i))
        project = "project-" + sha((SEED + str(i)).encode())[:8]
        services = [dict(name=f"service-{j}-{rng.randrange(100,999)}", port=rng.randrange(1024,30000), enabled=bool(rng.randrange(2)))
                    for j in range(1 + i % 4)]
        owners = [f"owner-{rng.randrange(100,999)}" for _ in range(1 + i % 3)]
        expected = dict(project=project, revision=rng.randrange(1,100), region=rng.choice(["east","west","north"]),
                        owners=owners, services=services,
                        limits=dict(requests_per_second=rng.randrange(10,1000), timeout_ms=rng.randrange(100,10000)))
        facts = [f"Project: {project}", f"Revision: {expected['revision']}", f"Region: {expected['region']}",
                 "Owners in order: " + ", ".join(owners), f"Requests per second: {expected['limits']['requests_per_second']}",
                 f"Timeout in milliseconds: {expected['limits']['timeout_ms']}"]
        facts += [f"Service {j+1}: name={s['name']}; port={s['port']}; enabled={'yes' if s['enabled'] else 'no'}" for j,s in enumerate(services)]
        instruction = ("Convert the following facts into one JSON object. Return JSON only. Use exactly the keys "
                       "project (string), revision (integer), region (string), owners (ordered string array), "
                       "services (ordered array of objects with name:string, port:integer, enabled:boolean), "
                       "limits (object with requests_per_second:integer and timeout_ms:integer). Preserve the listed orders.\n\n"
                       + "\n".join(facts))
        rows.append(dict(id=f"json:{i}", benchmark="json", group_id=project, instruction=instruction, expected=expected,
                         source="Project-authored synthetic extraction fixture; not production data"))
    return rows


def prepare(sources, output):
    output.mkdir(parents=True, exist_ok=False)
    source_manifest = json.loads((sources / "manifest.json").read_text(encoding="utf-8"))
    for name, entry in source_manifest.items():
        if sha((sources / entry["filename"]).read_bytes()) != entry["sha256"]:
            raise ValueError("Changed downloaded source: " + name)
    gsm_path = ROOT / "artifacts/asd-attribution-gsm-source.jsonl"
    math_path = ROOT / "artifacts/asd-budget-math500-source.jsonl"
    if sha(gsm_path.read_bytes()) != "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14" or sha(math_path.read_bytes()) != "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132":
        raise ValueError("Changed pinned legacy source cache")
    gsm = [json.loads(s) for s in gsm_path.read_text(encoding="utf-8").splitlines()]
    maths = [json.loads(s) for s in math_path.read_text(encoding="utf-8").splitlines()]
    mbpp = [json.loads(s) for s in (sources / "mbpp.jsonl").read_text(encoding="utf-8").splitlines()]
    human = [json.loads(s) for s in gzip.decompress((sources / "humaneval.jsonl.gz").read_bytes()).decode().splitlines()]
    squad = json.loads((sources / "squad.json").read_text(encoding="utf-8"))["data"]
    pools = dict(gsm8k=[dict(r,id=f"gsm8k:{i}",benchmark="gsm8k",source_index=i) for i,r in enumerate(gsm) if i>=32],
                 math500=[dict(r,id=f"math500:{i}",benchmark="math500",source_index=i) for i,r in enumerate(maths) if i>=24],
                 mbpp=[dict(r,id=f"mbpp:{r['task_id']}",benchmark="mbpp") for r in mbpp if 19<=r["task_id"]<=510],
                 humaneval=[dict(r,id="humaneval:"+r["task_id"].split("/")[-1],benchmark="humaneval") for r in human if int(r["task_id"].split("/")[-1])>=16],
                 json=fixtures(), squad=[])
    for article in squad:
        choices = [dict(id="squad:"+q["id"],benchmark="squad",group_id=article["title"],context=p["context"],
                        question=q["question"],answers=list(dict.fromkeys(a["text"] for a in q["answers"])))
                   for p in article["paragraphs"] for q in p["qas"]]
        pools["squad"].append(min(choices,key=rank))
    files, ids, sizes = {}, {}, {}
    for b in BENCHMARKS:
        choices = balanced(pools[b],lambda r:r["subject"]) if b=="math500" else sorted(pools[b],key=rank)
        chosen = choices[:32]
        if len(chosen)!=32:
            raise ValueError("Insufficient candidates: "+b)
        for i,r in enumerate(chosen):
            r["split"] = "development" if i<8 else "validation" if i<16 else "reserve"
        raw=(json.dumps(chosen,ensure_ascii=False,indent=2)+"\n").encode()
        name=b+".json"; (output/name).write_bytes(raw); files[name]=sha(raw)
        ids[b]={s:[r["id"] for r in chosen if r["split"]==s] for s in SPLITS}; sizes[b]=len(pools[b])
    legacy=json.loads((ROOT/"data/research/asd-budget/records.json").read_text(encoding="utf-8"))["sources"]
    source_manifest.update(gsm8k={**legacy["gsm8k"],"sha256":sha(gsm_path.read_bytes())},math500=legacy["math500"])
    manifest=dict(schema=1,seed=SEED,files=files,ids=ids,sources=source_manifest,pool_sizes=sizes,
                  selection="SHA256-ranked source IDs, excluding observed ranges; MATH subject round-robin; one SQuAD question per distinct article; 8 development/8 validation/16 reserve per benchmark",
                  exposure="GSM source >=32; MATH >=24; MBPP official test19..510; HumanEval >=16 conservatively excludes early Spec-Bench overlap; no generated-outcome filtering")
    (output/"manifest.json").write_bytes((json.dumps(manifest,ensure_ascii=False,indent=2)+"\n").encode())
    for b in ("mbpp","humaneval","squad"):
        name=source_manifest[b]["license_file"]; shutil.copyfile(sources/name,output/name)
    shutil.copyfile(ROOT/"data/research/asd-attribution/MATH_SOURCE_LICENSE.txt",output/"MATH_SOURCE_LICENSE.txt")
    shutil.copyfile(ROOT/"data/research/asd-attribution/GSM8K_LICENSE.txt",output/"GSM8K_LICENSE.txt")
    (output/"ATTRIBUTION.md").write_text("# Selected benchmark records\n\nGSM8K and HumanEval: MIT. MBPP source repository: Apache-2.0. MATH-500: retain original-source MIT notice and existing provenance qualification. SQuAD records: CC BY-SA 4.0, https://rajpurkar.github.io/SQuAD-explorer/ ; selected and reformatted from the pinned source, original article titles retained. JSON fixtures are project-authored synthetic data. These notices apply to their respective records and do not grant a new license to the whole repository.\n",encoding="utf-8")
    print(json.dumps(dict(records=192,development=48,validation=48,reserve=96,ids=ids),ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=ROOT/"data/research/asd-suite")
    args=parser.parse_args(); prepare(args.sources,args.output)
