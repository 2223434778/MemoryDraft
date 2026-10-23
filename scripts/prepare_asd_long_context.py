"""Use complete official train demonstrations, never an evaluated answer in input."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import shutil

from memorydraft.evaluation.long_context import LEVELS, STUDY, prompt, sha
from memorydraft.runtime.contracts import GenerationRequest
from scripts.prepare_asd_adaptation import SOURCE_SHA, normalized

ROOT = Path(__file__).resolve().parents[1]


def build(tasks, demonstrations, tokenizer):
    items = []
    for task in tasks:
        options = []
        for count in range(1,len(demonstrations)+1):
            messages = ({"role":"user","content":prompt(task["question"],demonstrations[:count])},)
            ids = tokenizer.apply_chat_template(list(messages),add_generation_prompt=True,enable_thinking=False)
            options.append((len(ids),count,messages,sha(json.dumps(ids).encode())))
            if len(ids) > 5500:break
        for level in LEVELS:
            length,count,messages,digest = min(options,key=lambda x:(abs(x[0]-level),x[1]))
            if not .75*level <= length <= 1.25*level:raise ValueError("Cannot reach requested size using complete demonstrations")
            request = GenerationRequest(f"long-{task['id']}-{level}",f"long-{task['id']}-{level}",1,messages,4096)
            items.append(dict(task_id=task["id"],level=level,prompt_tokens=length,prompt_sha256=digest,
                              examples=count,demonstration_ids=[d["id"] for d in demonstrations[:count]],request=asdict(request)))
    return items


def prepare(source, test_source, output, tokenizer):
    from scripts.prepare_asd_adaptation_followup import SOURCE_SHA as TEST_SHA
    if sha(source.read_bytes()) != SOURCE_SHA or sha(test_source.read_bytes()) != TEST_SHA:
        raise ValueError("Changed pinned official GSM train/test data")
    old = json.loads((ROOT/"data/research/asd-adaptation-followup/records.json").read_text(encoding="utf-8"))
    tasks = [r for r in old if r["study_split"] == "independent"][:8]
    excluded = {normalized(json.loads(line)["question"]) for line in test_source.read_text(encoding="utf-8").splitlines()}
    excluded.update(normalized(r["question"]) for r in json.loads((ROOT/"data/research/asd-adaptation/questions.json").read_text(encoding="utf-8")))
    pool = [dict(id=f"gsm8k-train:{i}",question=r["question"],answer=re.sub(r"<<[^>]*>>","",r["answer"]))
            for i,line in enumerate(source.read_text(encoding="utf-8").splitlines()) for r in [json.loads(line)]
            if normalized(r["question"]) not in excluded]
    pool.sort(key=lambda r:sha(("MemoryDraft-long-context-demonstrations-v1:"+r["id"]).encode()))
    demonstrations = pool[:48]
    if len(demonstrations) != 48 or len({normalized(r["question"]) for r in demonstrations}) != 48:
        raise ValueError("Insufficient unique demonstrations")
    items = build(tasks,demonstrations,tokenizer)
    output.mkdir(parents=True,exist_ok=False)
    for name,value in (("tasks.json",tasks),("demonstrations.json",demonstrations),("requests.json",items)):
        (output/name).write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+"\n").encode())
    shutil.copyfile(ROOT/"data/research/asd-attribution/GSM8K_LICENSE.txt",output/"GSM8K_LICENSE.txt")
    info = dict(schema=1,source_commit="3101c7d5072418e28b9008a6636bde82a006892c",source_train_sha256=SOURCE_SHA,source_test_sha256=TEST_SHA,
                task_selection="First eight independent records in the frozen followup input order; already observed tasks, new prompt conditions",
                demonstration_selection="SHA256 order; exclude complete official test and adaptation training/calibration questions; remove calculator annotations",
                tokenization="Qwen3-8B prepared local tokenizer; enable_thinking=False; complete nested demonstration prefixes, nearest input token length",
                levels=LEVELS,tasks=[r["id"] for r in tasks],demonstrations=[r["id"] for r in demonstrations],
                files={p.name:sha(p.read_bytes()) for p in sorted(output.iterdir())})
    (output/"manifest.json").write_bytes((json.dumps(info,indent=2)+"\n").encode())
    print(json.dumps(dict(tasks=len(tasks),requests=len(items),sizes={str(level):[r["prompt_tokens"] for r in items if r["level"]==level] for level in LEVELS})))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--test-source",type=Path,required=True)
    parser.add_argument("--tokenizer",type=Path,default=ROOT/"models/qwen3-8b")
    parser.add_argument("--output",type=Path,default=ROOT/f"data/workloads/{STUDY}")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    prepare(args.source,args.test_source,args.output,AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=False))
