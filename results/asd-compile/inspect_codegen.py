"""Read-only postrun summary; never export compiler cache source or tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True, help="Original server execution root")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
out = []
for shard in (0, 1):
    for path in (args.root/f"artifacts/inductor-{shard}").rglob("*.py"):
        text = path.read_text()
        if "async_compile.triton" not in text:
            continue
        names = re.findall(r"(?m)^([A-Za-z0-9_]+) = async_compile\.triton\(", text)
        out.append(dict(shard=shard, file=path.relative_to(args.root).as_posix(),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        triton_kernels=[dict(name=n, call_sites=text.count(n+".run(")) for n in names],
                        external_mm_call_count=sum("extern_kernels.mm(" in line for line in text.splitlines())))
args.output.write_bytes((json.dumps(out, indent=2)+"\n").encode())
