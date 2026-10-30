"""Package tracked source and small public evaluation inputs, never private assets."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts/branch-drift-source.zip')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    names = subprocess.check_output(['git','ls-files','-z','--','memorydraft','scripts','tests','configs',
        'data/research/asd-paper-extension','docs/BRANCH_DRIFT_PROTOCOL.md',
        'docs/BRANCH_DRIFT_OPTIMIZATION_PROTOCOL.md','docs/GATE_WIDTH_PROTOCOL.md','pyproject.toml',
        'results/asd-budget-tree/publication.json','results/asd-budget-tree/part-1/calibration.json',
        'results/asd-paper-extension/part-0/references.jsonl.gz',
        'results/asd-paper-extension/part-1/references.jsonl.gz'],cwd=root).decode().split('\0')
    path = root/args.output
    files = {}
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(n for n in names if n):
            raw = (root/name).read_bytes()
            files[name] = hashlib.sha256(raw).hexdigest()
            archive.writestr(name,raw)
        manifest = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            files=files,source_bytes='SHA-256 of packaged working-tree bytes')
        archive.writestr('BRANCH_DRIFT_SOURCE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    (root/'artifacts/BRANCH_DRIFT_SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(files=len(files),bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),commit=manifest['commit'])))


if __name__=='__main__':
    main()
