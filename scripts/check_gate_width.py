"""Verify published width-study evidence without a GPU or model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', action='store_true', help='Save final-check.json; refresh publication afterward')
    args = parser.parse_args()
    out = ROOT/'results/gate-width'
    publication = json.loads((out/'publication.json').read_bytes())
    for name, digest in publication['payloads'].items():
        if sha(out/name) != digest:
            raise ValueError('Payload SHA differs: '+name)
    if sha(ROOT/'scripts/report_gate_width.py') != publication['reporter_sha256']:
        raise ValueError('Use the published reporter source version')
    if sha(ROOT/publication['report']['path']) != publication['report']['sha256']:
        raise ValueError('Published report differs')
    audit = json.loads((out/'audit.json').read_bytes())
    if audit['timings'] != 792 or audit['matched_inputs'] != 66 or audit['strict_ar_differences']:
        raise ValueError('Matrix size or strict AR identity check failed')
    old = audit['previous_baseline_identity']
    if old['pairs'] != 396 or old['differences']:
        raise ValueError('Unchanged baseline output differs from previous study')
    execution = json.loads((out/'execution.json').read_bytes())
    manifest = json.loads((out/'source-manifest.json').read_bytes())
    count = len(manifest['files'])
    if (execution['status'] != 'completed' or execution['source_files_verified_before'] != count
            or execution['source_files_verified_after'] != count
            or any(execution['restore_exit_codes'].values())):
        raise ValueError('Source identity or service restoration failed')
    frozen = {'main': '27e36199f8a65acdb9a9c42ccb744ddfa9a9402b',
              'codex/infra-interview-v0.4': 'ea04290366b3caa809aba93b9417c7e7a148ce48',
              'infra-v0.4-interview^{}': 'ea04290366b3caa809aba93b9417c7e7a148ce48'}
    for ref, digest in frozen.items():
        actual = subprocess.check_output(['git', 'rev-parse', ref], cwd=ROOT, text=True).strip()
        if actual != digest:
            raise ValueError('Frozen reference changed: '+ref)
    links = 0
    for doc in ROOT.glob('docs/GATE_WIDTH*.md'):
        for link in re.findall(r'\]\(([^)]+)\)', doc.read_text(encoding='utf-8')):
            if not link.startswith(('https:', 'http:', '#')):
                if not (doc.parent/link.split('#')[0]).exists():
                    raise ValueError('Broken relative link: '+link)
                links += 1
    before = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    subprocess.run([sys.executable, '-m', 'scripts.report_gate_width', '--input', 'results/gate-width'], cwd=ROOT, check=True)
    if before != {p.name: sha(p) for p in out.iterdir() if p.is_file()}:
        raise ValueError('Publication rebuild changed bytes')
    result = dict(timings=792, matched_inputs=66, previous_baseline_identity_pairs=396,
        strict_ar_identity=66, feedback_history_valid=True, source_files=count,
        source_commit=manifest['commit'], relative_links=links, payloads_valid=True,
        rebuild_byte_identical=True, frozen_refs_unchanged=True, services_restored=True)
    if args.record:
        (out/'final-check.json').write_bytes((json.dumps(result, indent=2)+'\n').encode())
    print(json.dumps(result))


if __name__ == '__main__':
    main()
