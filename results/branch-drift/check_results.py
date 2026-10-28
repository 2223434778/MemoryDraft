import gzip,hashlib,json,re,subprocess
from pathlib import Path
from memorydraft.evaluation.branch_drift import summarize

root=Path(__file__).resolve().parents[2];out=root/'results/branch-drift'
publication=json.loads((out/'publication.json').read_bytes())['files']
for name,entry in publication.items():
    raw=(out/name).read_bytes()
    assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
rows=[json.loads(line) for line in gzip.decompress((out/'rows.jsonl.gz').read_bytes()).decode().splitlines()]
refs=json.loads(gzip.decompress((out/'references.json.gz').read_bytes()))
analysis=summarize(rows,refs)
assert analysis==json.loads((out/'analysis.json').read_bytes())
assert len(rows)==1639 and len(refs)==149
lookup={(r['request_id'],r['arm']):r for r in rows if r['stage']=='natural'}
for ref in refs:
    for arm in ('BudgetTree','DFlash'):
        row=lookup[ref['request_id'],arm]
        assert all(row[k]==ref[k] for k in ('token_ids','text','finish_reason','prompt_sha256','snapshot_id'))
for row in rows:
    assert all(b.get('tree_nodes',0)<=32 for b in row['rounds'])
    if row['asd']:
        assert all(b['relaxed_tokens']<=2 for b in row['rounds'])
overview=json.loads((out/'overview.json').read_bytes())
assert [overview['methods'][a]['identical_ar'] for a in ('BudgetTree-ASD','CoverageTree-ASD','DriftTree-ASD')]==[60,75,106]
reward=json.loads((out/'reward/provenance.json').read_bytes())
assert reward['status']=='completed' and reward['rows']==reward['scored_rows']==192 and reward['controls_passed']==8
assert reward['scorer_sha256']==hashlib.sha256((root/'scripts/run_branch_drift_reward.py').read_bytes()).hexdigest()
execution=json.loads((out/'execution.json').read_bytes())
assert execution['status']=='completed' and execution['source_files_verified_before']==execution['source_files_verified_after']==498
assert all(v==0 for v in execution['restore_exit_codes'].values())
expected=['27e36199f8a65acdb9a9c42ccb744ddfa9a9402b','ea04290366b3caa809aba93b9417c7e7a148ce48','ea04290366b3caa809aba93b9417c7e7a148ce48']
actual=subprocess.check_output(['git','rev-parse','main','codex/infra-interview-v0.4','infra-v0.4-interview^{}'],text=True).splitlines()
assert actual==expected
links=0
for name in ('docs/BRANCH_DRIFT_FINDINGS.md','docs/BRANCH_DRIFT_RESULTS.md','docs/BRANCH_DRIFT_PROTOCOL.md'):
    path=root/name
    for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf8')):
        if '://' not in target and not target.startswith('#'):
            assert (path.parent/target.split('#')[0]).exists(),target
            links+=1
result=dict(new_timed_generations=1639,matched_inputs=149,reward_rows=192,
    relative_links=links,analysis_rebuilt_from_published_gzip=True,
    strict_tokens_text_stop_input_all_match_ar=True,node_cap_and_exception_cap_valid=True,
    source_before_after_valid=True,services_restored=True,frozen_refs_unchanged=True)
(out/'final-check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(result,published_payloads=len(publication))))
