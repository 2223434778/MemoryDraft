"""Review large reward shifts without converting preferences into accuracy."""
import argparse
import json
from pathlib import Path

from memorydraft.evaluation.asd_paper_metrics import message_sha,open_units
from scripts.write_asd_paper_metrics_report import rewards

ROOT=Path.cwd()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=ROOT/'results/asd-paper-metrics')
    args=parser.parse_args();result=rewards(args.input)
    if result['status']!='complete':raise ValueError('Reward audit requires complete scores')
    units={u['request_id']:u for u in open_units(ROOT)}
    delta={(r['request_id'],r['arm']):r['reward_delta'] for r in result['samples']}
    notes={
        'mtbench:109':'The supplied direction reference and anonymous review agree: ASD ends north instead of south, while both new combinations end south. The positive reward shifts align with this factual improvement.',
        'mtbench:130':'ASD gives expected O(n+m) hashing in the second turn. Strict DFlash and both combinations retain sorting. Reward nevertheless ranks strict DFlash above ASD; the trained reward proxy can miss an explicit algorithmic requirement.',
        'mtbench:138':'All methods fail the all-uppercase phone-name constraint. Joint alone closes the second JSON object; it still scores below ASD. Preference and schema compliance must be reported separately.',
        'mtbench:145':'ASD and BudgetTree-ASD both mention additional carbonate reagents, so the large negative reward delta is not by itself proof that the combination lost that valid alternative. Joint describes direct reversal using CO2/water/pressure with insufficient chemical conditions. Carbonate precipitation and acid-dependent solubility require separate factual review; no automatic global correctness grade is assigned.',
        'alpaca:32069':'All methods explicitly interpret bi-annual as twice a year and provide an argument. The -5.125 reward delta for the two identical new-combination answers is a preference difference, not a demonstrated change to once every two years or a calibrated accuracy drop.',
        'alpaca:14689':'All completions discuss hybrid deep RL as well as a comparison. Their tables contain broad supervised/static-data simplifications. Do not attribute all reward difference to a newly introduced factual failure without a task-specific rubric.',
        'alpaca:653':'The lists differ in ordering, repeated items and terminology. No ground-truth item-by-item correctness metric was defined; the reward difference remains a preference observation.',
    }
    cases=[]
    for identity,note in notes.items():
        arms={}
        for arm,messages in units[identity]['answers'].items():
            text='\n\n'.join(m['content'] for m in messages if m['role']=='assistant')
            if identity=='alpaca:32069':assert 'twice a year' in text.lower()
            arms[arm]=dict(messages_sha256=message_sha(messages),assistant_characters=len(text),
                reward_delta_vs_asd=delta[identity,arm],
                mentions_sodium_carbonate=('sodium carbonate' in text.lower()) if identity=='mtbench:145' else None)
        cases.append(dict(request_id=identity,review=note,arms=arms))
    candidates=[r for r in result['samples'] if r['arm'] in ('BudgetTree-ASD','JointTree-ASD')]
    top=sorted(candidates,key=lambda r:(-abs(r['reward_delta']),r['request_id'],r['arm']))[:8]
    audit=dict(scope='Post-hoc review of largest new-method reward shifts and objective-control counterexamples; all original grades and reward scores unchanged.',
        top_eight_absolute_shifts=top,cases=cases,
        chemistry_sources=[
            'https://openstax.org/books/chemistry-2e/pages/15-3-coupled-equilibria',
            'https://www.colorado.edu/lab/lecture-demo-manual/general-reactions/s130-chemical-rxns-precipitation-cacl2-na2co3-caco3/s130-prep-notes'],
        conclusion='A trained pointwise model passed basic controls and supplied complete preferences. It still has domain and style limitations; neither reward wins nor losses are formal task accuracy.')
    (ROOT/'results/asd-paper-metrics/reward-case-audit.json').write_bytes((json.dumps(audit,indent=2)+'\n').encode())
    print(json.dumps(dict(reviewed_task_units=len(cases),large_shifts=len(top),frozen_scores_unchanged=True)))


if __name__=='__main__':main()
