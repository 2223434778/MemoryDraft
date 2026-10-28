"""Rebuild public tables directly from the completed branch-drift evaluation."""
import argparse
from collections import defaultdict
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics

from memorydraft.evaluation.asd_paper import BENCHMARKS, quality_ok
from memorydraft.evaluation.branch_drift import ARMS, prefix_length, summarize

ROOT = Path(__file__).resolve().parents[1]


def change(value):
    return '—' if value is None else f'{(value-1)*100:+.2f}%'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('results/raw/branch-drift'))
    args = parser.parse_args()
    source = ROOT/args.input
    out = ROOT/'results/branch-drift'
    out.mkdir(parents=True,exist_ok=True)
    info = json.loads((source/'provenance.json').read_bytes())
    if info['status'] != 'completed':
        raise ValueError('Refuse to publish incomplete generation results')
    refs = json.loads((source/'references.json').read_bytes())
    rows = [json.loads(line) for line in (source/'rows.jsonl').read_text(encoding='utf-8').splitlines()]
    for row in rows:
        if row['arm'] not in ('CoverageTree-ASD','DriftTree-ASD'):
            continue
        drift=row['asd']['branch_drift']
        count=row['asd']['relaxed_tokens']
        if (drift.get('selected_exceptions',0)!=count or drift.get('selected_compared_edges',0)!=count
                or not 0 <= drift.get('selected_js_sum',0) <= drift['max_js']*count+1e-6):
            raise ValueError('Selected exception lacks a valid sibling JS check: '+row['request_id'])
    analysis = summarize(rows,refs)
    if analysis != json.loads((source/'analysis.json').read_bytes()):
        raise ValueError('Independent report reconstruction differs from server analysis')
    for name in ('analysis.json','provenance.json','cost-table.json','tasks.json'):
        shutil.copyfile(source/name,out/name)
    # Token-ID reference arrays are compressed to keep the review diff readable.
    if (out/'references.json').exists():
        (out/'references.json').unlink()
    for name in ('rows.jsonl','warmup.jsonl','references.json'):
        with (source/name).open('rb') as f, gzip.GzipFile(filename=str(out/(name+'.gz')),mode='wb',mtime=0) as z:
            shutil.copyfileobj(f,z)
    reward = source/'reward/analysis.json'
    reward_rows = json.loads(reward.read_bytes()) if reward.exists() else []
    if reward.exists():
        destination = out/'reward'
        destination.mkdir(exist_ok=True)
        for path in (source/'reward').iterdir():
            if path.is_file():
                shutil.copyfile(path,destination/path.name)
    ref_index = {r['request_id']:r for r in refs}
    strict = []
    for row in rows:
        if row['arm'] not in ('BudgetTree','DFlash'):
            continue
        expected = ref_index[row['request_id']]['token_ids'][:row['max_new_tokens']]
        if row['token_ids'] != expected:
            strict.append(dict(request_id=row['request_id'],stage=row['stage'],arm=row['arm'],
                first_difference=prefix_length(row['token_ids'],expected),
                interpretation='Retained cross-forward identity difference; not automatically attributed to numeric error'))
    audit = dict(timed_generations=len(rows),matched_inputs=len(refs),
        all_prompts_and_snapshots_match=True,all_request_ledgers_reconcile=True,
        all_selected_drift_exceptions_have_covered_bounded_js=True,
        minimum_recorded_proposal_horizon=min(b['proposed'] for r in rows for b in r['rounds']),
        strict_vs_frozen_ar_differences=strict,
        strict_tree_vs_linear_natural_differences=[])
    natural = {(r['request_id'],r['arm']):r for r in rows if r['stage']=='natural'}
    for identity in ref_index:
        if natural[identity,'BudgetTree']['token_ids'] != natural[identity,'DFlash']['token_ids']:
            audit['strict_tree_vs_linear_natural_differences'].append(identity)
    mmlu_cases=[]
    for identity,reference in ref_index.items():
        if reference['benchmark']!='mmlu':
            continue
        choices={}
        for arm in ARMS:
            match=re.match(r'^\s*\(?([ABCD])\b',natural[identity,arm]['text'])
            choices[arm]=match[1] if match else None
        mmlu_cases.append(dict(request_id=identity,choices=choices,
            expected=natural[identity,'ASD']['quality']['expected_answer']))
    audit['mmlu_leading_choice_diagnostic']=dict(
        rule='Same leading-choice parser as prior extension; diagnostic only, primary grades unchanged',
        all_methods_equal_per_request=all(len(set(r['choices'].values()))==1 for r in mmlu_cases),
        correct={arm:sum(r['choices'][arm]==r['expected'] for r in mmlu_cases) for arm in ARMS},
        cases=mmlu_cases)
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    reviews=[]
    for row in rows:
        if row['stage']!='natural' or row['arm']!='DriftTree-ASD' or row['benchmark']!='humaneval':
            continue
        reference=ref_index[row['request_id']]
        if quality_ok(reference['quality']) is True and quality_ok(row['quality']) is False:
            reviews.append(dict(request_id=row['request_id'],candidate_quality=row['quality'],
                ar_quality=reference['quality'],asd=row['asd'],
                first_difference=prefix_length(row['token_ids'],reference['token_ids']),
                diff=list(difflib.unified_diff(reference['text'].splitlines(),row['text'].splitlines(),
                    fromfile='AR',tofile='Drift',n=2))))
    (out/'case-review.json').write_text(json.dumps(reviews,indent=2)+'\n',encoding='utf-8')
    lines = ['# 分支漂移门控：七任务实测结果','',
        '六方法同窗自然EOS比较；另五方法测128/EOS前缀。负数表示时延减少。配对比值按题平均；单次计时。各列均直接配对，不能将两个平均比值相除来推导方法间的比值。',
        '冻结旧AR输出仅供质量和轨迹参考，不用于本轮AR时延加速比。MT第二轮共享冻结AR首轮历史。','',
        '|任务|输入数|Drift vs 原树ASD|vs ASD|vs 严格树|vs Coverage|正确：原树ASD → Drift|AR逐token相同：原树ASD → Drift|',
        '|---|---:|---:|---:|---:|---:|---|---|']
    for benchmark in BENCHMARKS:
        family = analysis['families'][benchmark]
        a,b = family['BudgetTree-ASD']['natural'],family['DriftTree-ASD']['natural']
        quality = f"{a['correct']} → {b['correct']} / {b['scored']}" if b['scored'] else '开放质量另列'
        lines.append('|'+ '|'.join([benchmark,str(b['n']),change(b['BudgetTree-ASD']['ratio']),
            change(b['ASD']['ratio']),change(b['BudgetTree']['ratio']),change(b['CoverageTree-ASD']['ratio']),
            quality,f"{a['identical_ar']} → {b['identical_ar']} / {b['n']}"])+'|')
    lines += ['','## 等128-token前缀','',
        '仅比较双方实际生成128token的请求；自然EOS短回答不强制续写。各对照有效题数可以不同。','',
        '|任务|Drift vs 原树ASD|有效题数|vs ASD|有效题数|vs 严格树|有效题数|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for benchmark in BENCHMARKS:
        v=analysis['families'][benchmark]['DriftTree-ASD']['prefix']
        values=[]
        for baseline in ('BudgetTree-ASD','ASD','BudgetTree'):
            values.extend([change(v[baseline]['ratio']),str(v[baseline]['n'])])
        lines.append('|'+ '|'.join([benchmark,*values])+'|')
    lines += ['','## 自然质量、长度与轮数','',
        '数学沿用既定评分与有限标量等价补充；代码使用原提供测试，MMLU采用仅字母输出合同。开放任务不计正确率。','',
        '|任务|方法|正确/可评分|输出token|验证轮|近似token|与AR相同|上限截断|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for benchmark in BENCHMARKS:
        reference_rows=[r for r in refs if r['benchmark']==benchmark]
        reference_quality=[quality_ok(r['quality']) for r in reference_rows]
        scored=sum(q is not None for q in reference_quality)
        correct=sum(q is True for q in reference_quality)
        lines.append('|'+ '|'.join([benchmark,'AR冻结参考',f'{correct}/{scored}' if scored else '—',
            str(sum(len(r['token_ids']) for r in reference_rows)),'历史参考，不作同窗计时','0',
            str(len(reference_rows)),str(sum(r['finish_reason']=='length' for r in reference_rows))])+'|')
        for arm in ARMS:
            v=analysis['families'][benchmark][arm]['natural']
            lines.append('|'+ '|'.join([benchmark,arm,f"{v['correct']}/{v['scored']}" if v['scored'] else '—',
                str(v['tokens']),str(v['calls']),str(v['relaxed']),str(v['identical_ar']),str(v['truncated'])])+'|')
    lines += ['','## 分支门控覆盖','',
        '下表统计局部通过原ASD门槛的树边，包含最终未被提交的旁支；不能当作实际输出token的比例。',
        'Coverage与Drift使用相同构树和停止/真实兄弟覆盖要求，后续生成状态可因JS门控而分化。','',
        '|任务|方法|局部候选边|缺失兄弟|停止拦截|比较数|JS拒绝|平均JS|选中例外|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for benchmark in BENCHMARKS:
        for arm in ('CoverageTree-ASD','DriftTree-ASD'):
            v=analysis['families'][benchmark][arm]['natural']['drift']
            compared=v.get('compared_edges',0)
            mean=v.get('compared_js_sum',0)/compared if compared else 0.
            lines.append('|'+ '|'.join([benchmark,arm,*[str(int(v.get(k,0))) for k in
                ('eligible_edges','missing_reference','stop_blocked','compared_edges','drift_blocked')],
                f'{mean:.4f}',str(int(v.get('selected_exceptions',0)))])+'|')
    lines += ['','## 开放任务独立奖励偏好','',
        '固定Skywork-Reward-V2-Qwen3-8B，仅为独立偏好代理，非官方MT-Bench/AlpacaEval，也非正确率。相同文本按平局。','',
        '|任务|方法|对照|胜/平/负|偏好率|','|---|---|---|---:|---:|']
    for row in reward_rows:
        rate='—' if row['preference_rate'] is None else f"{row['preference_rate']*100:.2f}%"
        lines.append(f"|{row['benchmark']}|{row['arm']}|{row['baseline']}|{row['wins']}/{row['ties']}/{row['losses']}|{rate}|")
    if not reward_rows:
        lines.append('|待评分|—|—|—|—|')
    lines += ['','## 执行与复核','',
        f"共{len(rows)}条新计时，{len(refs)}条冻结AR参考。GPU加载/预热与离线奖励评分单列；在线构树、JS、传输、验证、KV全部计入wall。",
        f"严格方法与冻结AR身份差异{len(strict)}条；严格树与同窗DFlash自然差异{len(audit['strict_tree_vs_linear_natural_differences'])}题。详见audit.json，不将未诊断差异自动称数值误差。",
        '原数据以前已用于研究，本轮不称新保留集。完整回答和前缀指标分别报告；相同AR token数不是语义质量指标。',
        '原始小型公开输出、账本与轮级摘要见[results/branch-drift](../results/branch-drift/)，无完整logits/KV/权重。',
        '复核命令：`python -m scripts.write_branch_drift_report --input results/raw/branch-drift`。实现与选择规则见[协议](BRANCH_DRIFT_PROTOCOL.md)。','']
    (ROOT/'docs/BRANCH_DRIFT_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    files = {str(p.relative_to(out)).replace('\\','/'):dict(bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(out.rglob('*'))
        if p.is_file() and p.name!='publication.json' and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    (out/'publication.json').write_text(json.dumps(dict(files=files),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(rows=len(rows),references=len(refs),files=len(files),
        strict_ar_differences=len(strict),strict_tree_linear_differences=len(audit['strict_tree_vs_linear_natural_differences']),
        mmlu_leading_choices_equal=audit['mmlu_leading_choice_diagnostic']['all_methods_equal_per_request'])))


if __name__=='__main__':
    main()
