"""Rebuild the frozen seven-task speed, quality and compute comparison."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import statistics

from memorydraft.evaluation.asd_paper import ARMS,BENCHMARKS,STUDY,aggregate,load_records
from memorydraft.evaluation.asd_joint_tree import audit_allocation
from scripts.diagnose_transport_online import write

ROOT=Path(__file__).resolve().parents[1]


def read_rows(directory,name):
    result=[]
    for shard in (0,1):
        path=directory/f'part-{shard}/{name}.jsonl'
        raw=path.read_bytes() if path.is_file() else gzip.decompress(path.with_suffix('.jsonl.gz').read_bytes())
        result.extend(json.loads(line) for line in raw.decode().splitlines())
    return result


def judge_stats(directory):
    path=directory/'local-judge.jsonl'
    raw=path.read_bytes() if path.is_file() else gzip.decompress(path.with_suffix('.jsonl.gz').read_bytes())
    rows=[json.loads(x) for x in raw.decode().splitlines()];groups=defaultdict(list)
    if len(rows)!=192:raise ValueError('Incomplete local judge matrix')
    expected={r['id'] for r in load_records(ROOT) if r['benchmark'] in ('mtbench','alpaca')}
    if {(r['request_id'],r['arm'],r['order']) for r in rows}!={(i,a,o) for i in expected for a in ARMS if a!='ASD' for o in (0,1)}:
        raise ValueError('Changed judge task/method coverage')
    cells=defaultdict(dict)
    for r in rows:
        key=r['request_id'],r['arm']
        if r['order'] in cells[key]:raise ValueError('Duplicate judge order')
        cells[key][r['order']]=r
    for (identity,arm),orders in cells.items():
        if set(orders)!={0,1}:raise ValueError('Missing judge order swap')
        a,b=orders[0],orders[1]
        complete=a['candidate_score'] is not None and b['candidate_score'] is not None
        groups[a['benchmark'],arm].append(dict(request_id=identity,score=(a['candidate_score']+b['candidate_score'])/2 if complete else None,
            consistent=a['candidate_score']==b['candidate_score'] if complete else None,
            identical=a['status']==b['status']=='identical'))
    return {benchmark:{arm:dict(requests=len(values),scored=sum(v['score'] is not None for v in values),
        preference=statistics.mean(v['score'] for v in values if v['score'] is not None) if any(v['score'] is not None for v in values) else None,
        identical=sum(v['identical'] for v in values),order_consistent=sum(v['consistent'] is True for v in values),samples=values)
        for (b,arm),values in groups.items() if b==benchmark} for benchmark in ('mtbench','alpaca')}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,default=ROOT/f'results/{STUDY}')
    args=parser.parse_args();out=args.input
    rows,refs=read_rows(out,'rows'),read_rows(out,'references')
    result=aggregate(rows,refs,load_records(ROOT))
    if len(rows)!=4980 or len(refs)!=249:raise ValueError('Frozen paper coverage incomplete')
    resources=[]
    for shard in (0,1):
        p=json.loads((out/f'part-{shard}/provenance.json').read_text())
        if (p['status']!='complete' or not p['original_unchanged']
                or p['compiler']['graph_count']!=p['compiler_after_warmup']['graph_count']):
            raise ValueError('Incomplete worker or new formal graphs')
        resources.append(p)
    rollout=read_rows(out,'rollout')
    if len(rollout)!=40:raise ValueError('Incomplete realized-dialogue rollout')
    result['rollout_allocation_audit']=audit_allocation(rollout)
    if any(r['asd'] and not 0<=r['asd']['spent']<=8 for r in rollout):raise ValueError('Rollout budget exceeded')
    dialogue={}
    for arm in ARMS:
        selected=[r for r in rollout if r['arm']==arm]
        if len(selected)!=8:raise ValueError('Incomplete dialogue arm')
        dialogue[arm]=dict(conversations=8,seconds=sum(r['first_latency_seconds']+r['latency_seconds'] for r in selected),
            tokens=sum(len(r['first_token_ids'])+len(r['token_ids']) for r in selected),
            second_turn_truncated=sum(r['finish_reason']=='length' for r in selected))
    for identity in {r['conversation_id'] for r in rollout}:
        pair={r['arm']:r for r in rollout if r['conversation_id']==identity}
        if pair['DFlash']['token_ids']!=pair['BudgetTree']['token_ids']:raise ValueError('Strict dialogue mismatch')
    result.update(resources=resources,local_judge=judge_stats(out),realized_dialogue=dialogue)
    judge=json.loads((out/'local-judge-provenance.json').read_text())
    if judge['status']!='complete':raise ValueError('Local judge incomplete')
    result['local_judge_provenance']=judge
    write(out/'analysis.json',result)
    def pct(value):return '—' if value is None else f'{(value-1)*100:+.2f}%'
    lines=['# ASD论文七任务：五方法扩样本比较','',
        '241个未使用源任务，MT-Bench展开为249个匹配输入请求；4980计时+249 AR，另40次自有历史第二轮。',
        '自然EOS各2次、128/EOS各1次、固定256/EOS忽略各1次。全部在线成本计入wall，成本表/权重/ASD B8/g.25/m2冻结。',
        'DFlash底座、A6000/FP32/RMSNorm；不是论文DSpark/L20和原始完整提示/样本协议的精确复现。','',
        '|任务|方法|自然相对ASD|自然相对第一阶段组合|质量正确/已评分|截断|等128相对ASD（N）|固定256相对ASD|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for benchmark in BENCHMARKS:
        for arm in ARMS:
            v=result['families'][benchmark][arm];n,p=v['natural'],v['prefix']
            quality=f"{n['correct']}/{n['scored']}" if n['scored'] else '另评'
            fixed=pct(v['fixed']['ASD']['ratio']) if 'fixed' in v else '—'
            lines.append(f"|{benchmark}|{arm}|{pct(n['ASD']['ratio'])}|{pct(n['BudgetTree-ASD']['ratio'])}"
                f"|{quality}|{n['truncated']}|{pct(p['ASD']['ratio'])}（{p['ASD']['n']}）|{fixed}|")
    lines+=['','负数为时延降低；速度表用逐请求比值的题等权均值；另在JSON保存总token/总wall TPS、长度与请求区间。',
        'MT-Bench主速度表第二轮共享严格AR的首轮回答；下表的质量审查使用各方法接续自身首轮回答。',
        '代码标准测试不进提示，MBPP可用challenge另记；数学保守主评分、有限标量等价和待复核单列。','',
        '|固定256计算吞吐|方法|token/s|相对ASD TPS变化|',
        '|---|---|---:|---:|']
    for benchmark in BENCHMARKS:
        family=result['families'][benchmark];baseline=family['ASD']['fixed']['tps']
        for arm in ARMS:
            value=family[arm]['fixed']['tps']
            lines.append(f'|{benchmark}|{arm}|{value:.3f}|{pct(value/baseline)}|')
    lines+=['','上表TPS变化正数更快。固定256包含EOS后续写的计算，不能直接当自然请求质量或响应收益。','',
        '|开放任务|候选相对ASD|本地偏好（含平局）|已评/任务|相同输出|换序一致|',
        '|---|---|---:|---:|---:|---:|']
    for benchmark,arms in result['local_judge'].items():
        for arm,v in arms.items():
            score='—' if v['preference'] is None else f"{v['preference']*100:.2f}%"
            lines.append(f"|{benchmark}|{arm}|{score}|{v['scored']}/{v['requests']}|{v['identical']}|{v['order_consistent']}|")
    lines+=['','本地盲评使用同一Qwen3-8B Target、A/B双向换序和独立rubric；属于有偏好的辅助诊断，',
        '不是独立GPT-4裁判、官方MT-Bench分数或AlpacaEval胜率。输入超过4096 token不截断裁判材料，标为未评。',
        '开放任务不伪造准确率；模型加载、预热、冷编译和离线裁判成本另列。','']
    (out/'report.md').write_bytes('\n'.join(lines).encode())
    print(json.dumps(dict(timed=len(rows),ar=len(refs),realized_second_turns=len(rollout),all_execution_checks=True)))


if __name__=='__main__':main()
