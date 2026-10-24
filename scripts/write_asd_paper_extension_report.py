"""Reconstruct the independent cross-task and budget-matched control tables."""
import argparse
import json
from pathlib import Path
import statistics

from memorydraft.evaluation.asd_paper import BENCHMARKS
from memorydraft.evaluation.asd_paper_extension import ARMS,STUDY,aggregate,load_records,read_rows
from scripts.diagnose_transport_online import write

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/f'results/{STUDY}');args=parser.parse_args();out=args.input
    result=aggregate(read_rows(out,'rows'),read_rows(out,'references'),load_records(ROOT),read_rows(out,'frontier'))
    resources=[]
    for shard in (0,1):
        p=json.loads((out/f'part-{shard}/provenance.json').read_bytes())
        if p['status']!='complete' or not p['original_unchanged']:raise ValueError('Incomplete extension worker')
        if p['compiler']['graph_count']!=p['compiler_after_warmup']['graph_count']:raise ValueError('New formal graphs')
        resources.append(p)
    rollout=read_rows(out,'rollout');expected={r['id'] for r in load_records(ROOT) if r['benchmark']=='mtbench'}
    if {(r['conversation_id'],r['arm']) for r in rollout}!={(i,a) for i in expected for a in ARMS} or len(rollout)!=len(expected)*len(ARMS):
        raise ValueError('Incomplete realized-dialogue matrix')
    result.update(resources=resources,realized_dialogue={a:dict(conversations=len(expected),
        seconds=sum(r['first_latency_seconds']+r['latency_seconds'] for r in rollout if r['arm']==a),
        tokens=sum(len(r['first_token_ids'])+len(r['token_ids']) for r in rollout if r['arm']==a)) for a in ARMS})
    write(out/'analysis.json',result)
    pct=lambda v:'—' if v is None else f'{(v-1)*100:+.2f}%'
    lines=['# ASD七任务独立扩样与预算匹配消融','',
        f"{len(load_records(ROOT))}源任务，{result['ar_generations']}匹配请求；{result['raw_timed_generations']}正式主计时、"
        f"{result['frontier_generations']}控制生成、{len(rollout)}自身历史第二轮。",
        '四组共同Target/Draft、FP32/RMSNorm/native16；未修改主方法参数。固定计时每请求各一组strict–candidate–strict；自然两次、prefix128一次。',
        '固定生成量与自然质量分列。计时重复不同于论文四次，不冒称完整原协议复现；所有漂移样本保留。','',
        '|任务|方法|自然时延vs ASD|共同128时延vs ASD（N）|固定TPS|固定TPS增益vs ASD|自然质量|截断|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for benchmark in BENCHMARKS:
        family=result['families'][benchmark]
        for arm in ARMS:
            v=family[arm];n=v['natural'];p=v['prefix'];f=v['fixed'];quality=f"{n['correct']}/{n['scored']}" if n['scored'] else '另评'
            lines.append(f"|{benchmark}|{arm}|{pct(n['ASD']['ratio'])}|{pct(p['ASD']['ratio'])}（{p['ASD']['n']}）|"
                f"{f['tps']:.3f}|{pct(f['tps']/family['ASD']['fixed']['tps'])}|{quality}|{n['truncated']}|")
    lines+=['','自然时延为逐题配对比值均值；固定TPS为总token/总wall。MMLU主质量为只答字母的输出合同。',
        f"相邻严格控制漂移>3%：{result['fixed_drift_over_3pct']}/{len(result['fixed_drift'])}；未据此删题。",'',
        '## 预算匹配的速度—质量曲线','',
        '各族前四源任务预先冻结。B0/B8复用主实验首重复，其他配置各一次自然/固定生成；属于诊断切片，不用来重新调主方法。','',
        '|任务|配置|方法|请求|自然时延vs普通ASD|固定TPS|质量|截断|',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for row in result['frontier']:
        quality=f"{row['correct']}/{row['scored']}" if row['scored'] else '另评'
        lines.append(f"|{row['benchmark']}|{row['control']}|{row['arm']}|{row['requests']}|{pct(row['natural_vs_asd'])}|"
                     f"{row['fixed_tps']:.3f}|{quality}|{row['truncated']}|")
    lines+=['','开放任务由独立训练奖励模型另外审查；该代理不等于官方GPT-4 MT-Bench评分或AlpacaEval。',
        '旧七任务结果保持，只发布本轮全部正负结果；不将EOS后计算吞吐当作真实短请求收益。','']
    (out/'report.md').write_bytes('\n'.join(lines).encode())
    print(json.dumps(dict(raw_timed=result['raw_timed_generations'],ar=result['ar_generations'],
        frontier=result['frontier_generations'],rollout=len(rollout),all_execution_checks=True)))


if __name__=='__main__':main()
