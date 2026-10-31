"""Publish matched new-workload baselines without rewriting earlier experiments."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from scripts.report_expanded_suite import summarize
from scripts.run_expanded_baselines import ARMS,STRICT

ROOT=Path(__file__).resolve().parents[1]


def analyze(rows,prior):
    result=summarize(rows,arms=ARMS)
    indexed={(r['request_id'],r['arm']):r for r in rows}
    result['strict_identity_differences']=[]
    for r in rows:
        ref=indexed[r['request_id'],'AR']
        if r['arm'] in STRICT and any(r[k]!=ref[k] for k in ('token_ids','text','finish_reason')):
            result['strict_identity_differences'].append(dict(id=r['request_id'],arm=r['arm']))
    for benchmark,family in result['families'].items():
        for arm,v in family.items():
            selected=[r for r in rows if r['benchmark']==benchmark and r['arm']==arm]
            v['same_ar']=sum(all(r[k]==indexed[r['request_id'],'AR'][k] for k in ('token_ids','text','finish_reason')) for r in selected)
    old={(r['request_id'],r['arm']):r for r in prior}
    result['previous_output_differences']=[]
    for r in rows:
        key=r['request_id'],r['arm']
        if key not in old:continue
        if any(r[k]!=old[key][k] for k in ('prompt_sha256','snapshot_id','max_new_tokens')):
            raise ValueError('Changed prior input identity')
        if any(r[k]!=old[key][k] for k in ('token_ids','text','finish_reason')):
            result['previous_output_differences'].append(dict(id=key[0],arm=key[1]))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='results/raw/expanded-baselines')
    args=parser.parse_args()
    source=ROOT/args.input;out=ROOT/'results/expanded-baselines';out.mkdir(parents=True,exist_ok=True)
    info=json.loads((source/'provenance.json').read_bytes())
    if info['status']!='completed':raise ValueError('Unfinished experiment')
    raw=(source/'rows.jsonl').read_bytes() if (source/'rows.jsonl').exists() else gzip.decompress((source/'rows.jsonl.gz').read_bytes())
    rows=[json.loads(s) for s in raw.decode().splitlines()]
    if len(rows)!=info['expected']:raise ValueError('Missing cells')
    prior=[json.loads(s) for s in gzip.decompress((ROOT/'results/expanded-suite/rows.jsonl.gz').read_bytes()).decode().splitlines()]
    analysis=analyze(rows,prior)
    for r in rows:
        r['quality'].pop('metadata',None)
        r['rounds']=[{k:b[k] for k in ('advance','accepted','strict_accepted','relaxed_tokens','regret','tree_nodes','route') if k in b} for b in r['rounds']]
    (out/'rows.jsonl.gz').write_bytes(gzip.compress(('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n').encode(),mtime=0))
    for p in source.iterdir():
        if p.is_file() and p.name not in ('rows.jsonl','rows.jsonl.gz','tasks.json','analysis.json','publication.json') and p.resolve()!=(out/p.name).resolve():
            shutil.copyfile(p,out/p.name)
    (out/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n',encoding='utf-8')
    lines=['# 新增场景：AR、严格成本树与官方BASTION补全','',
        f"{analysis['requests']}固定请求、五组{analysis['generations']}自然生成。Qwen3-8B/DFlash、A6000、FP32/eager/RMSNorm。",'成本表复用同GPU已冻结的合成校准，算法与接受条件不变；本批计时单独比较。','',
        '|任务|方法|配对wall vs DFlash|配对wall vs 严格成本树|配对wall vs BASTION|AR/方法总wall|质量|AR一致|截断|token/s|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def pct(v):return f'{(v-1)*100:+.2f}%'
    for b,f in analysis['families'].items():
        for arm in ARMS:
            v=f[arm];q=f"{v['correct']}/{v['scored']}" if v['scored'] else (f"{v['score']*100:.2f}" if v['score'] is not None else '—')
            lines.append(f"|{b}|{arm}|{pct(v['DFlash']['paired_ratio'])}|{pct(v['BudgetTree']['paired_ratio'])}|{pct(v['BASTION']['paired_ratio'])}|{f['AR']['seconds']/v['seconds']:.3f}×|{q}|{v['same_ar']}/{v['n']}|{v['truncated']}|{v['tps']:.2f}|")
    lines+=['','配对wall是逐题比值均值；AR/方法总wall是混合请求时延比，含输出长度变化，不是论文per-token speedup。QA质量为F1×100、摘要为ROUGE-L×100，其他为正确数/可评分数。','',
        '## 混合总量','','|方法|秒|总token|实际token/s|AR/方法总wall|AR一致|','|---|---:|---:|---:|---:|---:|']
    for arm,v in analysis['totals'].items():
        same=sum(f[arm]['same_ar'] for f in analysis['families'].values())
        lines.append(f"|{arm}|{v['seconds']:.3f}|{v['tokens']}|{v['tokens']/v['seconds']:.2f}|{analysis['totals']['AR']['seconds']/v['seconds']:.3f}×|{same}/{analysis['requests']}|")
    lines+=['','## 身份与范围','',
        f"严格方法对本批AR输出差异{len(analysis['strict_identity_differences'])}，全部保留于analysis.json；DFlash/当前相对上一批输出差异{len(analysis['previous_output_differences'])}，不将跨批速度相除。",
        '182题输入/模板/顺序快照/输出上限与旧扩展实验逐项绑定。编辑沿用v3规范31/32可评分掩码；LongBench19/32首尾截取，QA128/摘要512、指令/代码2048、AIME4096，截断保留。',
        'BASTION调用已锁定官方源码、默认自适应32..8192树预算；同GPU已有独立合成校准/常数profile复用，不用测试答案调参数。官方统计同步计入整请求时间；不冒充原文BF16后端。',
        f"生成UTC {info.get('started_utc')} 至 {info.get('finished_utc')}，加载{info.get('load_seconds',0):.2f}s另列，质量评分不计入推理时延。",
        '协议：[EXPANDED_BASELINES_PROTOCOL.md](EXPANDED_BASELINES_PROTOCOL.md)。公开记录：[results/expanded-baselines](../results/expanded-baselines/)。',
        '`python -m scripts.report_expanded_baselines --input results/expanded-baselines`可从公开行重建。','']
    report=ROOT/'docs/EXPANDED_BASELINES_RESULTS.md';report.write_bytes('\n'.join(lines).encode())
    publication=dict(reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),payloads={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='publication.json'})
    (out/'publication.json').write_text(json.dumps(publication,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(generations=len(rows),totals=analysis['totals']),indent=2))


if __name__=='__main__':main()
