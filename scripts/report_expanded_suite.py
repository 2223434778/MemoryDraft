"""Rebuild expanded workload results without mixing literature and local timing."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

ROOT=Path(__file__).resolve().parents[1]
ARMS=('DFlash','ASD','BufferedKV-ASD')


def summarize(rows, *, arms=ARMS):
    ids={r['request_id'] for r in rows}
    indexed={(r['request_id'],r['arm']):r for r in rows}
    if len(indexed)!=len(rows) or set(indexed)!={(i,a) for i in ids for a in arms}:
        raise ValueError('Incomplete expanded comparison')
    for identity in ids:
        group=[indexed[identity,a] for a in arms]
        if len({(r['prompt_sha256'],r['snapshot_id']) for r in group})!=1:raise ValueError('Changed inputs')
        for r in group:
            if not math.isfinite(r['latency_seconds']) or r['latency_seconds']<=0:raise ValueError('Invalid timing')
            if len(r['token_ids'])>r['max_new_tokens']:raise ValueError('Output limit exceeded')
    families={}
    for benchmark in dict.fromkeys(r['benchmark'] for r in rows):
        family={}
        for arm in arms:
            selected=[r for r in rows if r['benchmark']==benchmark and r['arm']==arm]
            seconds=sum(r['latency_seconds'] for r in selected)
            tokens=sum(len(r['token_ids']) for r in selected)
            quality=[r['quality'] for r in selected]
            correct=[q.get('secondary_correct',q.get('correct')) for q in quality]
            scores=[q['score'] for q in quality if 'score' in q]
            ttft=[r['ttft_seconds'] for r in selected if r.get('ttft_seconds') is not None]
            value=dict(n=len(selected),seconds=seconds,tokens=tokens,tps=tokens/seconds,
                correct=sum(q is True for q in correct),scored=sum(q is not None for q in correct),
                score=statistics.mean(scores) if scores else None,
                truncated=sum(r['finish_reason']=='length' for r in selected),
                same_dflash=sum(r['token_ids']==indexed[r['request_id'],'DFlash']['token_ids'] for r in selected),
                calls=sum(len(r['rounds']) for r in selected),
                mean_ttft=statistics.mean(ttft) if ttft else None,
                peak_gpu_bytes=max(r['peak_gpu_bytes'] for r in selected),
                evaluation_errors=[r['request_id'] for r in selected if r['quality'].get('evaluation_error')])
            for key in ('instructions','loose_instructions'):
                values=[v for q in quality for v in q.get(key,[])]
                value[key]=dict(passed=sum(values),total=len(values))
            value['loose_correct']=sum(q.get('loose_correct') is True for q in quality)
            for baseline in arms:
                pairs=[(r,indexed[r['request_id'],baseline]) for r in selected]
                value[baseline]=dict(paired_ratio=statistics.mean(a['latency_seconds']/b['latency_seconds'] for a,b in pairs),
                    total_ratio=seconds/sum(b['latency_seconds'] for a,b in pairs),
                    lost=[a['request_id'] for a,b in pairs if b['quality'].get('secondary_correct',b['quality'].get('correct')) is True and a['quality'].get('secondary_correct',a['quality'].get('correct')) is False],
                    gained=[a['request_id'] for a,b in pairs if a['quality'].get('secondary_correct',a['quality'].get('correct')) is True and b['quality'].get('secondary_correct',b['quality'].get('correct')) is False])
            family[arm]=value
        families[benchmark]=family
    totals={a:dict(seconds=sum(f[a]['seconds'] for f in families.values()),tokens=sum(f[a]['tokens'] for f in families.values()),
        same_dflash=sum(f[a]['same_dflash'] for f in families.values())) for a in arms}
    return dict(generations=len(rows),requests=len(ids),families=families,totals=totals)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='results/raw/expanded-suite')
    args=parser.parse_args()
    source=ROOT/args.input;out=ROOT/'results/expanded-suite';out.mkdir(parents=True,exist_ok=True)
    info=json.loads((source/'provenance.json').read_bytes())
    if info['status']!='completed':raise ValueError('Unfinished generation')
    data=(source/'rows.jsonl').read_bytes() if (source/'rows.jsonl').exists() else gzip.decompress((source/'rows.jsonl.gz').read_bytes())
    rows=[json.loads(s) for s in data.decode().splitlines()]
    quality_version=3 if (source/'quality-v3-manifest.json').exists() else 2
    prefix=f'quality-v{quality_version}-'
    revision=json.loads((source/(prefix+'manifest.json')).read_bytes())
    if (source/'rows.jsonl').exists() and hashlib.sha256(data).hexdigest()!=revision['generation_rows_sha256']:
        raise ValueError('Rescoring changed generation rows')
    quality={(r['request_id'],r['arm']):r['quality'] for r in (json.loads(s) for s in (source/(prefix+'scores.jsonl')).read_text().splitlines())}
    for r in rows:
        key=r['request_id'],r['arm']
        if key in quality and r.get('quality_revision',0)<quality_version:
            r.setdefault('initial_quality',r['quality']);r['quality']=quality[key];r['quality_revision']=quality_version
        r['quality'].pop('metadata',None)
        r['rounds']=[{k:b[k] for k in ('advance','accepted','strict_accepted','relaxed_tokens','regret','tree_nodes','route') if k in b} for b in r['rounds']]
    analysis=summarize(rows)
    if len(rows)!=info['expected']:raise ValueError('Missing generated cells')
    compressed=gzip.compress(('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n').encode(),mtime=0)
    (out/'rows.jsonl.gz').write_bytes(compressed)
    for path in source.iterdir():
        if path.is_file() and path.name not in ('rows.jsonl','rows.jsonl.gz','tasks.json','analysis.json','publication.json'):
            if path.resolve()!=(out/path.name).resolve():shutil.copyfile(path,out/path.name)
    if (source/'tasks.json').exists():
        tasks=json.loads((source/'tasks.json').read_bytes())
        tasks=[{k:r[k] for k in ('id','benchmark','limit','subset','difficulty','contest_date','original_prompt_tokens','input_clipped') if k in r} for r in tasks]
        (out/'tasks-metadata.json').write_text(json.dumps(tasks,indent=2)+'\n',encoding='utf-8')
    (out/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n',encoding='utf-8')
    lines=['# 新场景扩展：实测结果','',f"{analysis['requests']}固定新请求、三组{len(rows)}自然生成；Qwen3-8B/DFlash、单A6000、FP32/eager/RMSNorm。",'文献基线另见[引用表](EXPANDED_LITERATURE_BASELINES.md)，不同配置不相除排名。','',
        '|任务|方法|配对耗时vs DFlash|配对耗时vs ASD|正确/可评分 或 指标|截断|DFlash一致|实际tokens/s|平均首token秒|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def pct(v):return f'{(v-1)*100:+.2f}%'
    for benchmark,family in analysis['families'].items():
        for arm in ARMS:
            v=family[arm]
            q=f"{v['correct']}/{v['scored']}" if v['scored'] else (f"{v['score']*100:.2f}" if v['score'] is not None else '—')
            lines.append(f"|{benchmark}|{arm}|{pct(v['DFlash']['paired_ratio'])}|{pct(v['ASD']['paired_ratio'])}|{q}|{v['truncated']}|{v['same_dflash']}/{v['n']}|{v['tps']:.2f}|{v['mean_ttft']:.3f}|")
    lines+=['','IFEval主表为严格整请求通过数；QA为F1×100，摘要为ROUGE-L×100，不是正确率。CanItEdit按统一canonical可运行掩码评分；AIME按最终boxed整数等价评分。',
        '时延含prefill与全部在线工作，输出变短也影响完整请求时间；配对均值与混合总量比不是同一统计。DFlash一致性不代替任务质量，也不冒充本轮AR验证。','',
        '## IFEval分项','','|方法|严格请求|宽松请求|严格指令|宽松指令|','|---|---:|---:|---:|---:|']
    for a in ARMS:
        v=analysis['families']['ifeval'][a];s,l=v['instructions'],v['loose_instructions']
        lines.append(f"|{a}|{v['correct']}/{v['n']}|{v['loose_correct']}/{v['n']}|{s['passed']}/{s['total']}|{l['passed']}/{l['total']}|")
    lines+=['','## 混合请求总量','','|方法|秒|token|实际tokens/s|DFlash一致|','|---|---:|---:|---:|---:|']
    for a,v in analysis['totals'].items():
        lines.append(f"|{a}|{v['seconds']:.3f}|{v['tokens']}|{v['tokens']/v['seconds']:.2f}|{v['same_dflash']}/{analysis['requests']}|")
    lines+=['','## 评分与范围','',
        'IFEval官方strict/loose；LiveCodeBench v6-only按难度固定8/8/8，官方全测试2s/test；CanItEdit descriptive32，原生成评分及修订评分同时保留。',
        f"编辑规范答案可运行{revision['canonical_scored']}/32，未评分{revision['unavailable']}。评分修订{quality_version}修复对象方法、科学依赖及源码检查环境；统一重评全部编辑输出，不修改token、时间或测试，旧评分保留。",
        'LongBench仅四个子集各8，3840-token首尾截取，保留原始长度/截取标记；不称完整长文榜单。QA128/摘要512输出上限，其他代码/指令2048、AIME4096，截断均计入结果。',
        '未做新参数扫描，三组同题配对，不能把单次子集点估计外推为所有请求稳定保证。原始长文/完整测试载荷不另行发布，数据版本/题目ID/SHA及获取/评分代码可复核。','',
        '## 执行记录','',f"生成{info.get('started_utc')}至{info.get('finished_utc')}，加载{info.get('load_seconds',0):.2f}s、一次合成校准{info.get('calibration_seconds',0):.2f}s另列。",
        '协议：[EXPANDED_SUITE_PROTOCOL.md](EXPANDED_SUITE_PROTOCOL.md)。[公开结果](../results/expanded-suite/)。',
        '`python -m scripts.report_expanded_suite --input results/expanded-suite` 可从公开压缩行重新汇总。报告/独立评分为GPU执行后追加，不冒充冻结推理源码。','']
    report=ROOT/'docs/EXPANDED_SUITE_RESULTS.md';report.write_bytes('\n'.join(lines).encode())
    manifest=dict(reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        payloads={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='publication.json'})
    (out/'publication.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(generations=len(rows),totals=analysis['totals']),indent=2))


if __name__=='__main__':main()
