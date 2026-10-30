"""Rebuild and publish the official tree comparison from complete raw rows."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import statistics

from scripts.diagnose_transport_online import write
from scripts.run_external_trees import ARMS, analyze

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='results/raw/external-trees')
    args=parser.parse_args()
    source,out=ROOT/args.input,ROOT/'results/external-trees'
    out.mkdir(parents=True,exist_ok=True)
    raw=(source/'rows.jsonl').read_bytes() if (source/'rows.jsonl').exists() else gzip.decompress((source/'rows.jsonl.gz').read_bytes())
    rows=[json.loads(line) for line in raw.decode().splitlines()]
    info=json.loads((source/'provenance.json').read_bytes())
    if info['status']!='completed' or len(rows)!=594:
        raise ValueError('Incomplete official comparison')
    analysis=analyze(rows)
    if analysis!=json.loads((source/'analysis.json').read_bytes()):
        raise ValueError('Independent analysis mismatch')
    (out/'rows.jsonl.gz').write_bytes(gzip.compress(raw,mtime=0))
    for path in sorted(source.iterdir()):
        if path.is_file() and path.name not in ('rows.jsonl','rows.jsonl.gz','publication.json'):
            if path.resolve()!=(out/path.name).resolve():shutil.copyfile(path,out/path.name)
    totals={}
    for arm in ARMS:
        selected=[r for r in rows if r['arm']==arm]
        seconds=sum(r['latency_seconds'] for r in selected)
        calls=sum(len(r['rounds']) for r in selected) if arm!='AR' else sum(len(r['token_ids']) for r in selected)
        nodes=[b['tree_nodes'] for r in selected for b in r['rounds'] if 'tree_nodes' in b]
        totals[arm]=dict(seconds=seconds,tokens=sum(len(r['token_ids']) for r in selected),calls=calls,
            mean_nodes=statistics.mean(nodes) if nodes else None,
            max_nodes=max(nodes) if nodes else None,
            identical_ar=sum(f[arm]['natural']['identical_ar'] for f in analysis['families'].values()),
            peak_gpu_bytes=max(r['peak_gpu_bytes'] for r in selected))
        totals[arm]['tps']=totals[arm]['tokens']/seconds
    write(out/'totals.json',totals)
    lines=['# 官方DDTree/BASTION与当前组合：直接比较','',
        '七任务、既有62开发源题/66匹配输入、九组594次自然4096/EOS生成。每格一次，逐请求打乱方法顺序。',
        '共同Qwen3-8B/DFlash、单A6000、FP32/eager/RMSNorm；外部官方生成算法共享已加载模型，非原文BF16/SDPA环境复现。','',
        '|方法|总秒|实际tokens/s|总wall vs DFlash|总wall vs BASTION|AR一致/66|输出token|验证/AR轮|平均树节点|峰GiB|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def pct(ratio):return f'{(ratio-1)*100:+.2f}%'
    for arm,v in totals.items():
        nodes=f"{v['mean_nodes']:.2f}" if v['mean_nodes'] is not None else '—'
        lines.append(f"|{arm}|{v['seconds']:.3f}|{v['tps']:.2f}|{pct(v['seconds']/totals['DFlash']['seconds'])}|{pct(v['seconds']/totals['BASTION']['seconds'])}|{v['identical_ar']}|{v['tokens']}|{v['calls']}|{nodes}|{v['peak_gpu_bytes']/2**30:.2f}|")
    lines+=['','总wall比包含回答长度/路径变化；下表是逐题时延比均值，两者统计不同。模型加载/校准/预热另列，全部在线请求工作计入wall。',
        'AR轮数按实际输出token数计；其他组为验证轮。平均接受推进含根/bonus的官方定义不等于严格草稿接受数。','',
        '|任务|方法|配对wall vs DFlash|配对wall vs ASD|配对wall vs BASTION|正确/可评分|截断|AR一致|tokens/s|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for benchmark,family in analysis['families'].items():
        for arm in ARMS:
            v=family[arm]['natural']
            quality=f"{v['correct']}/{v['scored']}" if v['scored'] else '—'
            lines.append(f"|{benchmark}|{arm}|{pct(v['DFlash']['ratio'])}|{pct(v['ASD']['ratio'])}|{pct(v['BASTION']['ratio'])}|{quality}|{v['truncated']}|{v['identical_ar']}/{v['n']}|{v['tps']:.2f}|")
    lines+=['','## 质量与执行边界','',
        '数学采用既有答案等价评分，代码采用原始隐藏测试的单次greedy通过率。MMLU为零样本生成字母合同，格式错误计错；这批开发题不等同完整MMLU知识能力。',
        'MT/Alpaca本轮没有新增裁判评分，表中—不代表质量相同。AR一致性只是保真度，不代替正确率。',
        f"严格方法与本窗口AR的完整输出差异：{len(analysis['strict_identity_differences'])}；详情见analysis.json。所有差异保留，不从成绩中剔除。",
        'DDTree32/128为总节点上限，官方API传31/127。BASTION保留默认32..8192自适应，未为使其变快或变慢另调预算。',
        'DDTree启用官方默认可选C++整理并记录是否成功；BASTION统计模式的内部同步计入实测，官方统计时钟不替代共同wall。',
        '这是统一执行栈的算法比较；若需比较各系统生产上限，需另外统一BF16并允许各自支持的优化后端。','',
        '## 复现','',
        f"运行{info.get('started_utc')}至{info.get('finished_utc')}，模型加载{info['load_seconds']:.2f}s、合成成本校准{info['calibration_seconds']:.2f}s；DDTree C++整理启用成功：{info['ddtree_cpp_compaction']}。",
        '协议：[EXTERNAL_TREES_PROTOCOL.md](EXTERNAL_TREES_PROTOCOL.md)。原始压缩行、成本探测、源码/上游SHA和测试记录见[公开结果](../results/external-trees/)。',
        '`python -m scripts.report_external_trees --input results/external-trees`。报告脚本为GPU运行后的离线后处理，不冒充冻结执行源。','']
    report=ROOT/'docs/EXTERNAL_TREES_RESULTS.md'
    report.write_bytes('\n'.join(lines).encode())
    write(out/'publication.json',dict(reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        payloads={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file() and p.name!='publication.json'}))
    print(json.dumps(dict(totals=totals,strict_differences=analysis['strict_identity_differences']),indent=2))


if __name__=='__main__':main()
