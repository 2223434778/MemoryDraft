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
    frozen={r['request_id']:r for r in json.loads((source/'frozen-references.json').read_bytes())}
    reference_audit=[r['request_id'] for r in rows if r['arm']=='AR'
        and any(r[k]!=frozen[r['request_id']][k] for k in ('token_ids','text','finish_reason','prompt_sha256','snapshot_id'))]
    write(out/'reference-audit.json',dict(new_ar_vs_frozen_differences=reference_audit))
    slowest={arm:max((r for r in rows if r['arm']==arm),key=lambda r:r['latency_seconds']) for arm in ARMS}
    indexed={(r['request_id'],r['arm']):r for r in rows}
    same_output={}
    for baseline in ('DFlash','DDTree-32','BASTION','BudgetTree'):
        pairs=[(r,indexed[r['request_id'],baseline]) for r in rows if r['arm']=='BufferedKV-ASD']
        pairs=[(a,b) for a,b in pairs if all(a[k]==b[k] for k in ('token_ids','text','finish_reason'))]
        same_output[baseline]=dict(n=len(pairs),paired_ratio=statistics.mean(a['latency_seconds']/b['latency_seconds'] for a,b in pairs),
            total_ratio=sum(a['latency_seconds'] for a,b in pairs)/sum(b['latency_seconds'] for a,b in pairs),
            current_calls=sum(len(a['rounds']) for a,b in pairs),baseline_calls=sum(len(b['rounds']) for a,b in pairs))
    dd_pairs=[(indexed[r['request_id'],'DDTree-32'],r) for r in rows if r['arm']=='BASTION']
    execution_audit=dict(same_output=same_output,bastion_node_sizes=sorted({b['tree_nodes'] for a,bastion in dd_pairs for b in bastion['rounds']}),
        dd32_bastion_equal_advances=sum([b['advance'] for b in a['rounds']]==[b['advance'] for b in btree['rounds']] for a,btree in dd_pairs))
    write(out/'execution-audit.json',execution_audit)
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
    lines+=['','## 每组最慢请求','','|方法|请求|秒|输出token|停止|','|---|---|---:|---:|---|']
    for arm,r in slowest.items():
        lines.append(f"|{arm}|{r['request_id']}|{r['latency_seconds']:.3f}|{len(r['token_ids'])}|{r['finish_reason']}|")
    lines+=['','## 相同输出的辅助归因','',
        '主结果保留全部66输入。下面只在完整token/文本/停止均相同的子集上比较当前方案，用于区分执行效率与输出路径变化，不替代主表。',
        '|对照|同输出请求数|当前配对wall变化|当前总wall变化|当前/对照验证轮|','|---|---:|---:|---:|---:|']
    for baseline,v in same_output.items():
        lines.append(f"|{baseline}|{v['n']}|{pct(v['paired_ratio'])}|{pct(v['total_ratio'])}|{v['current_calls']}/{v['baseline_calls']}|")
    lines+=[f"本轮BASTION实际节点规模为{execution_audit['bastion_node_sizes']}；与DDTree32逐轮推进长度完全相同的请求{execution_audit['dd32_bastion_equal_advances']}/66。因此两者接近的表现应结合实际相同的预算/推进轨迹解释。"]
    lines+=['','## 质量与执行边界','',
        '数学采用既有答案等价评分，代码采用原始隐藏测试的单次greedy通过率。MMLU为零样本生成字母合同，格式错误计错；这批开发题不等同完整MMLU知识能力。',
        'MT/Alpaca本轮没有新增裁判评分，表中—不代表质量相同。AR一致性只是保真度，不代替正确率。',
        f"严格方法与本窗口AR的完整输出差异：{len(analysis['strict_identity_differences'])}；详情见analysis.json。所有差异保留，不从成绩中剔除。",
        f"本轮AR与旧冻结AR完整输出/输入身份差异：{len(reference_audit)}；见reference-audit.json。旧AR只用于输入/历史核验，不代替本轮实测时延。",
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
        findings_sha256=hashlib.sha256((ROOT/'docs/EXTERNAL_TREES_FINDINGS.md').read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((ROOT/'docs/EXTERNAL_TREES_PROTOCOL.md').read_bytes()).hexdigest(),
        preparer_sha256=hashlib.sha256((ROOT/'scripts/prepare_external_trees.py').read_bytes()).hexdigest(),
        payloads={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file() and p.name!='publication.json'}))
    print(json.dumps(dict(totals=totals,strict_differences=analysis['strict_identity_differences']),indent=2))


if __name__=='__main__':main()
