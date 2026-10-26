"""Reconstruct infra acceptance evidence using only the Python standard library."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[3];out=root/'results/infra-tree-v04'
    execution=json.loads((out/'execution-verification.json').read_bytes())
    verification=json.loads((out/'acceptance/verification.json').read_bytes())
    manifest=json.loads((out/'INFRA_TREE_SOURCE_MANIFEST.json').read_bytes())
    assert execution['status']==verification['status']=='complete'
    assert execution['commit']==manifest['commit']
    assert execution['files_verified_before']==execution['files_verified_after']==len(manifest['files'])
    actual_source=verification['provenance']['source']['files']
    expected_source={n[len('memorydraft/'):]:h for n,h in manifest['files'].items()
                     if n.startswith('memorydraft/') and n.endswith(('.py','.json','.txt'))}
    assert actual_source==expected_source
    rows=[json.loads(line) for line in gzip.decompress((out/'acceptance/benchmark/rows.jsonl.gz').read_bytes()).splitlines()]
    methods={'AR','DFlash','ASD','BudgetTree','BudgetTree-ASD'}
    grouped=defaultdict(list)
    for row in rows:grouped[row['request_id'],row['method']].append(row)
    ids={r['request_id'] for r in rows}
    assert len(rows)==60 and len(ids)==6 and set(grouped)=={(i,m) for i in ids for m in methods}
    config=json.loads((out/'acceptance/runtime.json').read_bytes())
    cost_digest=hashlib.sha256((out/'acceptance/costs.json').read_bytes()).hexdigest()
    assert cost_digest==config['cost_tree']['sha256']
    graphs=verification['warmup_graphs']
    for (identity,method),entries in grouped.items():
        assert sorted(r['repeat'] for r in entries)==[0,1]
        reference=grouped[identity,'AR'][0]
        assert entries[0]['token_ids']==entries[1]['token_ids']
        for r in entries:
            assert math.isfinite(r['latency_seconds']) and r['latency_seconds']>0
            assert (r['prompt_sha256'],r['snapshot_id'],r['execution_profile'])==(reference['prompt_sha256'],reference['snapshot_id'],reference['execution_profile'])
            assert r['counters']['compile_graphs_before']==r['counters']['compile_graphs_after']==graphs
            if method in {'ASD','BudgetTree-ASD'}:
                p=r['asd'];assert r['approximate'] and r['correct'] is None and not r['quality_evaluated']
                assert 0<=p['spent']<=p['budget']==8 and p['gate']==.25 and p['cap']==2
                assert p['blocks']==r['counters']['rounds'] and 0<=p['relaxed_tokens']<=2*p['blocks']
            else:assert r['correct'] and not r['approximate'] and r['token_ids']==reference['token_ids']
            if method in {'BudgetTree','BudgetTree-ASD'}:
                assert r['tree']['sha256']==cost_digest
                assert r['counters']['tree_rounds']==r['counters']['rounds']
                assert r['counters']['tree_built_nodes']>=r['counters']['tree_verification_nodes']
    equivalence=json.loads((out/'acceptance/research-equivalence.json').read_bytes())
    assert len(equivalence)==12 and len({(r['request_id'],r['method']) for r in equivalence})==12
    for item in equivalence:
        row=grouped[item['request_id'],item['method']][0]
        assert item['exact'] and item['rounds']==row['counters']['rounds']
        assert item['output_sha256']==hashlib.sha256(json.dumps(row['token_ids']).encode()).hexdigest()
    http=json.loads((out/'acceptance/http.json').read_bytes())
    assert http['stream_exact'] and http['json']['token_ids']==http['sse_complete']['token_ids']
    assert http['json']['asd']==http['sse_complete']['asd'] and http['json']['tree']==http['sse_complete']['tree']
    assert verification['http_json_sse_exact'] and verification['cancellation_recovery_methods']==['BudgetTree','BudgetTree-ASD']
    tests={}
    for label,path in [('local',out/'local-checks/tests-full.xml'),('server',out/'tests-server.xml')]:
        suite=ET.fromstring(path.read_bytes()).find('testsuite');a={k:int(suite.attrib[k]) for k in ('tests','errors','failures','skipped')}
        assert a['errors']==a['failures']==0
        tests[label]=dict(passed=a['tests']-a['skipped'],skipped=a['skipped'])
    installation=json.loads((out/'local-checks/installation.json').read_bytes())
    assert installation['isolated'] and not installation['torch_installed'] and not installation['fastapi_installed']
    assert installation['source_commit']==manifest['commit'] and installation['canonical_package_files_verified']==88
    result=dict(status='complete',source_commit=manifest['commit'],source_sha256=manifest['source_sha256'],
        source_files=len(manifest['files']),independent_requests=6,timed_rows=60,strict_rows=36,approximate_rows=24,
        research_equivalence=12,cost_table_sha256=cost_digest,formal_new_graphs=0,tests=tests,
        http_json_sse_exact=True,transport='ASGI TestClient with real GPU engine; not TCP load testing',
        cancellation_recovery=['BudgetTree','BudgetTree-ASD'],quality_evaluated=False,
        maximum_peak_gpu_bytes=max(r['peak_gpu_bytes'] for r in rows),installation=installation,
        summary=verification['summary'])
    lines=['# Infra v0.4 成本树 + DFlash + ASD 验收','',
        '统一配置、CLI、服务JSON/SSE和benchmark已接入成本树，保留原严格方法；近似ASD必须显式启用。','',
        f"本机完整回归 **{tests['local']['passed']} passed / {tests['local']['skipped']} skipped**；A6000服务/树专项 **{tests['server']['passed']} passed**。",
        f"已安装0.4.0 wheel：{installation['wheel_bytes']}B，SHA-256 `{installation['wheel_sha256']}`。无Torch/无FastAPI的隔离环境中3项CLI/demo通过，88个包文件与本次GPU源逐字节一致。",'',
        '## 真实GPU部署验收','',
        '- 单A6000、Qwen3-8B/审计DFlash、FP32/eager attention/RMSNorm/thread4、B8/g.25/M2。真实CLI在合成前缀校准成本，自动输出绑定哈希的运行配置。',
        '- 6个已观察公开请求（GSM2、HumanEval1、MT首轮1、Alpaca2），五方法两重复，共60主计时。另6条AR参考用于运行时身份检查。',
        '- AR/DFlash/BudgetTree共36计时行逐token、prompt、snapshot一致；ASD和组合24行校验同输入/预算，correct为null，不冒称答案质量通过。',
        '- 严格树与组合各6次，共12次同成本表的研究实现对照，输出/轮数/ASD账本一致。两个树方法取消后恢复相同结果。',
        '- 真实GPUengine的ASGI TestClient JSON/SSE输出、成本表身份和预算一致。它验证应用接口及单worker链路，不是新增TCP压测或容量结论。',
        f"- 正式新增编译图0；峰allocated约{result['maximum_peak_gpu_bytes']/2**30:.3f}GiB。加载、离线校准和预热另列。",'',
        '## 验收批次计时（不作新研究结论）','',
        '|方法|观测/独立请求|自然输出token/s|相对AR的总时延比倒数|严格身份|',
        '|---|---|---:|---:|---|']
    for method in ('AR','DFlash','ASD','BudgetTree','BudgetTree-ASD'):
        r=result['summary']['methods'][method]
        status='通过' if r['strict_correct'] else '近似，任务质量未评'
        lines.append(f"|{method}|{r['observations']}/{r['independent_requests']}|{r['tokens_per_second']:.3f}|{r['speedup']:.3f}×|{status}|")
    lines+=['','自然停止和近似路径可能改变输出长度，上表是小规模部署验收，不替代七任务独立研究或开放质量评估。完整在线成本计入，插桩不混作正常计时。',
        '成本表依赖模型和运行环境，新设备应重新校准。原研究收益与质量取舍见[七任务结果](ASD_PAPER_EXTENSION_FINDINGS.md)，使用方式见[部署说明](INFRA_TREE_GUIDE.md)。','',
        '## 来源和复核','',
        f"实际GPU源提交 `{manifest['commit']}`，{len(manifest['files'])} canonical Git文件执行前后SHA一致；source SHA `{manifest['source_sha256']}`。",
        f"本次成本表SHA `{cost_digest}`。原始行、校准样本、HTTP结果、测试、执行验证与源码manifest保存在[results/infra-tree-v04](../results/infra-tree-v04)。",
        '原infra-v0.3与研究固定分支保持；新版独立归档单列，不把后续文档提交冒充GPU执行源。','',
        '```powershell',
        '.venv\\Scripts\\python.exe -m results.infra-tree-v04.helpers.audit --check',
        '.venv\\Scripts\\python.exe -m results.infra-tree-v04.helpers.verify_publication --git',
        '.venv\\Scripts\\python.exe -m scripts.verify_frozen_releases','```','']
    outputs={out/'analysis.json':(json.dumps(result,indent=2)+'\n').encode(),
             root/'docs/INFRA_TREE_VALIDATION.md':'\n'.join(lines).encode()}
    for path,raw in outputs.items():
        if args.check:assert path.read_bytes()==raw,str(path)
        else:path.write_bytes(raw)
    print(json.dumps({k:result[k] for k in ('status','timed_rows','strict_rows','approximate_rows','research_equivalence')}))


if __name__=='__main__':main()
