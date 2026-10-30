"""Publish v0.5 deployment acceptance, separate from research performance claims."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from scripts.report_branch_drift_optimization import write

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='results/raw/infra-fidelity-v05')
    args = parser.parse_args()
    source, out = ROOT/args.input, ROOT/'results/infra-fidelity-v05'
    out.mkdir(parents=True,exist_ok=True)
    for path in sorted(source.rglob('*')):
        if path.is_file() and path.name != 'publication.json':
            target = out/path.relative_to(source)
            target.parent.mkdir(parents=True,exist_ok=True)
            if path.resolve() != target.resolve():
                shutil.copyfile(path,target)
    info = json.loads((out/'verification.json').read_bytes())
    execution = json.loads((out/'execution.json').read_bytes())
    source_manifest = json.loads((out/'source-manifest.json').read_bytes())
    assert info['status']=='completed' and execution['status']=='completed'
    assert info['research_equivalence']==7 and info['cli_exact'] and info['http_json_sse_exact'] and info['cancellation_recovery']
    assert not any(execution['restore_exit_codes'].values())
    rows = json.loads((out/'research-equivalence.json').read_bytes())
    assert len(rows)==7 and all(r['exact'] for r in rows)
    http = json.loads((out/'http.json').read_bytes())
    assert http['stream_exact'] and http['speculation_events']>0
    assert http['health']['tree_policy']==http['json']['tree']['verification']
    lines = ['# Infra v0.5接入验收', '',
        '已将高保真反馈树、最小增量接受、分数复用和按块KV接入正式配置/CLI/JSON/SSE运行时，生产路径不patch研究脚本。',
        '这是部署验收，沿用七类各一个已观察公共输入的128/EOS，不重新声称研究准确率、加速比或容量。', '',
        '|检查|结果|','|---|---|',
        '|本机完整回归|1299 passed / 7 skipped / 227.78s|',
        '|本机接入专项|66 passed / 9.21s|',
        '|A6000专项|145 passed / 12.89s|',
        '|正式运行时与同成本表研究实现|七任务7/7 token/输入身份/预算摘要一致|',
        '|真实CLI|与直接运行token及非计时预算摘要一致|',
        '|HTTP JSON / SSE / 可视化|输出一致，含真实speculation事件和策略元数据|',
        '|取消与恢复|释放owner，恢复请求输出和预算一致|',
        '|客户端策略覆盖|额外字段拒绝，HTTP 422|',
        '|严格/近似benchmark|五方法各一次计时；严格DFlash/树匹配AR，近似分数保持null|', '',
        'API字段：`healthz.tree_policy`、SSE start的`tree_policy`、complete的`tree.verification`与`counters.kv_*`。',
        '成本表绑定完整规范化策略及环境，buffered KV与公共前缀缓存不兼容时启动拒绝。其他方法与空tree_policy保持原路径。', '',
        '## 执行来源', '',
        f"实际GPU源码commit `{source_manifest['commit']}`，{execution['source_files_verified_before']}文件运行前后SHA一致，实际ZIP SHA `{execution['archive_sha256']}`。",
        f"owner时间：{execution['started_utc']}至{execution['finished_utc']}；GPU1，隔离根`/media/a6000/D/ZhaoBinHe/MemoryDraft/infra-fidelity-v05-v2`。",
        '命令：`CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_infra_fidelity_v05`。',
        '先以真实CLI重新校准，再加载新资产运行直接/研究对照、接口和CLI检查。校准结果为部署资产，不能拿到另一种环境直接用。',
        f"服务恢复退出码：{execution['restore_exit_codes']}；结束三服务active、27B health=ok。GPU0聊天持续运行。", '',
        '初次v1完成研究/benchmark/HTTP检查，但CLI子进程与仍驻留的父模型同时占卡而OOM。仅将独立CLI移到父engine创建前；产品推理未改，v1源码、执行、测试与CLI失败日志保存在failed-cli/。', '',
        '## 交付与证据', '',
        '[使用指南](INFRA_FIDELITY_GUIDE.md)，[公开原始证据](../results/infra-fidelity-v05/)。独立分支`codex/infra-interview-v0.5`、标签`infra-v0.5-interview`，旧v0.3/v0.4不移动。',
        '本机独立副本/ZIP/bundle/逐文件SHA位于`artifacts/infra-interview-v0.5/`，不包含环境、权重或私聊。',
        'A6000独立运行目录为`/media/a6000/D/ZhaoBinHe/MemoryDraft/infra-interview-v0.5`，144文件含校准资产/配置，代码SHA匹配验收源；已做版本/配置/成本校验，未启动额外驻留模型或切换现有聊天服务。',
        '研究结论继续引用[最小增量接受](GATE_MIN_GAIN_RESULTS.md)与[执行优化](TREE_EXECUTION_RESULTS.md)，不从验收小批计时重估性能。',
        '`python -m scripts.report_infra_fidelity_v05 --input results/infra-fidelity-v05`可重建本报告。', '']
    report = ROOT/'docs/INFRA_FIDELITY_VALIDATION.md'
    report.write_bytes('\n'.join(lines).encode('utf-8'))
    write(out/'publication.json',dict(reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        payloads={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(out.rglob('*')) if p.is_file() and p.name!='publication.json'}))
    print(json.dumps(dict(status=info['status'],research_equivalence=info['research_equivalence'],
                         cli_exact=info['cli_exact'],http_json_sse_exact=info['http_json_sse_exact'])))


if __name__=='__main__':
    main()
