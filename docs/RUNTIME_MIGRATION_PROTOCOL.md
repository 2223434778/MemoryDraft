# 旧正结果到 v0.3 运行时的迁移验证

2026-10-18。运行前口径；目的为工程回归，不是新的独立研究确认。

## 固定输入与停止条件

- 使用 `results/reuse-confirmation/` 原4项目、8请求、合法初始历史及上一轮已完成回答。原参考、原配置及原96次记录先通过既有审计；不重新生成历史，不将本轮输出写回历史。
- 旧确认实际源码清单SHA为`2518e677cde7d1ee4d7c4201b560d14f7a44bd75500c95e2497c0c072e2b46c2`。从Git固定提交`dd0077fb6ed278837f9e3de48f946008519816ff`恢复旧解码，`f43239af59c49c17ee4b9d7749473259e501251d`恢复旧索引，按原清单SHA校验；已与服务器保留的原运行源码归档核对。最初草稿将索引差异误认为换行，恢复校验失败后查明是后来新增的`include_user_records=False`开关，已在任何GPU运行前修正。当前批次只有assistant历史；旧索引与新版对齐索引仍逐例检查tokens/记录边界/来源指纹一致，不假定等价。
- 同一进程、同一份Qwen3-8B/DFlash权重、A6000 GPU0、TF4.57.3、FP32/eager、4 CPU线程。R长度32、最小匹配6、冷却1、pending上限1024，原768输出预算。
- 全部8请求先重新运行AR，逐token、Target prompt及快照匹配原参考；任何不一致保存failure并停止计时。新参考只作核验，不进入历史。

## 五组交错计时

1. `Legacy-D`：原解码循环，关闭R，沿用原端到端计时入口。
2. `Legacy-R`：原解码循环，历史与当前user文本作为候选。
3. `Runtime-D`：v0.3原生产入口的DFlash适配器。
4. `Runtime-R`：v0.3原生产入口的R+DFlash；候选默认含历史及system/user文本。
5. `Runtime-R-aligned`：v0.3入口，仅将候选当前文本对齐为最后一条user文本；通过实验脚本临时构造器包装实现，不修改生产策略或Target输入。

每请求、每组3重复，共120计时。固定种子按请求/重复打乱顺序，每次重新读取快照及冷建索引。旧入口持有SQLite连接，新入口按请求开关连接，这是被测工程成本之一。所有在线tokenization、索引、前向、KV维护、GPU同步和解码计入；加载、独立预热、核验AR及离线资源检查分开。

计时不经HTTP，不启用profiler；不新增路由、自适应候选长度或格式归一化。每组保存tokens、prompt/snapshot哈希、R/D轮数和R接受量，严格匹配冻结参考，遇到错误即停止。旧输入文件在运行前后校验不变。

## 判断与限制

主要比较同批`Legacy-D/Legacy-R`及`Runtime-D/Runtime-R`，观察旧收益是否保留；`Runtime-D/Legacy-D`与`Runtime-R-aligned/Legacy-R`观察迁移开销。`Runtime-R`与对齐组分离候选资源变化。R/D计数用于检查是否改变实际执行路径，不据计数直接折算收益秒数。

报告以项目为重采样单位给出2000次bootstrap95%区间；只有4个项目，不泛化生产分布。工程回归预算暂取10%，以当前/旧版同批耗时比判断，仅为工程门槛，不等于显著性检验。若发现额外开销，先报告，不根据本批结果调整策略后冒充同一次确认。

这批合成任务的历史必要性已被事后归因削弱。本次只验证已有复用收益能否迁移到infra，不重启H、不恢复新论文主张，也不把本次测试当成新工作负载泛化。

## 恢复与执行

在含完整历史的Git checkout中、使用DFlash虚拟环境，按字节恢复原代码；Python写入避免PowerShell重定向转换编码。运行时还会根据原provenance重新校验SHA。

```bash
python -c "from pathlib import Path; import subprocess; from memorydraft.evaluation.runtime_migration import LEGACY_COMMITS; p=Path('artifacts/runtime-migration-legacy'); p.mkdir(parents=True,exist_ok=True); [p.joinpath(n).write_bytes(subprocess.check_output(['git','cat-file','blob',LEGACY_COMMITS[n]+':'+s])) for n,s in [('legacy_decode.py','memorydraft/backends/dflash_reuse.py'),('legacy_suffix.py','memorydraft/retrieval/suffix.py')]]"
python -m scripts.run_runtime_migration --config configs/runtime-dflash.json --legacy artifacts/runtime-migration-legacy --output artifacts/runtime-migration-NEW
python -m scripts.write_runtime_migration_report --source artifacts/runtime-migration-NEW --output artifacts/runtime-migration-NEW/report.md
```

恢复/执行说明在计时完成后补写，运行时绑定的是上方已冻结实验口径及当时协议SHA；新增操作说明不更改已执行配置或旧provenance。
