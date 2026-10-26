# DSpark 成本树等价优化协议（2026-10-26）

目的：修复逐父token反复Markov投影及构建32节点后丢弃尾部的开销。保留原DSpark、DFlash分支、权重和结果；优化在 `codex/dspark-tree-opt`，部署新目录 `dspark-tree-opt`。

## 两项改动

1. 请求内LRU64行：vanilla Markov偏置只依赖上一token，复用同形状GEMV计算的完整偏置行；不跨请求预热，不改变加法/softmax/topk顺序。151936词表FP32上限约37.1MiB，实际用量记录；权重对象/版本/设备/类型变化拒绝缓存。
2. 严格上界：最佳优先队列的未来节点质量不超过当前前沿最大质量u。当前累计A、节点n，未来k的收益最多为`(A+(k-n)u)/原cycle(context,k)`。对全部未来k逐一比较，并向上补浮点误差；只有都不可能改善当前最佳才停止。保留原严格大于/小树优先的同分规则；不假设时延表单调或目标单峰。

未做批量GEMM、权重重训、长块推断或接受阈值放宽。旧每卡两套成本表冻结，使本轮选树语义与原版可严格对齐。

## 数据与计时

- 使用现有 `asd-paper-extension` 固定顺序：Alpaca前16、MT前4对话、HumanEval前4。24源任务/28匹配输入；12个Alpaca此前未用于DSpark，但均为已观察扩展数据，标为开发回放。
- 四组：DFlash-Tree-ASD、DSpark-ASD、DSpark-Tree-ASD、DSpark-Tree-ASD-Fast。Qwen3-8B、FP32/eager/RMSNorm-inductor；B8/g.25/M2；DFlash原生15与DSpark原生7候选。置信度裁剪关闭。
- 自然4096/EOS两遍，首遍确定性随机顺序、次遍逆序；共同前缀128/EOS一遍；固定256/忽略EOS一遍。共448主计时。所有方法本轮同卡配对，不使用旧时延作对照。
- 复用旧AR的输入/快照及MT第一轮共同历史，校验实际messages和prompt SHA；不计算对旧AR速度比。各方法完整质量、长度、截断保留。
- 所有请求缓存从空开始；构造、填充、淘汰、选树摘要计算及生成全部计时。所有DSpark树逐轮完整tokens/parents/depths/语义元数据摘要，核对输出、文本、停止及ASD账本；只允许built_nodes/conditional_distributions/search_bound_stopped计算统计不同。正式新图必须0。
- 两A6000按每族源序号奇偶分片；Target/DFlash/DSpark同时驻留，避免不同常驻内存造成混淆。新输出前冻结本协议和源码manifest，禁止看结果后选参数或题目。

## 机制消融及边界

七族各前2源任务，严格线性DSpark生成128并临时捕获每请求前≤4个真实GPU状态；同冻结成本表比较原版、仅缓存、仅上界、二者。每变体4遍，首遍预热另列，余3遍交替顺序；每遍请求冷缓存。全部选树精确相同才能进入主计时。只导出计时、计数和语义校验状态，不导出logits/隐藏特征/KV。

这只能优化构树成本，无法消除7候选块的轮数上限。HumanEval4题作为短块边界检查；MT8匹配单轮不当作自身历史对话级结论。不同Draft的近似输出仍可能有质量取舍，工程等价不代表近似算法无损。

## 命令与产物

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_dspark_tree_fast.py tests/test_dspark.py tests/test_dspark_tree_opt_report.py --basetemp artifacts/pytest-dspark-opt-ready --junitxml=artifacts/dspark-opt-ready-tests.xml
.venv\Scripts\python.exe -m scripts.package_asd_tree_source --output artifacts/dspark-tree-opt-source.zip --manifest DSPARK_TREE_OPT_SOURCE_MANIFEST.json
```

```bash
cd /media/a6000/D/ZhaoBinHe/MemoryDraft/dspark-tree-opt
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_dspark_tree_opt_pipeline
```

每worker上限10800秒，测试/报告各有独立超时；独立源目录、两owned进程和新编译缓存。结果 `results/raw/dspark-tree-opt`，发布 `results/dspark-tree-opt`、`docs/DSPARK_TREE_OPT_RESULTS.md` 及结论；源码前后逐文件SHA核验。
