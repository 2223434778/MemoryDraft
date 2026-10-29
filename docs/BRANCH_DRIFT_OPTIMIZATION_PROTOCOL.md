# 按需JS、门控节点分配和当前token约束

2026-10-29：用户授权实现三项改动并评价。旧Drift、旧成本树ASD、聊天部署与infra冻结版本保留。

## 固定规则与方法

沿用Qwen3-8B/匹配DFlash、FP32/eager/RMSNorm、线程4、原生16块、树8–32节点/Top4、已测物理GPU1成本表。Target输入、停止规则、4096自然输出上限保持。ASD B8/gate .25/cap2；完整词表T1 JS/ln2≤.1、真实Target首选兄弟覆盖和EOS保护保持。

七方法：ASD、严格BudgetTree、原DriftTree-ASD、LazyDrift-ASD、GateTree-ASD、TokenCap-ASD、GateCap-ASD。

- Lazy只改变检查顺序。先按预算/数量/覆盖/EOS筛选并排序所有可行端点，顺序严格沿用最长路径、最低regret、最低索引。按顺序检查路径的未知JS边，失败的边排除所有经过它的路径，缓存已检查边；找到第一条通过的路径即结束。未检查的边保留乐观许可，但其端点均排在已确认赢家之后。相同树/logits下，最终路径、恢复token和账本应与eager完全相同。旁支诊断计数允许不同。
- GateTree沿用Lazy。单节点扩展代替强制Top2成对组；rank0质量权重q0，其余权重qr*(qr/q0)^1，路径权重连乘，除以原实测周期成本选节点数。兄弟top1优先于其余兄弟，实际Target首选仍可能缺失，验证时拒绝该近似。此为偏向严格主干的Draft启发式，不是校准的JS通过概率，也不保证新树保真度单调改善。当前成本表尚未拟合JS开销，实际在线成本全部计入wall。
- TokenCap仅在Lazy上加独立raw regret≤ln4，即当前候选/Target首选概率比≥.25，与剩余后缀长度无关。
- GateCap组合节点分配与TokenCap。阈值和权重固定后不按结果调参。

## 任务与指标

从上轮既有149输入确定性选取低成本开发子集：GSM8、MATH12、HumanEval8、MBPP8、MMLU前14学科各1、MT前4源对话/8轮、Alpaca8，共62源任务/66匹配输入。HumanEval优先保留76，MATH保留全部12，包含已知循环372；是诊断覆盖，不能称新保留集。其余顺序沿用上轮哈希选择。每方法每输入自然4096/EOS及128/EOS各一次，共924计时。各方法随机交错，不删除错误/截断题。

复用冻结AR输出作质量/身份参考，不复用旧AR时延。MT第二轮共享冻结AR首轮历史。原评分、完全相同数、首次分歧前缀、输出长度、轮数与近似数量并报；开放任务本轮只评价保真/速度，不将AR一致称语义正确或官方偏好分数。共同128只比较双方实际达到128的请求。新增gate_seconds/js_seconds为原同步CPU取值边界下的wall摘要，allocation_seconds计入树摘要；不是独立GPU kernel剖析，不从总时间扣除。

主要要求：Lazy与eager在自然及前缀全部输入的token、停止原因、所选路径、逐轮regret和总预算相同；若不一致保留失败并调查。节点分配与TokenCap分别对Lazy及ASD比较，保留质量取舍。固定状态下门槛变严不构成整个生成轨迹保真度单调提高的证明。

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_branch_drift.py tests/test_branch_drift_optimization.py tests/test_branch_drift_evaluation.py tests/test_budget_tree.py tests/test_dflash_reuse.py --basetemp artifacts/pytest-drift-opt --junitxml=artifacts/drift-opt-tests.xml --tb=short
```

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. \
  ../.venv-dflash/bin/python -u -m scripts.run_branch_drift_optimization
```

运行于独立目录；先检查27B空闲，再暂时释放GPU1，结束自动恢复27B及适配器。GPU0聊天保持。结果追加至PROGRESS和独立报告。
