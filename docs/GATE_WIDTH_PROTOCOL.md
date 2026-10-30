# 保真门控不变的树宽度对照

2026-10-30：用户授权根据GSM慢因实现并实测。复用上轮固定62源/66匹配输入、七任务开发诊断子集；保留MATH372/101、HumanEval76，非新保留集。GSM8、MATH12、HumanEval8、MBPP8、MMLU14、MT4源/8轮、Alpaca8。每方法自然4096/EOS及128/EOS各一次，共792计时；方法顺序按输入/阶段打乱，全部错误和截断保留。

## 固定方法与对照

六组：ASD、严格BudgetTree、当前GateCap-ASD（惩罚1）、WidthHalf-ASD（.5）、WidthZero-ASD（0）、WidthFeedback-ASD。

所有近似树保持B8、原gate .25、每块cap2、完整词表T1 JS/ln2≤.1、真实Target首选兄弟覆盖、EOS保护、独立raw regret≤ln4。均为lazy JS，树Top4/8–32节点、原生16块与原成本表。只改变构树的旁支权重qr*(qr/q0)^penalty；不改变Target提示、权重、停止或接受门槛。系数0接近原成本树单节点分配，精确并列下排序可不同，不保证整段输出与AR一致或保真单调变化。

反馈组从1启动。检查已提交路径终点的恢复token：若它是Draft该位置第2–4候选、却未成为该节点的子节点，记一次可由当前候选池覆盖的漏选。EOS、到达原生horizon和词表候选池外均不记此信号。最近4轮无/1次/≥2次信号，下一轮系数分别1/.5/0。状态只在当前请求内，按序更新，不读取未来答案或其他方法轨迹。每次构树使用不可变预算/宽度快照；Target恢复token复用原验证结果，不额外前向。候选rank元数据仅供反馈，不保存完整logits。

## 评价与归因

分别报告：自然请求直接配对时延、输出长度、正确数/截断、AR完全一致、共同128前缀、节点/深度/轮数。开放任务本轮不新增偏好评分。GSM重点比较惩罚降低是否以更宽的树减少此前337轮；不同轨迹上的轮数与树统计不作为同状态独立因果证明。反馈组逐轮重放4轮历史，核验惩罚只由过去信号决定。

Qwen3-8B/匹配DFlash、FP32/eager/RMSNorm、物理GPU1、线程4，模型/数据/评分/成本表沿用前轮。旧AR输出供保真/质量参考，不借用旧时延宣称同窗AR加速。MT第二轮共享冻结AR首轮历史。全部反馈/构树/JS/KV开销计入wall，构树入口计时可能包含未完成Draft GPU工作，不能称纯CPU耗时。

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_gate_width.py tests/test_branch_drift.py tests/test_branch_drift_optimization.py tests/test_dflash_reuse.py --basetemp artifacts/pytest-gate-width --junitxml=artifacts/gate-width-local.xml --tb=short
```

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. \
  ../.venv-dflash/bin/python -u -m scripts.run_gate_width
```

运行于独立gate-width-v1目录。确认27B无活动生成后暂时释放GPU1，finally恢复原先active服务；GPU0聊天保持。旧实验、默认服务与infra冻结引用保留。

执行补充：v1在生成前因pytest临时目录父目录缺失失败，保留失败元数据；补齐启动脚本后实际运行于gate-width-v2。正式源3ccf98e，792计时及结果见[结论](GATE_WIDTH_FINDINGS.md)和[完整表](GATE_WIDTH_RESULTS.md)。数据、门槛、系数和反馈规则未因预检失败改变。
