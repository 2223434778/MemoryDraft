# Draft 内部拼接：预计收益调用 OT 与动态长度

2026-10-19。保留 raw Draft prefix + raw history tail。完整矩阵可靠性、二次 OT 求解、矩阵阈值均不参与新策略；旧代码/结果保留为历史记录。

## 冻结决策

1. 精确 suffix≥6 仍优先，跳过 Draft，与原实现相同。近似回退仍至多4次，每次共享至多8个内部锚点候选。Target prompt、greedy严格验证、快照与预算不变。
2. Draft 已执行后，先产生原有廉价 shortlist。输入特征只有第一候选 Edit 分数、与第二候选的分差（单候选记1）、第一候选保留前缀长度、与当前 Draft 的连续一致长度、池大小、最短保留前缀长度。长度归一化32、池大小归一化8。不使用请求ID/项目ID作特征、当前答案、OT分数或矩阵。
3. 校准数据固定为 `results/ot-matrix/offline/frames.jsonl.gz` 的14个已观察状态；结果标签是原 Draft、Edit、TailOT 三条既有策略的已验证连续接受长度。OT这里只负责旧标量 TailOT 排序。不是新训练语料，也不是独立测试集。每次预测排除当前整个项目，其他项目每个只取最近一个状态，再取最近3个项目；平方距离权重 `1/(0.25+distance)`。不按新结果调参。其他项目包含不同用户的公开合成开发统计，只有汇总数值进入模型，历史原文/索引继续按用户隔离。
4. 候选动作：保留原 Draft（15 token），Edit拼接或TailOT拼接，各选16/24/32上限并受剩余预算约束。预计推进量为 `1 + weighted_mean(min(previous_accepted, length))`，即用经验生存曲线处理不同长度；不是把相似度当概率。以前路线曾回退 Draft 的样本仍保留其实际接受量，候选不足时实际块可能更短，估计有误差。
5. A6000 单独测量 context256/768/1536、验证15/16/24/32候选的 Target 成本及16 pending特征的 DFlash 成本，2次预热后7次中位数；另测共享 shortlist+policy 与每候选TailOT CPU成本。上下文取最近档、长度线性插值。成本包括模型完整词表输出；不是生产负载延迟拟合。
6. 比较 `预计推进量 / (Draft成本 + Target验证成本 + shortlist/policy成本 + 额外OT成本)`。同轮Draft已经付出；这里将它作为未来循环的摊销成本，因此该式是 renewal-rate 近似，并非当前轮“省掉Draft”。预测OT只有超过最佳廉价动作2%才调用；不足3个校准项目或单候选不调用OT。OT分数仅用于真正调用后的排序/.65候选门槛，不作为接受概率。
7. 不根据本次请求反馈更新校准模型；不读取当前答案作在线决策。对当前请求的误预测也原样保存。不缩短 Draft 前向本身，不改变 KV、严格验证或冷却反馈。

## 对照和运行顺序

- Suffix-R、Guided-Edit、Splice-Edit、Splice-TailOT。
- Gate-OT：预计收益路由，拼接长度固定32。
- Budget-Edit：预计收益选择 Draft/Edit 与长度，禁止OT。
- Budget-OT：预计收益选择 Draft/Edit/OT 与长度。

固定与完整矩阵实验相同的8个已知开发请求，7方法×8请求机制检查一次=56次，先严格比对AR token IDs、Target输入、快照。策略有实际决策且检查通过后再两次交错端到端计时=112次。决策/候选日志只在机制运行收集；计时包含加载后每请求的快照、tokenization、索引构建、策略校验、检索、OT、全部Draft/Target、输出整理等实际在线成本，模型加载和成本校准单独报告。

离线检查对14个状态做排除当前项目的预测，记录预测误差，不用此误差挑参数。只能判断小样本机制是否有价值，不能声称真实负载或独立泛化。新的解码路径若遇到旧样本没有覆盖的状态，预测可能很差；即使预计收益为正，也须以端到端计时为准。短块也可能增加未来轮数，局部rate未显式建模未来精确suffix与pending状态。

参考：SVIP（[Draft Model Knows When to Stop](https://arxiv.org/abs/2411.18462v2), EMNLP2025）支持按动态接受情况调整候选长度的研究方向。本实现采用项目排除的经验接受长度及实测成本，并非复现其熵阈值；DFlash并行位置的分布不直接当作校准的Target接受概率。

## 命令（独立服务器源码目录）

```bash
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONPATH=.
PY=/media/a6000/D/ZhaoBinHe/MemoryDraft/.venv-dflash/bin/python
$PY scripts/profile_benefit_costs.py --config configs/runtime-ot-benefit.json --output results/raw/ot-benefit/cost
$PY scripts/probe_benefit_offline.py --costs results/raw/ot-benefit/cost/costs.json --output results/raw/ot-benefit/offline
$PY scripts/run_benefit_probe.py --stage mechanism --config configs/runtime-ot-benefit.json --costs results/raw/ot-benefit/cost/costs.json --offline results/raw/ot-benefit/offline --output results/raw/ot-benefit/mechanism
$PY scripts/run_benefit_probe.py --stage timing --config configs/runtime-ot-benefit.json --costs results/raw/ot-benefit/cost/costs.json --offline results/raw/ot-benefit/offline --mechanism results/raw/ot-benefit/mechanism --output results/raw/ot-benefit/timing
```

