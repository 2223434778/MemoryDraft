# Utility 错误轨迹与计算收益诊断（运行前冻结）

仅开发诊断，两个已观察任务：gsm8k:21（Utility错答23，标准答案14）、math500:7（正确，置信度组曾局部改善）。不新增筛题、不扫描参数、不训练风险模型。本轮不实现未经支持的新在线定价规则。

来源为results/asd-confidence/mixed-b/mechanism/rows.jsonl.gz的ASD-Utility。包含每块所有实际接受的非EOS mismatch，包括第二个例外；按已提交token ID核对并恢复每个状态之前的原始regret。gsm取全部9个；math7固定取第一个、最大regret（平局较早）、最后一个，共3个。不得按分支结果选择状态。

共享Qwen3-8B/DFlash原版本、FP32/eager、greedy、enable_thinking=false、空历史、原prompt、总上限2048、原ASD可行域B8/g.25/M2和Utility价格2。两个GPU各运行一题。模型只读现成快照，不导出权重。

## 两种互补干预

1. **完整策略单次拒绝**：先从原始prompt重现完整Utility输出与账本。每个状态独立从原prompt再跑；到原block/index时，在原候选前截断接受前缀，让既有Target验证输出恢复token。保留所有先前例外的原始支出，不扣被截断尾部，不重置请求预算。之后继续原Utility，允许自然连锁变化。必须核对原block全部proposal、argmax、regret(1e-3)、原接受长度与已花预算；干预前输出逐ID相同，干预位置必须为记录的recovery。拒绝晚于首个mismatch的位置时，保留本块更早已接受的例外。该干预改变一个接受决策，也会丢弃该块后续候选，不能称为后续完全不变的单token因果效应。
2. **同前缀严格分支**：原prompt+源轨迹到例外之前的精确token前缀，分别强制proposal/recovery一个token；两分支重新prefill，后续均严格DFlash。先单独Target前向核对argmax/regret(1e-3)，不复用旧suffix logits。每个分支各运行自然EOS（总2048上限）及最多128 continuation token（遵守EOS）两种范围，固定种子打乱四个执行顺序。前缀可能包含以前的例外；这只测给定前缀下该替换、严格后续的影响，不等于完整Utility策略效果。

合计2次完整baseline、12次完整单次拒绝、48次严格分支，共62次生成（不含独立单步Target核查）。仅1次/条件；两分支都是实际重新计算，不在改token后继续借用旧验证矩阵。质量错误与EOS截断全部保留，不根据结果提前停止。独立Target同状态核查或基线重现失败时按机制错误停止并保存失败，不能凑统计。

## 分析规则

- 质量：完整策略拒绝是否修复/损伤答案；严格自然两分支是否出现正确性差异。若单次拒绝不能修复，不据此声称例外无害，可能有交互或更早错误。本轮不按答案文本回改规则。
- 收益：自然长度、后续验证轮单列；短分支只有双方都生成完整128 continuation token才比较相同token数量的轮密度。重prefill时延仅为诊断（1次、不同前缀分支、包含prefill），不能发布在线加速比或显著性。不将缩短输出当作每token效率提高。
- 信号：同时保存原regret、Run、p_top/p_proposal/pair_mass、Target排名、Draft概率、独立Target熵、剩余预算。寻找质量差异状态时必须同时检查regret/Run是否可比；小样本只描述，不训练或宣称校准。相对概率比与原regret重复，不能包装成新信号。
- 后续决策：若发现明确质量差异，再判断置信度是否在regret/Run之外提供信息；若仅计算轮有差异，优先调查未来计算收益。若无有效信号，报告负结果，不继续扫alpha或把复杂度升级为成功。

## 运行

独立目录 `/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-utility-causal`。命令（各题各一张卡）：

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 ../.venv-dflash/bin/python -u -m scripts.run_asd_utility_causal_probe --config configs/runtime-dflash.json --task-id gsm8k:21 --output results/raw/asd-utility-causal/gsm
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 ../.venv-dflash/bin/python -u -m scripts.run_asd_utility_causal_probe --config configs/runtime-dflash.json --task-id math500:7 --output results/raw/asd-utility-causal/math
```

执行前冻结commit与源码manifest，执行后核验原始证据和源码SHA。公开紧凑输出位于results/asd-utility-causal，不修改之前报告与infra冻结引用。
