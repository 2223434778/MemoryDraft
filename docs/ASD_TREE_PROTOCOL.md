# 第一阶段：成本受控候选树 + ASD

2026-10-23 GPU前冻结。目标是在当前Qwen3-8B/DFlash上验证第一步组合；不实现剩余regret驱动构树的第二步，不宣称完整官方BASTION复现。

## 来源与实现范围

- BASTION论文 https://arxiv.org/html/2605.29727v1，使用式1/3/6的路径概率乘积、路径质量和best-first思想。
- 上游可见commit `1577b5d9ee28783e15358f1dd791844e9f966172`；LICENSE已核验MIT、Copyright (c) 2026 Z Lab。独立实现，无上游代码导入；复用本项目已有树attention/KV代码。
- 使用本硬件实测表插值代替论文roofline模型。每卡在固定合成128/1024-token前缀，对8/12/16/20/24/32节点各1热身+3计时取中位数；另测一次固定warmup Draft及构树开销。成本表只依赖执行计时，不用评测质量。
- 保留原生16块、最多15 proposal深度，每位置Top4、best-first构造最多32节点；在8..32节点中最大化预期推进/预计cycle成本。每轮最多构建32节点，不依赖全局最优或unimodality假设。在线TopK/归一化/CPU传输/heap/mask/选路/KV整理全部计入wall。
- 树上ASD对每个父→子边使用父节点对应Target logit，原始B=8/g=.25/m=2。局部门控分母为原生窗口剩余位置，与树宽无关；全局选择最长可行连续根路径，同长度优先低regret、再按稳定节点顺序。只扣实际选中路径，EOS后不继续收费。预算0恢复严格greedy；未选分支不能污染KV或Draft特征。

## 冻结评测

四组DFlash、ASD、BudgetTree、BudgetTree-ASD，共同FP32/highest/TF32 off/eager/RMSNorm、原权重、原Target prompt和评分器。无LoRA、无新训练、无历史检索；公共前缀缓存不参与短输入比较，既有工程成果继续保留。

使用六类各8道既有development任务：GSM8K、MATH-500、MBPP、HumanEval、SQuAD、合成JSON。它们均已观察，明确为开发机制评估，不冒充新独立保留集。各族偶数索引给GPU0、奇数给GPU1；每方法自然EOS/4096上限3次、128/EOS2次。共960正式计时+48同题AR，warmup/校准另计。先检查各族前4题的执行有效性，再继续其余题；不根据速度、正确率或方向删题/改参/选择方法。

固定请求内method随机顺序。严格两组必须逐token/text/EOS与同题AR一致；所有重复轨迹/账本稳定，所有组Target prompt和快照一致。模型加载、冷编译与校准单列；正式新图0，峰allocated<=46GiB、编译cache<=4GiB/卡。两进程各7200秒超时，只清理runner自己拥有的进程组。

分别报告：完整请求时延、输出长度/截断/质量、验证轮与节点数；遵守EOS的共同128-token子集；组合相对线性ASD及相同构树的严格组的题等权配对比。原始logit regret不等同质量风险。正确数、lost/gained任务以及小样本区间均保留，不将输出变短自动算算法计算收益。同树额外推进只说明局部机制，不代替真实端到端收益。

统计报告覆盖/重复/严格AR/input/budget检查失败则不发布通过结论。运行source manifest按实际字节逐SHA验证前后，发布只含公开任务输出和小型元数据，不含权重/KV/logits/hidden。

## 命令与输出

本机：`.venv\Scripts\python.exe -m pytest -q tests/test_budget_tree.py tests/test_draft_tree.py tests/test_dflash_reuse.py tests/test_asd_tree_report.py`

服务器新隔离目录：`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_tree_pipeline`

原始结果`results/raw/asd-budget-tree/`；报告重建`python -m scripts.write_asd_tree_report --input results/asd-budget-tree`。执行commit、准确目录、ZIP/source SHA和真实运行记录在PROGRESS及最终报告追加。
