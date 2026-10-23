# 第二阶段：计算成本与剩余regret联合构树

2026-10-23，GPU运行前冻结。沿用[第一阶段](ASD_TREE_PROTOCOL.md)的模型、数据、成本表和接受规则，新增两个构树消融。不新增训练或扫描参数，不覆盖旧证据与面试版本。

## 算法与来源

复用BASTION式路径概率和best-first成本受控树，以及ASD实际选中路径的累计logit regret。来源为[BASTION](https://arxiv.org/html/2605.29727v1)、[ASD](https://arxiv.org/html/2608.03447v3)，本轮没有引入新的上游代码。第一阶段成本模型选择8..32节点，但有些树提前分叉、未保留完整线性Draft；第二阶段让实际剩余预算决定必须保留多长原始主干，剩余节点仍按路径概率扩展。

三种树分配：

- BudgetTree-ASD：第一阶段原best-first，保持原来节点、顺序和路径选择。
- SpineTree-ASD：先保留原Draft argmax主干，直到窗口末端/EOS/节点上限，再扩best-first分支。隔离“保留连续主干”的效果。
- JointTree-ASD：先保留预算决定的原始主干，再扩best-first。隔离动态分配相对固定主干的增量。

设原生proposal窗口K（通常15），局部门控g=.25，单块例外cap=m=2，u=min(m,K)。原局部门控下，单块可接受regret上界C=g*u*(2K-u+1)/2。实际剩余预算R=max(0,B-spent)，B=8。强制主干长度为ceil(K*min(1,R/min(B,C)))；若K/B/R/g/m为零则长度0。K15时C=7.25，R=8/7.25/3.625/.01/0对应15/15/8/1/0。EOS是叶子，实际保留可更短。

这只是无需训练的资源尺度启发式，不把Draft概率当Target正确率，也不声称C是预计实际花费。预算充足时完整主干保留线性ASD的可行选项；预算下降时将更多节点交给分支，零预算恢复原best-first构树。强制主干计入节点上限和在线构树开销；成本目标只在包含强制主干的候选前缀中比较。仅实际选中路径收费，未选分支不扣费。

## 冻结比较

六组：DFlash、ASD、BudgetTree、BudgetTree-ASD、SpineTree-ASD、JointTree-ASD。相同Qwen3-8B/原生16 DFlash、FP32/highest/TF32 off/eager/RMSNorm、相同Target输入、相同树上ASD B8/g.25/m2与最长可行路径选择。只改构树；不改接受阈值、质量评分或prompt。原四组逐token/text/轮次/账本/停止原因/输入SHA/快照复现第一阶段；全部960次检查，计时允许变化。

六类各8道既有development：GSM8K、MATH-500、MBPP、HumanEval、SQuAD、合成JSON。这48题已观察，只作机制开发评估。偶数索引GPU0、奇数GPU1；自然EOS/4096上限各3次、128/EOS各2次，共1440计时+48同题AR。每卡720计时，pilot24共720计时后核执行有效性，再完成其余24。无基于速度/质量的提前停止或重选，保留全部负结果。

冻结各物理卡第一阶段calibration.json及publication SHA；无本轮重新校准，不用新结果调成本表。实际wall包含新增快照、主干、TopK/传输/heap/mask、验证、选路、KV和包装开销。模型加载、冷编译和读取冻结表成本另列；每卡10次预热生成（AR/DFlash各1、六组各1、zero/strict各1），无额外校准生成。

报告自然时延、长度、质量、截断、验证轮、验证节点；等128遵守EOS，只比较双方完整128的题。Joint分别比较线性ASD、第一阶段组合与固定主干消融；原四组本轮计时用于匹配比较，禁止拿跨批计时直接充当新收益。逐轮核对剩余预算快照、主干长度和实际扣费。评分正确只代表既定评分器，不保证完整推理无错；新质量变化单独复核。

正式新编译图0，allocated峰<=46GiB、cache<=4GiB/卡，两owned worker各7200秒，outer7600秒。运行源码全部canonical Git blobs前后SHA核验，公开小型输出/元数据，不包含权重、KV、logits或hidden。失败不覆盖旧源或旧结果。

## 命令与位置

本机检查：`.venv\Scripts\python.exe -m pytest -q tests/test_joint_tree.py tests/test_asd_joint_tree.py tests/test_asd_tree_report.py tests/test_budget_tree.py tests/test_draft_tree.py tests/test_dflash_reuse.py`。

冻结提交后打包：`.venv\Scripts\python.exe -m scripts.package_asd_tree_source --output artifacts/asd-joint-tree-source.zip --manifest ASD_JOINT_TREE_SOURCE_MANIFEST.json`。

服务器新目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-joint-tree`：`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_joint_tree_pipeline`。

原始`results/raw/asd-joint-tree/`，公开`results/asd-joint-tree/`；重建`python -m scripts.write_asd_joint_tree_report --input results/asd-joint-tree`。准确commit、SHA、实际测试/执行窗口和结论追加至PROGRESS与结果报告。
