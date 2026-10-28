# 分支漂移门控：实现与七任务评价

本轮直接实现并进行多数据集评价。新方法为研究候选，原BudgetTree-ASD、聊天部署及infra冻结引用保留。新分支`codex/branch-drift`。

## 方法

同一父节点下，近似候选b与Target真实argmax a必须都在已验证树中。比较Target在分别提交a/b后的完整词表下一token分布：固定T=1的JS/ln2，主阈值0.10。仅在原ASD条件和此门控均通过时允许近似边；缺少真实argmax兄弟时拒绝近似；禁止近似绕过或提前提交EOS。严格匹配边保留。后续仍按最长路径、同长度最小regret选择，账本只扣最终路径，KV/特征沿用现有提交代码。

成对树按Draft Top-2兄弟组一起加入，Top-3/4为单节点组；以组总概率排序，复用实测GPU成本选择节点数。所有节点含锚点仍≤32，树深≤15，组不能被截成半对。对照是Target验证后识别的真实首选，Draft Top-2不保证覆盖它。无需新增Target前向，JS计算和构树全部计时。距离为局部诊断，不是整个回答分布的理论界。

## 统一对照

- DFlash：原线性严格验证。
- ASD：原线性DFlash+ASD。
- BudgetTree：原严格成本树。
- BudgetTree-ASD：原组合。
- CoverageTree-ASD：成对树+停止保护+真实兄弟覆盖要求；JS阈值1，计算与主方法一致。
- DriftTree-ASD：同上，JS阈值0.10。

Coverage/Drift直接比较隔离JS门槛影响；相对原组合的变化还包含构树、覆盖与停止保护。原ASD仍B8/gate.25/cap2。Target Qwen3-8B、配套DFlash、FP32/eager、RMSNorm工程底座、线程4、原生block16。读取物理GPU1的旧成本表并核验SHA。模型/提示/评分沿用现有项目。

## 数据与工作量

从既有`asd-paper-extension`公开任务确定性选择，按seed `MemoryDraft-branch-drift-v1`与ID哈希排序；MMLU每学科一题，覆盖57学科。无按输出结果筛选。GSM8K16、MATH-50012、HumanEval16、MBPP16、MMLU57、MT-Bench8个双轮源对话、Alpaca16：141源任务/149匹配输入。

这些任务以前已被评测，本轮是多领域方法评价，不称全新保留集。复用冻结AR token/text供轨迹和质量参考；**不把旧AR时延与本轮时延混成同窗加速比**。MT第二轮共享冻结AR首轮历史，保证同题方法输入一致；不冒充各方法自身历史对话。

每个输入六方法各1次自然EOS生成，上限4096；ASD、严格树、原组合、Coverage、Drift另各1次128/EOS前缀生成。共1639条新计时，单次结果不声称重复稳定性。顺序按请求/阶段确定性打乱。所有任务包括错误/截断均保留，不因数值跨前向形状差异停止其他任务。

## 指标

- 自然完整回答时延、输出长度、验证轮数、吞吐；同题配对时延比，直接读取对应基线，不除两个平均比值。
- 等128仅统计双方实际达到128的请求，不忽略EOS、不强制EOS后续写。
- 数学原评分/既定有限标量等价补充，代码原提供测试，MMLU仅字母输出合同；开放回答不编造正确率。
- 与冻结AR的token完全相同数、首次分歧位置/公共前缀比例，不能把它解释为全序列KL/JS。
- 局部门控候选、实际兄弟缺失、停止拦截、已比较距离/拒绝、实际选中例外、regret。局部候选计数包含最终不可达节点，不当作实际输出比例。
- 对开放任务使用既有固定Skywork Reward V2权重做独立偏好代理，沿用固定基本控制；与官方MT-Bench/AlpacaEval指标区分。

## 运行

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. \
  ../.venv-dflash/bin/python -u -m scripts.run_branch_drift
# 开放质量评分在本机RTX3090执行，输入先从服务器下载。
.venv\Scripts\python.exe -m scripts.run_branch_drift_reward \
  --model-directory artifacts/models/skywork-reward-v2-qwen3-8b
```

服务器隔离目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/branch-drift`。模型目录以实际核实路径为准；聊天8B在GPU0继续运行，空闲27B服务在评测期间暂时暂停，结束后恢复。模型加载、预热和奖励评分单列，在线门控/传输/构树/KV开销计入wall。结果发布于`results/branch-drift/`，失败记录与执行命令同步更新PROGRESS。
