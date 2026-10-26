# DSpark + 成本树 + ASD：八组对照结果

官方 DSpark 检查点、按父 token 计算条件分数的成本树、树上 ASD 和 KV/特征提交已经接通，并完成真实 A6000 比较。**原 DFlash + 成本树 + ASD 保留为当前主方案；DSpark 组合在本轮 Alpaca 上取得额外收益，作为自由生成场景的候选保留。**

本轮比较已观察的七类任务各4个源任务，共28源任务、32匹配输入（MT为4个双轮对话），完成1024次主计时和32条自然AR参考。这是小规模底座回放比较，原先514源任务的DFlash结果继续有效，并保持原文件与分支。

## 1. 换成 DSpark 后，实际发生了什么

每种Draft分别运行严格线性、线性ASD、严格成本树、成本树+ASD四组。下表直接比较两种完整组合；负数表示DSpark更快，正数表示更慢。

|任务|匹配输入|DSpark组合相对DFlash组合的自然时延变化|原评分正确数：DFlash→DSpark|
|---|---:|---:|---|
|GSM8K|4|+16.61%|3→3/4|
|MATH-500|4|+44.95%|2→2/4|
|HumanEval|4|+50.90%|4→4/4|
|MBPP|4|+22.81%|3→3/4|
|MMLU|4|−1.48%|3→3/4|
|MT-Bench|8轮|+20.14%|本轮未独立裁判|
|Alpaca|4|**−2.63%**|本轮未独立裁判|

自然时延采用逐请求配对时延比的均值。MMLU很短，本轮接近持平；不能把该4题切片解释成普遍优势。完整八组自然耗时、质量、轮次和固定TPS见[完整结果](DSPARK_TREE_RESULTS.md)，原始主表见[服务器报告](../results/dspark-tree/report.md)。

## 2. 成本树在 DSpark 上仍然能贡献收益

换底座的结果与模块在该底座上的增量，是两个分别测量的问题。

|任务|DSpark树+ASD相对DSpark线性ASD的自然时延变化|固定256总TPS变化|
|---|---:|---:|
|GSM8K|−6.78%|−4.01%|
|MATH-500|+31.78%|−3.80%|
|HumanEval|+7.18%|−4.33%|
|MBPP|+1.61%|−4.49%|
|MMLU|−0.91%|−0.96%|
|MT-Bench|−2.26%|+5.96%|
|Alpaca|**−10.45%**|**+20.76%**|

Alpaca的增量有完整的计算证据：DSpark线性ASD的实际Draft接受长度2.632→树+ASD的3.917，验证轮152→120；输出token合计552→590，回答更长仍省时。遵守EOS的共同128-token子集（2题）也省时14.55%。这支持成本树与ASD可以在DSpark上形成有效组合。

相对原DFlash组合，Alpaca的实际Draft接受长度2.625→3.917、验证轮144→120，输出522→590；共同128子集（2题）省时11.78%，固定总TPS提高10.71%，自然请求省时2.63%。这些指标共同呈现了该场景的正结果。

GSM的DSpark内部自然收益则伴随输出1160→1073，而共同128慢6.36%、固定TPS下降4.01%；应保留其真实请求收益，同时区分路径变短与每token计算提速。

## 3. 为什么 DSpark 没有自动成为全面升级

### 原生窗口不同，长接续场景的上限也不同

可用官方检查点是`dspark_qwen3_8b_block7`：一次提供7个候选，树的最大proposal深度也是7。旧DFlash保留原生16位置窗口，包含锚点和15个候选。两种Draft均保持原生长度。

这个差别在代码和长数学解答中非常直接：

|任务|DFlash组合实际Draft接受长度|DSpark组合实际Draft接受长度|验证轮：DFlash→DSpark|
|---|---:|---:|---:|
|HumanEval|10.233|6.262|43→65|
|MATH-500|10.431|6.348|633→910|
|Alpaca|2.625|3.917|144→120|

HumanEval和MATH中，旧组合已经能平均一次采用超过7个Draft token，而当前DSpark检查点的整轮候选上限就是7。在Alpaca这类原接受长度较低的场景，DSpark反而能够增加实际推进量。这解释了为什么底座更换的效果随任务改变。

### 条件树多了一笔需要计算的成本

DFlash树可以直接复用同一次起草的各位置分布。DSpark树的后续分数依赖父token；当前实现按需要计算Markov修正，并缓存相同“深度、父token”的条件分布。

实际校准中，两种骨干前向都约8.3ms；DFlash构树辅助成本约0.58ms，DSpark条件构树约9.3ms。后者包括额外的条件投影、归一化、Top-K和同步。这个校准是成本线索，不是逐项在线profile的因果分摊；正式请求已将所有这些成本计入。

因此，树的接受增量要足够大，才能覆盖这笔新增成本。Alpaca本轮覆盖了它；一些原本就有长接续的任务没有覆盖。后续若优化DSpark树，优先考虑批量条件投影/候选预取，并把长窗口作为独立实验检查。

### 论文中的 Draft 质量优势有具体配置

[DSpark论文](https://arxiv.org/html/2607.05147v1#S4)的主比较采用统一重训的起草器、7候选、温度1，报告的是接受长度优势。本轮采用贪心解码、旧Z Lab DFlash-15检查点与官方DSpark-7检查点，训练来源和窗口同时不同，不能把论文中的接受长度提升直接当作本项目的时延增益。

DSpark本身也有正结果：严格线性DSpark相对严格DFlash在本轮MBPP省时10.81%、Alpaca省时15.64%，两者输出都与AR相同。这些结果支持按检查点、窗口和条件树开销，为具体工作负载选择底座。

## 4. 质量和统计口径

五类可自动评分任务的组合正确数分别相同，但MATH正确集合不同：156从正确30度变为错误60；327从DFlash组合4096截断、没有最终答案，变为DSpark组合正常结束、最终15正确。原评分全部保留，没有根据新答案改判。MATH表采用原先规定的有限标量等价补充口径。

HumanEval固定256总TPS中DSpark组合比DFlash组合高14.47%，但自然请求慢50.90%。约53%的固定token在首次出现EOS ID之后；此外，“总token/总时间”和“逐题时延比均值”是不同统计量。固定TPS与正常代码请求时延必须分别阅读。

MT第二轮统一使用本轮AR首轮回答，比较8个匹配轮次；不混入各方法自身历史，也不把轮次区间当作独立对话级区间。本轮开放回答保留，未新增奖励模型或人工质量裁判，开放时延收益不代表已证明质量保持。

## 5. 接入正确性、保留和复现

- 官方模型：`deepseek-ai/dspark_qwen3_8b_block7@03326e5043815da1f81b109078b2889737c26017`；下载与服务器加载均核验完整权重SHA。共享Target的嵌入/输出头之前，逐块核验与检查点完全相等。
- 参考源码：DeepSpec `005e03b81cec38b7da6399833d609ee89a2587f2`。两卡各检查context4/17，适配器与官方骨干输出最大差均0、Markov greedy token相同；原文件Git blob与派生SHA公开。
- 本机专项113通过/2跳过，冻结前补充20通过；服务器151通过/2跳过。自然严格四组共128个对照全部与AR逐token相同，重复质量无变化，正式新增编译图为0。
- 源码冻结`62e801067dcb990ea9ae9a18ed1c09f02e0d04ab`，2531文件前后SHA一致。原始ZIP1600197字节，SHA256 `cbf14e2dbaa69513c9c60203c535f6ac70a4b941301dc14e26c78efe4d985dd7`；统计JSON和Markdown已在Windows从原始行重建，逐字节相同。
- 2026-10-26 14:25启动，15:00确认流水线完成、主进程退出、两卡19MiB/0%。正式峰allocated约40.128GiB；硬件观察到高温，未调整风扇/频率/功率；题内随机/反序配对，两卡各跑完整8组。
- 原`codex/asd-splice`仍为`8e7a80b5b1af3b09886659e3549691be3ce6493e`；1851份旧配置/结果/核心实现Git对象一致，旧174份发布文件及面试版引用核验通过。新实现、模型目录、数据与报告均独立保存。

协议和运行入口见[DSPARK_TREE_PROTOCOL](DSPARK_TREE_PROTOCOL.md)、[run_dspark_tree.py](../scripts/run_dspark_tree.py)、[流水线](../scripts/run_dspark_tree_pipeline.py)。[完整八组表](DSPARK_TREE_RESULTS.md)、[原始数据](../results/dspark-tree/analysis.json)、[质量与保留审计](../results/dspark-tree/derived-audit.json)、[重建校验](../results/dspark-tree/local-checks/report-reconstruction.json)可直接复核。

复核命令（仓库根目录）：

```powershell
.venv\Scripts\python.exe -m results.dspark-tree.helpers.finalize_results
```

服务器完整重跑需新建隔离源码目录、准备独立DSpark模型目录，并将已公开的`reference-source/modeling.py`和`markov_head.py`放入新root的`artifacts/dspark-upstream/`；原结果目录有防覆盖检查。完整启动命令记录在PROGRESS。
