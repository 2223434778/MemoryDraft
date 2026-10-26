# DSpark 优化复测结论（2026-10-26）

两项等价优化已完成并实测有效：请求内复用 Markov 偏置，以及用保守上界提前结束构树。相对原 DSpark + 成本树 + ASD，本轮三个任务族完整生成分别省时 **2.55%–4.26%**；112 次配对生成的完整选树摘要、输出 token/文本、停止位置和 ASD 账本一致，没有为提速修改权重、Target 输入或接受条件。

## 实测收益

两张 A6000，同卡配对，24 源任务/28 匹配输入、四组 **448 次主计时**。Alpaca 16 源请求、HumanEval 4 源请求、MT 4 对话/8 个固定共同历史的匹配单轮。自然4096/EOS两遍、前缀128/EOS一遍、固定256/忽略EOS一遍；各请求缓存从空开始，在线填充和淘汰计时。

|任务|优化版较原DSpark树组合：自然用时|优化版较DFlash树组合：自然用时|优化版较原DSpark树组合：固定TPS|
|---|---:|---:|---:|
|Alpaca，N16|−2.55%|−16.88%|+2.74%|
|HumanEval，N4|−4.26%|+44.53%|+4.34%|
|MT匹配单轮，N8|−3.05%|+16.40%|+3.26%|

负的用时变化表示更快。自然用时为逐请求配对比的均值，TPS为总token/总耗时。共同128输出的比较：Alpaca较原DSpark省2.16%、较DFlash省14.86%（N10）；MT较原DSpark省2.59%（N4）；HumanEval较原DSpark省4.19%（N2）。完整比较及线性DSpark-ASD对照见[自动汇总](DSPARK_TREE_OPT_RESULTS.md)。

**Alpaca的16.88%不能全部归于这次工程优化。** 原DSpark树组合在本轮相同16请求中，已经比DFlash树组合平均省时14.68%；这次改动本身带来的是相对原DSpark的2.55%。上一轮七族各4题的Alpaca结果是另一组规模，不能将跨轮差值当优化增量。固定256的优化组合较DFlash总TPS提高12.42%，共同128也有收益，说明本轮优势同时出现在固定工作量中。

## 为什么有效，为什么幅度有限

原构树会在不同深度、不同轮次反复计算同一个父token的全词表Markov偏置。偏置只依赖父token和固定权重，因此可使用请求内LRU64行缓存，保持原GEMV形状、加法顺序与概率计算不变。自然生成首次重复的缓存命中率约为Alpaca55.68%、HumanEval59.73%、MT57.97%，缓存容量上限约37.09MiB，不跨请求预先填充。

原实现通常先建32节点，再选一个较小的树。新实现利用最佳优先搜索的质量上界：若所有后续树规模的最佳可能收益仍不如已有树，就停止搜索，保留原选树。未来各规模逐一检查，向上补浮点误差，不假设成本曲线单调。自然生成中，每轮实际构建节点平均从32变为Alpaca28.85、HumanEval10.35、MT18.58；验证用的树保持相同。

七类任务各2源任务、49个真实GPU状态的四变体消融中，组合构树耗时下降17.87%–67.88%。这是局部环节收益。端到端还包含Target验证、Draft骨干和请求前填充，因此最终省时为2.55%–4.26%。缓存也没有省掉全部条件分布的softmax/topk计算。

**7候选的块长约束仍在。** 本轮HumanEval自然生成，DFlash组合共43轮，DSpark原树及优化树均65轮；优化只让这65轮便宜一些，无法通过等价构树把轮数变成43轮。因此HumanEval仍慢44.53%，MT仍慢16.40%。相比线性DSpark-ASD，优化树在Alpaca和MT省时6.21%/5.38%，HumanEval仍慢2.68%。

## 质量、范围与保留方式

- 优化版与原DSpark树的112次配对生成逐项相同，因此本次改动没有新增质量取舍；这不等于原ASD输出与严格AR相同。HumanEval四组均4/4通过原测试。
- 开放任务没有重做独立裁判。不同Draft之间答案与长度仍可不同，不能只凭速度宣称跨底座质量等价。
- 全部题均来自已观察的扩展数据；12个Alpaca此前未用于DSpark，但不宣称为全新保留集。七族消融只测构树，未新增数学、MBPP或MMLU端到端提速结论。MT不是各方法自身历史的对话级评估。
- 建议保留此DSpark优化分支及Alpaca场景的正结果；DFlash继续作为通用主线。若追求明显更大的DSpark收益，需要解决候选质量/原生短块造成的验证轮数问题，而非继续仅靠构树优化。

## 复核与产物

源码封存提交 `692ce0f72afcb2a706171fdd9d8a312e4b4deddc`，2586文件执行前后SHA一致，正式新增编译图0。服务器164测试通过/2跳过，本机相关回归125通过/2跳过，另9项报告/参考覆盖检查通过。GPU峰值约39.28GiB；不与不同任务规模的旧峰值直接比较。

源码清单、448原始行、28复用参考、224构树消融行、服务器测试和原始报告在[results/dspark-tree-opt](../results/dspark-tree-opt)。结果ZIP为1155114字节，SHA-256 `1297a8c074f597379ddb7a80d2d91cd17358c501367a57e9987409bafb0eaf15`。本机从原始记录重建analysis/report与服务器封存字节完全相同，[审计](../results/dspark-tree-opt/audit.json)记录输入来源、缓存成本和选树计数。

原 `codex/dspark-tree`、`codex/asd-splice` 和面试版引用保留；旧2576文件的Git对象映射单独保护。实现见[dspark_tree_fast.py](../memorydraft/backends/dspark_tree_fast.py)，执行参数与命令见[冻结协议](DSPARK_TREE_OPT_PROTOCOL.md)。

```powershell
.venv\Scripts\python.exe -m results.dspark-tree-opt.helpers.audit
.venv\Scripts\python.exe -m results.dspark-tree-opt.helpers.verify_publication --git
```
