# 分支漂移门控：结论与使用边界

实现和七任务评价已完成。**新机制明显提高了与AR的逐token一致性，同时带来速度和质量取舍。保留为独立研究候选，原BudgetTree-ASD继续保留。** 本轮还保留一个更简单的CoverageTree-ASD对照，其MMLU格式恢复值得继续利用。

完整数字见[实测表](BRANCH_DRIFT_RESULTS.md)，规则见[协议](BRANCH_DRIFT_PROTOCOL.md)，原始公开输出与核对证据见[results/branch-drift](../results/branch-drift/)。代码、测试和报告所在分支为`codex/branch-drift`。

## 做了什么

Qwen3-8B和配套DFlash权重沿用。成对树优先把Draft Top-2兄弟一起加入，保留原节点上限32、Top-4、原生15个候选位置和实测GPU成本表。Target仍每轮一次tree attention，分别产生两个兄弟在各自真实祖先上下文下的下一token分布。

近似边必须先通过原ASD B8/gate.25/cap2，再通过以下约束：真实Target首选兄弟存在、不能近似绕过或提前接受EOS、完整词表固定T=1的JS/ln2≤0.10。最后仍选最长可行路径，并只提交该路径KV/特征/预算。CoverageTree-ASD使用完全相同的构树、覆盖要求、停止保护和JS计算，阈值设1；它隔离新增JS门槛的影响。

这不增加单轮Target前向次数；拒绝更多候选仍可能增加总验证轮数，JS/构树本身也有成本，均计入wall。

## 评价覆盖

141个既有公开源任务、149个匹配输入：GSM8K16、MATH-50012、HumanEval16、MBPP16、MMLU57学科各1题、MT-Bench8个双轮对话、Alpaca16。六方法自然生成+五方法128/EOS，共1639条新计时，无按结果删除题目。旧AR输出供身份/质量参考，不借用旧AR时延宣称同窗加速。MT第二轮共享冻结AR首轮历史。

|任务|Drift相对原组合自然时延|正确：原组合→Drift|与AR逐token相同：原组合→Drift|
|---|---:|---|---|
|GSM8K|+10.53%|15→16/16|0→7/16|
|MATH-500|+13.41%|10→8/12|0→3/12|
|HumanEval|+1.01%|13→12/16|11→15/16|
|MBPP|+4.21%|9→9/16|8→14/16|
|MMLU|-2.44%|17→28/57，含仅字母格式合同|37→54/57|
|MT-Bench|-2.50%|独立奖励偏好另列|3→6/16|
|Alpaca|+17.83%|独立奖励偏好另列|1→7/16|

负数更快。数学沿用既定有限标量等价补充，代码沿用原提供测试。每列是对应方法直接同题配对比值的均值，不能把两个相对原组合的均值相除，推导它们之间的时延比。

## 有效的是哪一部分

全149输入的AR逐token完全一致数为：原组合60、Coverage75、Drift106；近似接受token分别538、421、90。因此，覆盖/停止约束已有一部分保真效果，JS门槛又增加31个完全一致输出。这个统计不能替代全序列分布距离或语义正确性证明。

MMLU原组合17→28的合同正确数提升全部也出现在Coverage，Coverage较原组合快3.34%，Drift快2.44%。六方法可提取的首选项逐题相同、均40/57正确，因此这是**输出格式恢复，不是知识正确率提高**；不能独归功于JS，更不能把补充首选项分数替换原评分。停止保护与真实兄弟覆盖尚未再作单独消融。

JS对质量的影响随任务不同：MATH较Coverage正确7→8，HumanEval13→12；GSM/MBPP/MMLU没有新增正确数。它能抑制偏离，却没有成为可靠的正确性判别器。

## 开放回答有可保留的正结果

固定独立Skywork Reward V2，192条均成功评分、8/8基本控制通过；MT按16个匹配轮次评价，Alpaca16题，重复文本缓存后实际136个独立输入。此处为偏好代理，非官方MT-Bench/AlpacaEval。

- MT相对原组合：Drift为8胜3平5负，偏好率59.38%，同时自然时延省2.50%。这是本批可以保留的完整请求正结果。相对Coverage为5胜2平9负、偏好37.50%，时延快4.11%，仍有取舍。
- Alpaca相对原组合：Drift为10胜1平5负，偏好65.62%，自然时延慢17.83%；相对Coverage也是10胜1平5负，时延慢10.81%。可以报告偏好改善与时延代价。
- 相对严格成本树，Drift的MT/Alpaca偏好分别43.75%/46.88%，自然时延分别慢0.77%/1.93%。严格树仍是很强的保真基线。

自然时延和等生成量必须分开看。MT逐题输出长度比均值0.9102，双方达到128的10题上Drift较原组合慢12.24%；自然时延收益伴随回答路径/长度变化，不能写成每token验证吞吐提高。Alpaca长度比均值1.1308，共同128的12题慢13.43%。GSM/MATH共同128分别慢13.20%/5.65%。HumanEval/MBPP共同128都只有1题，不概括全部代码任务。

## 为什么低JS仍可能放过错误

HumanEval76 `is_simple_power`是直接反例。Drift仅接受一次例外，所选兄弟JS=0.042759、regret=2.011280，均通过现有门槛；与AR的首个文本差异把`if x < 1`改成`if x == 1`，后续控制流也改变，原测试失败。具体代码差异与原测试错误在[case-review.json](../results/branch-drift/case-review.json)。

AR会在`(x,n)=(1,12)`返回True，Drift代码的`if x == 1: return n == 1`返回False。下一token分布接近，并不意味着当前比较条件等价。当前例外regret对应的候选/首选概率比约0.134；原ASD按剩余长度归一化允许了该选择，JS没有补足当前token语义的约束。

因此，本轮可确认：**兄弟后续分布是可实现的保真辅助信号，单步JS不构成语义安全证书。** 若继续改进，优先考虑当前token独立的候选/首选概率比下限（等价于regret上限）或关键结构保护；这是后续方向，本轮未据此改阈值或重跑挑选结果。

## 当前决定

保留Drift和Coverage代码、全部正负结果，以及MMLU/MT的适用场景。默认推理与已有infra仍沿用原组合。严格保真需求优先考虑现有严格树：本批149个自然请求全部与冻结AR一致，整体仍有很强竞争力。当前JS点没有显示通用的增量吞吐优势，不据此宣称全面替代原组合。

## 执行与复核

- GPU实际源`d7662c04544ebdb0bc55a8448284f1d34b5b5119`，498个打包文件运行前后SHA均一致。源码ZIP SHA256 `6f8421f045a70778563ba8de0fc9de8da7959ab6e74d1c8c1c0d39c5508f9f05`；打包清单记录working-tree实际字节，不冒充canonical Git blobs。
- 隔离目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/branch-drift-v3`，GPU1；正式生成2026-10-28 12:05:57–13:02:51 UTC，加载20.83秒单列，峰allocated36.03GiB。GPU0的8B聊天继续运行；27B模型/适配服务在末尾恢复，health返回ok。
- 服务器109 passed；本机runtime/SSE集成50 passed；最终核心/评价24 passed。另保留早期102 passed/2 skipped及报告相关45 passed XML，各批有重叠，不相加成独立测试数。
- 1639输入/阶段/方法格、预算账本、所选已许可兄弟计数与JS累计上界、输入/快照及下载文件哈希均核对；严格方法与本批冻结AR身份差异0，严格树与DFlash自然差异0。Windows独立重建analysis与服务器JSON语义相同。
- 初次Linux全零Top-K并列夹具、模型目录缺少只读符号链接、评测适配器递归asdict丢失TreeCosts类型导致的预检查/预热失败均保留。正式行未生成前修复。奖励评分初次遭Windows GBK解码失败，改显式UTF-8后192/192完成；失败元数据也保存，不计入性能结果。
- GPU完成后仅补齐空树索引的显式long dtype，24测试通过；实际记录最小候选horizon为1，正常非空索引原本已为long。另更新离线UTF-8读取和报告元数据，没有更改实测门槛/构树/接受策略或重写原计时。

复现命令与模型/任务固定信息见协议。离线表可用`python -m scripts.write_branch_drift_report --input results/raw/branch-drift`重建；已发布压缩数据可直接运行`python -m results.branch-drift.check_results`核对。该检查同时核对本工作区的main/infra冻结引用；发布包payload SHA见[publication.json](../results/branch-drift/publication.json)。
