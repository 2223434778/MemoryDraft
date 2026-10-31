# DFlash + 成本树 + ASD：实验总览

> 直接查看：[所有基线 × 所有数据集大总表](#all-baselines)。另有[当前方案15类任务明细](#current-results)、[九组同机基线汇总](#baseline-results)。当前方案为`BufferedKV-ASD`：DFlash + 高保真反馈成本树 + ASD + 最小增量接受 + 分数/KV复用。历史`BudgetTree-ASD`结果与最新方案分别保留。

更新于2026-10-31。这份文档汇总已经完成的实验，不新增GPU运行，也不重新挑选任务或参数。第4节历史大样本表及第5–7节原组合分析保留2026-10-28的结果；最新方案与新增数据见第4.1–4.2节。

**最新方案的收益已覆盖指令遵循、代码编辑和摘要。** 新增182请求/546生成中，混合总耗时较DFlash/ASD降低12.96%/12.16%，DFlash输出一致146/182对ASD31/182；CanItEdit正确17/31对ASD15/31，AIME25为5/30对ASD7/30。长文短答案的收益很小，摘要有明显提速且保留ROUGE-L变化与截断，不能概括为所有任务质量或速度都提高。

2026-10-28的原组合也保留独立成果：七任务独立扩样中，组合相对普通ASD在GSM8K、MBPP、Alpaca的自然回答时延分别降低7.72%、10.21%、17.89%；MT-Bench按各方法自身历史进行真实双轮生成时降低12.93%。严格成本树也能独立改善DFlash的速度，且保留本轮DFlash输出。MATH、HumanEval和MMLU存在不同的速度或质量取舍，完整保留在下表中。

分阶段报告、原始结果、代码和部署验收均已提交GitHub。原总览发布于 `codex/chat-live-draft`；分支漂移续研附注与新报告位于 `codex/branch-drift`。仓库默认 `main` 保留旧版本，直接查看默认首页不会看到这里的最新内容。

<!-- ALL_BASELINES_MATRIX_START -->
<a id="all-baselines"></a>

## 0. 所有基线 × 所有数据集：一张大总表

**行是数据集/协议，列是方法。** 本表覆盖当前正式比较的九组同机方法及已有引用的EAGLE-3，共10列方法、15行本机任务，另加8行论文配置。它同时展示已有结果和缺口；并非每个方法都已在每个数据集上运行。Joint、DSpark、Run/Utility等历史候选仍在对应历史报告，不冒充这两批匹配题上的新基线。

- **实测格：第一行=相对同批DFlash的逐题平均耗时变化；第二行=质量。** 负数更快。质量通常为正确数/可评分数，Qasper/MultiFieldQA为F1×100，MultiNews/GovReport为ROUGE-L×100；“质量未评”只表示本批没有质量评分。
- **A批**：原七任务66匹配开发输入，A6000/FP32/eager；九组594生成。**B批**：新增182请求，同机同底座三组546生成。B的AR/其他树方法没有匹配新结果，标“未测”；A的EAGLE-3也未实测。
- **C批“文献”格：相对AR的per-token latency倍率及τ**，不是前两批的耗时变化。它来自A100/BF16/Qwen3-8B原文，单独配置行，禁止跨A/B/C直接排名。未录表示既有引用表没有收录该格，未伪造准确率。
- A的MT/Alpaca质量未新增评测；MMLU为字母输出合同。B的CanItEdit32题计时、31题可评分，LongBench19/32输入首尾截取；除AIME25全30题外均为子集。截断、输出一致性、TPS、TTFT与各题明细见第4.1节和原报告。

|数据集（批次；生成请求数）|AR|DFlash|ASD（DFlash）|严格成本树|原树+ASD|当前方案|DDTree-32|DDTree-128|BASTION|EAGLE-3|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|GSM8K（A；8）|+355.12%<br>8/8|+0.00%<br>8/8|-11.94%<br>8/8|-11.32%<br>8/8|-9.70%<br>7/8|-10.90%<br>8/8|-9.77%<br>8/8|+53.22%<br>8/8|-9.42%<br>8/8|未测|
|MATH-500（A；12）|+473.28%<br>8/12|+0.00%<br>8/12|+24.16%<br>8/12|-5.32%<br>8/12|+44.43%<br>10/12|-7.71%<br>9/12|-4.05%<br>8/12|+65.58%<br>8/12|-2.77%<br>8/12|未测|
|HumanEval（A；8）|+412.97%<br>7/8|+0.00%<br>7/8|+0.65%<br>7/8|-5.53%<br>7/8|-6.50%<br>7/8|-4.35%<br>7/8|-4.16%<br>7/8|+54.26%<br>7/8|-3.87%<br>7/8|未测|
|MBPP（A；8）|+150.89%<br>4/8|+0.00%<br>4/8|-12.56%<br>5/8|-18.72%<br>4/8|-23.38%<br>5/8|-16.50%<br>4/8|-18.65%<br>4/8|+26.62%<br>4/8|-18.63%<br>4/8|未测|
|MMLU（A；14）|+129.77%<br>5/14|+0.00%<br>5/14|+2.37%<br>1/14|-1.70%<br>5/14|+0.58%<br>2/14|-2.95%<br>5/14|-0.15%<br>5/14|+45.73%<br>5/14|-0.07%<br>5/14|未测|
|MT-Bench（A；8）|+291.70%<br>质量未评|+0.00%<br>质量未评|-0.18%<br>质量未评|-12.01%<br>质量未评|-12.83%<br>质量未评|-14.11%<br>质量未评|-11.78%<br>质量未评|+45.63%<br>质量未评|-10.99%<br>质量未评|未测|
|Alpaca（A；8）|+128.98%<br>质量未评|+0.00%<br>质量未评|-12.05%<br>质量未评|-17.36%<br>质量未评|-26.34%<br>质量未评|-16.65%<br>质量未评|-17.71%<br>质量未评|+38.73%<br>质量未评|-17.80%<br>质量未评|未测|
|IFEval（B；64）|未测|+0.00%<br>52/64|+2.69%<br>53/64|未测|未测|-15.33%<br>52/64|未测|未测|未测|未测|
|LiveCodeBench v6（B；24）|未测|+0.00%<br>8/24|+46.94%<br>9/24|未测|未测|-7.34%<br>8/24|未测|未测|未测|未测|
|CanItEdit（B；32）|未测|+0.00%<br>17/31|-2.65%<br>15/31|未测|未测|-3.22%<br>17/31|未测|未测|未测|未测|
|Qasper（B；8）|未测|+0.00%<br>43.14|-0.68%<br>45.38|未测|未测|-1.59%<br>43.14|未测|未测|未测|未测|
|MultiFieldQA-en（B；8）|未测|+0.00%<br>55.60|-1.90%<br>56.00|未测|未测|-2.46%<br>55.60|未测|未测|未测|未测|
|MultiNews（B；8）|未测|+0.00%<br>26.46|-6.88%<br>26.41|未测|未测|-20.43%<br>26.46|未测|未测|未测|未测|
|GovReport（B；8）|未测|+0.00%<br>34.87|-3.69%<br>35.48|未测|未测|-19.48%<br>34.61|未测|未测|未测|未测|
|AIME25（B；30）|未测|+0.00%<br>5/30|-2.61%<br>7/30|未测|未测|-8.78%<br>5/30|未测|未测|未测|未测|
|GSM8K（C；论文配置）|参考1.00×|文 4.96×<br>τ=6.51|未录|未录|未录|未录|未录|未录|文 6.92×<br>τ=8.85|文 3.10×<br>τ=2.74|
|MATH-500（C；论文配置）|参考1.00×|文 6.06×<br>τ=8.25|未录|未录|未录|未录|未录|未录|文 8.02×<br>τ=10.53|文 2.78×<br>τ=2.51|
|AIME25（C；论文配置）|参考1.00×|文 5.91×<br>τ=7.20|未录|未录|未录|未录|未录|未录|文 7.69×<br>τ=9.46|文 2.92×<br>τ=2.45|
|HumanEval（C；论文配置）|参考1.00×|文 5.25×<br>τ=6.64|未录|未录|未录|未录|未录|未录|文 7.64×<br>τ=9.44|文 3.01×<br>τ=2.69|
|MBPP（C；论文配置）|参考1.00×|文 4.91×<br>τ=6.04|未录|未录|未录|未录|未录|未录|文 7.10×<br>τ=8.65|文 2.73×<br>τ=2.24|
|LiveCodeBench（C；论文配置）|参考1.00×|文 5.90×<br>τ=7.64|未录|未录|未录|未录|未录|未录|文 7.64×<br>τ=9.91|文 2.29×<br>τ=1.91|
|MT-Bench（C；论文配置）|参考1.00×|文 2.81×<br>τ=4.46|未录|未录|未录|未录|未录|未录|文 4.31×<br>τ=6.28|文 2.68×<br>τ=2.24|
|Alpaca（C；论文配置）|参考1.00×|文 2.28×<br>τ=3.15|未录|未录|未录|未录|未录|未录|文 3.54×<br>τ=4.68|文 2.10×<br>τ=1.77|

**覆盖情况：87个本机实测格、63个本机未测格；C另收录24个论文方法格和8个AR参考格。** “未测”说明本轮匹配对照缺失，不代表该方法无效或项目历史中从未运行过。

A来源：[实测完整表](EXTERNAL_TREES_RESULTS.md) · [统计](../results/external-trees/analysis.json)；B来源：[实测完整表](EXPANDED_SUITE_RESULTS.md) · [统计](../results/expanded-suite/analysis.json)；C来源：[已有文献引用与适用条件](EXPANDED_LITERATURE_BASELINES.md)，原文[BASTION v4 Table 1](https://arxiv.org/html/2605.29727v4#S4.T1)。历史原组合大样本与其他候选仍按各自数据/模型设置保留，不用跨批数值补齐空格。
<!-- ALL_BASELINES_MATRIX_END -->

## 1. 这套方法是什么

原组合的实验名称是 **BudgetTree-ASD**；后续高保真反馈门控与执行优化版本记作 **BufferedKV-ASD**。底座为Qwen3-8B Target和与它配套的官方DFlash Draft，原生block=16：一个已确认锚点，最多15个候选位置。

```text
已确认前缀与Target特征
  → DFlash并行预测各位置的候选分布
  → 每位置取Top-4，按概率与实测GPU成本选择8～32节点的候选树
  → Target用tree attention一次验证多个分支
  → ASD在请求剩余预算内选择可行的最长连续路径
  → 只提交这条路径对应的token、KV、特征和预算
```

树覆盖线性Top-1草稿没有提供的替代接续；ASD允许在预算内越过少量分歧，继续利用后续候选。每条分支的Target分数来自它自己的祖先上下文，不能串用其他分支的验证结果。

构树借鉴BASTION的best-first路径概率思想，结合每张GPU实测节点成本；ASD来自原论文，DFlash复用官方模型。项目完成的是这些机制的适配、组合、状态提交、多任务评测和infra落地。构树只使用请求当时可用的Draft预测、成本表和预算，不使用未来Target分数或标准答案。详细解释见[通俗架构说明](ASD_TREE_ARCHITECTURE.md)、[第一阶段实现与来源](ASD_TREE_FINDINGS.md)。

## 2. 对照组与统一设置

|名称|实际含义|验证规则|
|---|---|---|
|AR|只用Target逐token生成|严格贪心|
|DFlash|DFlash线性草稿|严格最长一致前缀|
|ASD|同一DFlash线性草稿 + ASD|允许预算内近似接受|
|BudgetTree|DFlash + 成本受控树|严格沿树匹配|
|BudgetTree-ASD|DFlash + 成本受控树 + ASD|允许预算内近似接受，原组合|
|JointTree-ASD|进一步让剩余regret预算参与主干／树形分配|研究候选与消融|
|BufferedKV-ASD|高保真反馈成本树 + ASD + 最小增量接受 + 分数/KV复用|当前方案；JS与当前token分歧约束、请求内反馈调宽，近似路径至少多推进2个token才接受|

主要研究使用两张A6000，各卡独立运行组内对照；不是两卡并行加速一个请求。共享Target/Draft、输入、FP32、eager attention及RMSNorm工程底座；ASD主参数为 `B=8, gate=.25, cap=2`。模型加载、预热、离线校准和离线质量评分单列；在线构树、Top-K、传输、mask、选择、验证和KV整理均计入请求wall。

RMSNorm等工程改进在各对照间共享，其收益不重复计入树的算法增量。近似ASD可能改变中间token、回答长度和最终答案，任务正确率与逐token身份分别检查。

## 3. 已完成哪些实验

|批次|任务范围|完成量|报告与原始结果|
|---|---|---|---|
|成本树 + ASD开发检查|六类各8题，共48个已观察开发请求|四组960次计时 + 48条AR参考|[结论](ASD_TREE_FINDINGS.md) · [结果](../results/asd-budget-tree/)|
|剩余预算联合构树|同一48题，增加Joint与主干消融|六组1440次计时 + 48条AR参考|[结论](ASD_JOINT_TREE_FINDINGS.md) · [结果](../results/asd-joint-tree/)|
|首轮七任务扩样|241个新源任务，249个匹配输入|五组4980次计时 + 249条AR参考|[结论](ASD_PAPER_FINDINGS.md) · [结果](../results/asd-paper/)|
|AR／接受长度／开放质量补全|复用上述249个输入|747次新固定长度计时、120条独立奖励评分|[结论](ASD_PAPER_METRICS_FINDINGS.md) · [结果](../results/asd-paper-metrics/)|
|更大规模独立扩样|514个新源任务，530个匹配输入，七任务四组|9540条主计时、530条AR参考、640次预算控制、64次自身历史第二轮生成、320条奖励评分|[结论](ASD_PAPER_EXTENSION_FINDINGS.md) · [结果](../results/asd-paper-extension/)|
|infra v0.4部署验收|6个已观察公开请求，五方法两重复|60次计时、12次研究／运行时对照；本机1144测试、服务器专项150测试通过|[验收](INFRA_TREE_VALIDATION.md) · [结果](../results/infra-tree-v04/)|
|分支漂移门控|149个匹配输入|1639次计时、192次奖励评分|[结论](BRANCH_DRIFT_FINDINGS.md) · [结果](../results/branch-drift/)|
|按需JS与节点分配|66个已观察匹配输入|924次计时，132对eager/lazy决策一致|[结论](BRANCH_DRIFT_OPTIMIZATION_FINDINGS.md) · [结果](../results/branch-drift-optimization/)|
|反馈调宽|66个已观察匹配输入|792次计时|[结论](GATE_WIDTH_FINDINGS.md) · [结果](../results/gate-width/)|
|最小增量接受|66个已观察匹配输入|198次计时|[全表](GATE_MIN_GAIN_RESULTS.md) · [结果](../results/gate-min-gain/)|
|分数/KV执行复用|66个已观察匹配输入|198次计时，132对输出/路径/预算一致|[全表](TREE_EXECUTION_RESULTS.md) · [结果](../results/tree-execution/)|
|官方外部树基线补全|既有62开发源题/66匹配输入，七任务九组|594次自然生成，含本窗口66次AR|[结论](EXTERNAL_TREES_FINDINGS.md) · [全表](EXTERNAL_TREES_RESULTS.md) · [结果](../results/external-trees/)|
|新场景扩展|182个新请求，八个数据集/子集，三组|546次自然生成，编辑输出统一离线重评|[结论](EXPANDED_SUITE_FINDINGS.md) · [全表](EXPANDED_SUITE_RESULTS.md) · [结果](../results/expanded-suite/)|
|infra v0.5部署验收|七任务研究等价、真实CLI/SSE、可视化与取消恢复|本机1299测试通过，A6000专项145测试通过|[验收](INFRA_FIDELITY_VALIDATION.md) · [结果](../results/infra-fidelity-v05/)|

这些批次有任务复用，不能把各行样本量直接相加后宣称独立问题总数。计时重复、质量评分也不是新增独立问题。最后一轮9540是原始计时行；将严格前／后控制合并后，`analysis.json.timed_generations`为8480，`raw_timed_generations`仍为9540。

## 4. 历史主表：原组合最大一轮独立扩样

以下全部使用**自然停止的完整回答时延**。变化率为同题配对时延比的均值，负数更快、正数更慢。三列分别直接读取与对应基线的配对统计，不能将不同列的均值相除来推导另一个基线。

|任务／匹配输入数|组合 vs ASD|组合 vs DFlash|严格成本树 vs DFlash|自然质量：ASD → 组合|
|---|---:|---:|---:|---|
|GSM8K／128|-7.72%|-14.90%|-10.25%|114 → 117／128|
|MATH-500／64|+5.50%|+2.44%|-6.19%|44 → 45／64，含有限标量等价补充|
|HumanEval／64|-3.26%|-1.46%|-2.64%|53 → 52／64|
|MBPP／64|-10.21%|-20.15%|-14.87%|31 → 32／64|
|MMLU／114|+1.30%|+3.41%|-0.09%|39 → 37／114，仅字母输出合同|
|MT-Bench／32|-5.51%|-13.64%|-13.92%|开放回答，单列偏好与任务性质|
|Alpaca／64|-17.89%|-19.67%|-17.15%|开放回答，单列偏好与任务性质|

来源：[analysis.json](../results/asd-paper-extension/analysis.json)的 `families.<task>.<method>.natural.<baseline>.ratio`，质量来自同一自然组及[质量复核](../results/asd-paper-extension/quality-audit.json)。完整结果还包括区间、每题长度、截断和失分／得分集合，见[完整表](ASD_PAPER_EXTENSION_RESULTS.md)。

- MATH的保守主评分为42 → 43／64；44 → 45使用既定有限标量等价补充，不能混成另一套新评分。
- MT-Bench为16个源对话、32个匹配轮次。主表第二轮共享同题严格AR首轮历史；按对话合并两轮，组合相对ASD省10.12%。各方法使用自身真实历史另行生成时省12.93%，两种输入定义分别报告。
- 严格成本树与DFlash在530个自然请求上输出token、文本、停止一致；包含其他阶段共1590组对照。它说明树本身存在计算收益。某题严格组与单token AR存在有限精度分歧，不能将这一结果扩大成全量逐字等同AR。
- 组合在GSM8K／MBPP的正确数有净增加，也有具体题目的得失；这不是所有题都保持相同答案的保证。

<!-- CURRENT_EXPERIMENT_TABLE_START -->
<a id="current-results"></a>

## 4.1 当前方案完整任务表（2026-10-31）

统一列出当前`BufferedKV-ASD`的15类任务。A为官方外部树比较的既有66匹配开发输入，B为新增场景182请求；每行只与同批次、同题、同机器基线配对，不把A/B拼成新的独立实验。两批均为Qwen3-8B/DFlash、单A6000、FP32/eager/RMSNorm、greedy，成本表在各批合成校准；输出上限和评分按各自协议固定。

耗时变化是逐题`当前/基线`比值的算术平均减1，负数更快。质量三列顺序为DFlash／ASD／当前：问答为F1×100、摘要为ROUGE-L×100，其余为正确数/可评分数；未评不表示质量相同。输出一致性单列，A参考本窗口AR，B参考DFlash，不把B的身份写成新验证的AR身份。

|批次·任务|生成请求数|当前耗时vs DFlash|当前耗时vs ASD|质量：DFlash / ASD / 当前|当前输出一致|当前截断|
|---|---:|---:|---:|---|---|---:|
|A·GSM8K|8|-10.90%|+3.83%|8/8 / 8/8 / 8/8|AR 7/8|0|
|A·MATH-500|12|-7.71%|-8.89%|8/12 / 8/12 / 9/12|AR 9/12|0|
|A·HumanEval|8|-4.35%|-4.25%|7/8 / 7/8 / 7/8|AR 8/8|0|
|A·MBPP|8|-16.50%|-3.93%|4/8 / 5/8 / 4/8|AR 8/8|0|
|A·MMLU|14|-2.95%|-2.28%|5/14 / 1/14 / 5/14|AR 13/14|0|
|A·MT-Bench|8|-14.11%|-13.20%|未评 / 未评 / 未评|AR 4/8|0|
|A·Alpaca|8|-16.65%|-3.55%|未评 / 未评 / 未评|AR 7/8|0|
|B·IFEval|64|-15.33%|-13.86%|52/64 / 53/64 / 52/64|DFlash 55/64|3|
|B·LiveCodeBench v6|24|-7.34%|-13.63%|8/24 / 9/24 / 8/24|DFlash 23/24|0|
|B·CanItEdit|32|-3.22%|+0.60%|17/31 / 15/31 / 17/31|DFlash 31/32|0|
|B·Qasper|8|-1.59%|-0.86%|43.14 / 45.38 / 43.14|DFlash 8/8|0|
|B·MultiFieldQA-en|8|-2.46%|-0.47%|55.60 / 56.00 / 55.60|DFlash 8/8|0|
|B·MultiNews|8|-20.43%|-13.30%|26.46 / 26.41 / 26.46|DFlash 7/8|4|
|B·GovReport|8|-19.48%|-16.24%|34.87 / 35.48 / 34.61|DFlash 7/8|5|
|B·AIME25|30|-8.78%|+4.60%|5/30 / 7/30 / 5/30|DFlash 7/30|10|

- A的MMLU为生成字母合同，格式错误计错；MT-Bench/Alpaca本批没有新增裁判评分。它们不能借用历史其他输出的奖励分。
- B的IFEval主列为严格整请求通过数，严格指令DFlash/ASD/当前为85/86/85，分母97；宽松整请求为52/56/53，分母64。
- B的CanItEdit32题全部计入速度，规范答案31/32可运行，统一v3评分；不可评分的`78_llm_inference`对三组使用相同掩码。原始评分、v2/v3及源码清单均保留。
- B的LongBench四个子集各8题，其中19/32输入采用3840-token首尾截取；QA输出上限128、摘要512。除AIME25完整30题外，B均为固定子集，不是全量排行榜。MultiNews/GovReport当前截断4/8、5/8，不以ROUGE-L替代事实正确率。
- B的AIME当前/DFlash正确集合均5/30，ASD7/30；输出上限4096，当前/DFlash/ASD截断10/11/11。更高输出一致性不保证超越普通ASD的任务正确率。

两批混合总量分别列出；下表总耗时比与上表逐题配对均值不是同一统计，不跨批相除，也不将不同任务质量相加成总正确率。

|批次|请求/实际生成|当前总秒|当前总wall vs DFlash|当前总wall vs ASD|总token：DFlash / ASD / 当前|当前/ASD参考一致数|
|---|---:|---:|---:|---:|---|---|
|A|66/594|253.650|-12.96%|-14.53%|24391 / 26416 / 23605|AR 56/66、14/66|
|B|182/546|1972.739|-12.96%|-12.16%|131275 / 135321 / 131201|DFlash 146/182、31/182|

A同输出56请求中，当前对DFlash/BASTION配对耗时少8.66%/0.64%；B同输出146请求中，对DFlash少10.77%。这些事后子集用于说明收益包含计算效率，不替代保留全部题目的主表。

来源：A读取[analysis.json](../results/external-trees/analysis.json)的`families.<task>.<arm>.natural`与[totals.json](../results/external-trees/totals.json)；B读取[analysis.json](../results/expanded-suite/analysis.json)的`families.<task>.<arm>`与`totals`。A使用`<baseline>.ratio`，B使用`<baseline>.paired_ratio`；没有用两个均值相除推导新基线。逐题、截断、完整输出与来源SHA见[A全表](EXTERNAL_TREES_RESULTS.md)、[B全表](EXPANDED_SUITE_RESULTS.md)和[B结论](EXPANDED_SUITE_FINDINGS.md)。

<a id="baseline-results"></a>

## 4.2 九组同机基线与论文引用

九组同机基线沿用A批594次生成，不新增GPU运行。下表是各方法66请求的混合总量；不同输出路径和长度也影响总wall，不能把全部差异解释成单步执行效率。完整逐任务九方法质量/耗时见[A全表](EXTERNAL_TREES_RESULTS.md)。

|方法|总秒|实际token/s|总wall vs DFlash|总wall vs BASTION|AR输出一致/66|
|---|---:|---:|---:|---:|---:|
|AR|1257.316|19.40|+331.44%|+367.11%|66|
|DFlash|291.423|83.70|+0.00%|+8.27%|66|
|ASD|296.780|89.01|+1.84%|+10.26%|14|
|BudgetTree|263.316|92.63|-9.64%|-2.17%|66|
|BudgetTree-ASD|282.545|98.26|-3.05%|+4.97%|20|
|BufferedKV-ASD|253.650|93.06|-12.96%|-5.76%|56|
|DDTree-32|265.687|91.80|-8.83%|-1.29%|66|
|DDTree-128|445.613|54.74|+52.91%|+65.55%|66|
|BASTION|269.167|90.62|-7.64%|+0.00%|66|

新增场景的外部基线优先沿用已核验的[文献引用表](EXPANDED_LITERATURE_BASELINES.md)。以下为BASTION v4 Table 1的Qwen3-8B、T=0、A100/BF16原文数值，倍率为相对AR的per-token latency比，τ含bonus；不是本机实测，也不据此计算本轮方法的速度排名。

|任务|EAGLE-3相对AR倍率 / τ|DFlash相对AR倍率 / τ|BASTION相对AR倍率 / τ|
|---|---:|---:|---:|
|GSM8K|3.10× / 2.74|4.96× / 6.51|6.92× / 8.85|
|MATH-500|2.78× / 2.51|6.06× / 8.25|8.02× / 10.53|
|AIME25|2.92× / 2.45|5.91× / 7.20|7.69× / 9.46|
|HumanEval|3.01× / 2.69|5.25× / 6.64|7.64× / 9.44|
|MBPP|2.73× / 2.24|4.91× / 6.04|7.10× / 8.65|
|LiveCodeBench|2.29× / 1.91|5.90× / 7.64|7.64× / 9.91|
|MT-Bench|2.68× / 2.24|2.81× / 4.46|4.31× / 6.28|
|Alpaca|2.10× / 1.77|2.28× / 3.15|3.54× / 4.68|

原文：[BASTION Table 1](https://arxiv.org/html/2605.29727v4#S4.T1)。其硬件、精度、prompt、样本和统计口径与A/B不同。B没有新跑AR/EAGLE-3/DDTree/BASTION，不填造这些方法的新题准确率；DFlash/ASD仅作为同输入同机质量与速度锚点重跑。IFEval/CanItEdit及本轮截取后的四个LongBench子集没有可直接填入的匹配文献数字，保留实际三组结果。
<!-- CURRENT_EXPERIMENT_TABLE_END -->

## 5. 为什么加速：更长实际接受前缀，更少验证轮

最大扩样首个自然重复的轨迹如下。接受长度是**实际输出的Draft token数／验证轮数**，排除Target首token、纠正和bonus；裁去末轮没有真正输出的候选。

|任务|实际Draft接受长度：ASD → 组合|验证轮数：ASD → 组合|自然输出token：ASD → 组合|
|---|---:|---:|---:|
|GSM8K|5.440 → 6.648|5673 → 4750|36594 → 36338|
|MBPP|4.564 → 5.879|567 → 463|3189 → 3217|
|Alpaca|1.951 → 2.848|7489 → 5578|22106 → 21468|

共同达到128个输出token的配对中，组合相对ASD的时延变化为GSM8K -9.63%（127题）、Alpaca -18.36%（41题）。MBPP为-6.97%，但只有3题，不能代表全部代码任务。这提供了固定生成量下的计算增量证据，不能将所有收益都解释成回答更短。

树增加了每轮候选与验证节点成本。例如GSM验证节点／输出token从2.480增至3.674；它通过更少的昂贵验证轮换取端到端收益，候选并非越多越好。MATH的接受长度也提高，但逐题输出长度比均值增加11.08%，自然时延仍变慢，说明只看接受长度不足以判断实际加速。

早期开发检查也保留了清晰的分解：共同7题、各128token时，线性ASD为157轮，第一组合126轮，Joint为122轮。该结果支持树和ASD的互补机制；对应同状态／固定主干分析见[第一阶段](ASD_TREE_FINDINGS.md)、[Joint消融](ASD_JOINT_TREE_FINDINGS.md)。后续七任务没有确认Joint对第一组合的稳定新增收益，因此当前通用主线仍是BudgetTree-ASD。

## 6. 质量、负结果和适用场景

**已有收益可以保留，同时应按场景选用严格树或树 + ASD。**

- GSM8K：组合在两轮独立扩样均有自然与共同128时延收益；适合作为当前组合的代表场景。
- MBPP：最新扩样有速度与正确数净改善；旧249请求批次曾慢71.85%、正确14 → 12／32，不能用新一批结果覆盖旧失败，也不能推广为全部代码任务。
- HumanEval：最大扩样快3.26%，正确数53 → 52／64，存在功能正确性取舍。严格成本树保留DFlash结果，是另一条可用路径。
- MATH：旧32题批次快11.42%、正确26 → 28；新64题批次慢5.50%、正确44 → 45。切分和任务路径改变会影响结果，不能概括为稳定提速或稳定退化。最新批次还保留长循环／截断和真实数学错误。
- MMLU：当前组合不占优势。仅字母输出合同从39降至37／114；各方法可提取的首选项虽同为78／114，但该诊断不能替换原定输出合同评分。
- MT-Bench／Alpaca：有实际自然时延收益。最大扩样的独立奖励模型偏好相对ASD为50.00%／46.09%，分别是7胜2平7负、26胜7平31负；不能仅凭速度声称开放回答质量全面提高。任务性质检查与偏好分开保存。

开放质量来自固定独立偏好训练权重Skywork Reward V2及单列的任务性质审查。它不是官方GPT-4 MT-Bench分数、AlpacaEval胜率或通用正确率。早期未通过换序控制的裁判结果仍留档，没有冒充有效主证据。

固定256-token吞吐还要单独看：某些短回答在EOS后继续计算。例如HumanEval固定TPS增益48.42%，自然回答时延只改善3.26%；前者不能写成正常代码回答提速48.42%。

## 7. AR、论文指标和其他路线

AR基线已经补测：在旧249个匹配输入上执行747次固定长度三连计时。GSM8K严格DFlash／AR为同窗直接实测3.845×；组合／AR的4.709×是借助DFlash锚点做的跨会话校准估计，不能标成组合与AR本轮同窗实测。详见[指标补全](ASD_PAPER_METRICS_FINDINGS.md)。

评测覆盖ASD论文的七个任务名，但模型、硬件、数据子集、提示／质量协议和重复次数存在明确差异。这里是Qwen3-8B／DFlash／A6000，论文主表采用另一配对；Stanford Alpaca子集也不等同于AlpacaEval。因此本项目新方法的增量以本机同环境直接对照为准，文献数值仅作单独注明的参考，见[论文对应关系](ASD_PAPER_ALIGNMENT.md)和[文献基线](ASD_LITERATURE_REFERENCE.md)。

DSpark曾单独适配和优化。它在部分Alpaca场景保留正结果，但没有取代DFlash作为当前主线；原版本、等价构树优化、完整正负结果都已归档，见[DSpark比较](DSPARK_TREE_FINDINGS.md)和[优化复测](DSPARK_TREE_OPT_FINDINGS.md)。ASD-Run、Utility、置信度等探索见[候选统一比较](ASD_COMPARISON_FINDINGS.md)，不将其局部结果混算为本组合的增量。

## 8. 已经完成的infra交付

成本树已进入统一runtime、CLI、JSON／SSE服务、benchmark及树过程可视化。支持严格BudgetTree与显式近似BudgetTree-ASD，成本表绑定模型／运行环境哈希，异设备需要重新校准。真实部署验收完成研究实现与运行时的输出、轮次、预算、停止及取消恢复对照，详见[infra v0.4验收](INFRA_TREE_VALIDATION.md)。

最新高保真反馈树、最小增量接受与执行复用也已接入[infra v0.5](INFRA_FIDELITY_GUIDE.md)，独立冻结于`codex/infra-interview-v0.5`／`infra-v0.5-interview`；旧v0.3/v0.4保持。v0.5交付与线上网页是否切换分别记录，不把研究表格更新当作线上部署。

研究成果与面试交付分别保留：当前聊天网页已有组合推理入口；面试版 `codex/infra-interview-v0.4`／`infra-v0.4-interview` 独立冻结，说明见[面试版位置](INFRA_TREE_INTERVIEW.md)。网页交互验收和少量部署计时不替代上述多任务研究评测。

## 9. 从哪里复核

|内容|入口|
|---|---|
|算法通俗解释|[ASD_TREE_ARCHITECTURE.md](ASD_TREE_ARCHITECTURE.md)|
|当前方案七任务与九组同机基线|[结论](EXTERNAL_TREES_FINDINGS.md) · [全表](EXTERNAL_TREES_RESULTS.md) · [统计](../results/external-trees/analysis.json)|
|当前方案新增八任务、逐题评分与论文引用|[结论](EXPANDED_SUITE_FINDINGS.md) · [全表](EXPANDED_SUITE_RESULTS.md) · [统计](../results/expanded-suite/analysis.json) · [文献](EXPANDED_LITERATURE_BASELINES.md)|
|最大扩样的主结论、全表和协议|[结论](ASD_PAPER_EXTENSION_FINDINGS.md) · [全表](ASD_PAPER_EXTENSION_RESULTS.md) · [协议](ASD_PAPER_EXTENSION_PROTOCOL.md)|
|原始统计、来源和重建审计|[analysis.json](../results/asd-paper-extension/analysis.json) · [audit.json](../results/asd-paper-extension/audit.json)|
|数学／代码／格式质量复核|[quality-audit.json](../results/asd-paper-extension/quality-audit.json)|
|开放偏好与匿名任务性质|[reward-analysis.json](../results/asd-paper-extension/reward-analysis.json) · [open-contract-audit.json](../results/asd-paper-extension/open-contract-audit.json)|
|早期七任务与AR指标补全|[七任务](ASD_PAPER_FINDINGS.md) · [指标补全](ASD_PAPER_METRICS_FINDINGS.md)|
|执行源、数值分歧和中断续跑|[源码清单](../results/asd-paper-extension/ASD_PAPER_EXTENSION_CONTINUATION_SOURCE_MANIFEST.json) · [续跑协议](ASD_EXTENSION_CONTINUATION_PROTOCOL.md)|
|部署与复现命令|[INFRA_TREE_GUIDE.md](INFRA_TREE_GUIDE.md) · [PROGRESS.md](PROGRESS.md)|

既有离线复核入口（不启动新的GPU生成）：

```powershell
.venv\Scripts\python.exe -m results.asd-paper-extension.inspect_results
.venv\Scripts\python.exe -m results.asd-paper-metrics.inspect_results
.venv\Scripts\python.exe -m results.infra-tree-v04.helpers.audit --check
```

本次总览对主表直接配对比值、样本与计时计数、接受长度和相关相对链接做静态核对；没有重跑旧完整测试或把本次文档提交当作实验执行源码。
