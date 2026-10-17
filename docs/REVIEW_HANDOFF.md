# Astra 审阅入口

## 最新完成：R/DFlash 场景开发与独立确认（2026-10-18）

先读 [REUSE_PILOT_RESULTS](REUSE_PILOT_RESULTS.md) 与对应运行前协议。新增 `backends/dflash_reuse.py` 薄适配已审计MIT循环，连续R积累有效Target特征，回切Draft补齐独立KV；保留严格greedy及原后端。源代码已推送 `f43239a`。

24次接入检查和90次开发计时全部与AR一致。配置连续修订中Global12.053s、DFlash16.284s（1.351×），Cursor14.191s；游标减少查询却增加Draft轮数，不保留其为必需增强。P1/P3的当前prompt含旧产物，不能自动当作长期记忆增益；开发同后端关闭R性能控制尚不完整。

因此按预先声明的 [REUSE_CONFIRMATION_PROTOCOL](REUSE_CONFIRMATION_PROTOCOL.md) 完成4个新项目/4种schema/两轮修订/三重复，见 [REUSE_CONFIRMATION_RESULTS](REUSE_CONFIRMATION_RESULTS.md)。96次全部严格一致；Global33.862s、官方DFlash41.992s（1.240×，[1.198,1.286]，延迟降低19.4%）、关闭R适配42.000s、同策略Context44.427s（Context/Global1.312×，[1.290,1.333]）。参数未改变，4个项目均快于DFlash，8/8当前JSON逐字段语义检查通过且无预算截断。

Global仅123轮Draft，相比关闭R的498轮大幅减少；246轮R全部来自过去回答。简单架构的场景适配取得收益，概率反馈模型和游标均未构成此结果。仍只有4个受控合成schema，不外推自然用户场景或全基线领先。开发源码57a013ca…、确认源码2518e677…分别保留，不替换实际来源。首轮零命中查询失败只在接入阶段，已修复并保留证据，未并入性能。

复算：`python -m scripts.write_reuse_report` 和 `python -m scripts.write_reuse_confirmation_report`。发布清单123份原始/压缩哈希，数据库和模型未上传。服务器完整66 passed、1 skipped；DFlash环境相关18项通过，任务语义审计另有测试。

## 优先修订：聚焦适用场景与复用架构（2026-10-17）

用户明确目标为特定场景性能优秀，允许借鉴成熟架构，无需击败所有基线。以 [研究方案第 1.1 节](MEMORY_REUSE_RESEARCH_PLAN.md#11-优先场景与发现规则) 及修订后的阶段/评价标准为准。主候选是同项目长产物的多轮局部修订；开发阶段发现并冻结适用条件，再用新项目确认。

优先审查场景定义是否可部署、对照是否公平、改动是否有可测作用。核心只需 AR、成熟 DFlash、一个最相关复用基线；完整概率反馈模型为可选增强，不要求先实现才能验证场景价值。允许针对场景的架构适配与成本取舍，继承部分标明来源。无新实现或 GPU 结果。

## 新方向方案：验证反馈驱动的历史片段复用（2026-10-17）

用户要求详细规划，见 [MEMORY_REUSE_RESEARCH_PLAN](MEMORY_REUSE_RESEARCH_PLAN.md)。这是研究方案，未实现、未跑新实验。主线候选是同一严格验证器下，对历史接续的连续接受长度及完整成本做估计，再选择 R 或成熟 DFlash。

优先审查与 SuffixDecoding 动态长度/混合策略、EfficientEdit 重匹配、ReSpec 位置反馈的重叠；本方案没有将这些已有机制宣称为首创。需要证明正确处理部分观测的接受长度估计，比同资源的匹配长度/接受率 EMA 强基线产生独立净收益。

工程先核验连续 R 后 DFlash KV 与 Target 特征的补齐。实验分正确性、开发机会诊断、独立测试三阶段；过去历史相对当前 prompt 的独立价值单独消融。H 保留，当前没有扩大训练或根据旧测试调参数。

## 最新：H 蒸馏与独立 DFlash 基线（2026-10-17）

先读 [H_TRAINING_RESULTS](H_TRAINING_RESULTS.md) 与运行前 [H_TRAINING_PROTOCOL](H_TRAINING_PROTOCOL.md)。两个交付已跑完：官方DFlash固定快照、独立环境72次严格一致，2.213× AR；合成2.840×，公开六类别各一条1.894×。这不是完整公开24请求复现，也未集成R/记忆。许可、revision、权重/代码SHA见manifest/provenance。

H使用24训练/8开发/8测试新项目，全部共享四类模板结构，非自然历史或新任务家族泛化。40条过去回答先冻结；训练只用训练项目的当前Target硬标签、相关/空/无关三条件，LoRA1,146,880参数，开发loss选择第2轮，之后才生成测试标签。原始/训练后各自D/H共用权重，原生D控制均关闭assistant阈值且候选长度恒定。

417个完整前缀中，未训练H相对同指令D为+0.506 [0.068,1.021]；训练后H为−0.009 [-0.084,0.056]。测试168次全正确：训练D比原始D快1.353×；训练D/H耗时比仅1.009×、同路由off/on0.991×，两者区间跨1。H在弱Draft上的增益可被更强的空历史Draft追平；不能只宣称历史无效，也不能把蒸馏收益全算作H。

开发门控为空：清单有稀疏质量增量但路由更慢，数值筛选路由更快但稀疏质量差为零。测试仍保留实际H和H-on，未用关闭H宣称研究成功。DFlash和H测试集合不同，禁止直接比较两表加速比。初期资产任务在计时前停止，冻结历史复用与正式源码来源均有记录，不混入无效计时。

复算：`python scripts/write_training_report.py`，核对Target标签、配对完整性、项目隔离、实际历史快照及summary。下一步是否研究局部H或R+DFlash融合需新的协议；本轮没有根据测试追加训练或筛选样例。

## 最新：Target 历史下的在线 H 开关（2026-10-17）

先读 [H_TEACHER_ROUTER_RESULTS](H_TEACHER_ROUTER_RESULTS.md)，协议 [H_TEACHER_ROUTER_PROTOCOL](H_TEACHER_ROUTER_PROTOCOL.md)。这轮补齐上一轮尚未测量的联合条件：R/H 共用已冻结 Target 过去回答，同一完整 R 索引和路由，仅改变 H 是否可用；没有重新生成历史或调节参数。

72 次计时与 8 次 smoke 全部与原 AR 相同。AR=81.109s，H-off=49.356s（1.643× AR），H-on=53.934s，缓存 on=52.043s，启用 H 慢 9.27%/5.44%；off/on 95% 请求区间 [0.888,0.932]，off/缓存 on [0.922,0.964]，两个时间比均小于 1。验证轮数 807→816，小模型起草 77→96，六项目启用 H 均更慢。

实际 H 60 轮接受 96/120；对齐旧同前缀离线评分的 D_native 为 90/120，单条指定 Target 历史 H_teacher 为 93/120。不要将 H 自身接受比例或局部六个 token 差值当作全局收益，也不要把跨批耗时差直接归因于历史来源。当前结果支持“额外 H 质量与 R/D 的重叠、路由机会和全部启用成本不足以形成净收益”；不证明所有 H 策略或真实长期分布无效。

运行前实现已推送 `b7e1e53`/`0419b0b`，实际 GPU 源码 SHA `436c2daaa920df93a6c0b745da0ede3d6a2b39c3d273f3bef2df32b4a88c640a`。报告可用 `python scripts/write_h_teacher_router_report.py` 从 gzip 记录离线重建；测试与发布清单见最新 PROGRESS。

## 最新两个 H 归因实验（2026-10-17）

先读 [H_CAUSAL_RESULTS](H_CAUSAL_RESULTS.md) 和预先固定的 [H_CAUSAL_PROTOCOL](H_CAUSAL_PROTOCOL.md)。旧历史只来自原始 order=1 片段：生成模拟 order=0 过去请求与 Target 的 order=1 回答，冻结并保存哈希后才读当前 order=10 请求。

来源对照：H_manual 平均连续接受长度 1.663、H_teacher 1.753，直接配对增量 +0.090，六请求 bootstrap [+0.008,+0.238]，主要来自一个文档修改项目。首位命中 86.09%→91.17%。这支持本组数据的来源对齐假设，不是自然用户分布上的结论；长度/细节也可能变化。CPU 审计发现 51 个来源改善位置中，31 个也可由同来源静态 R 取得，不能把所有提升都算成 H 的独有价值。

真正相同路由的 H 开关：RHD_off=54.550s，on=60.086s，缓存 on=58.082s，108 次计时全部与固定 Target 一致。off/on 保持相同 R、D/AR、EMA、驻留及冷却逻辑；启用 H 增加其检索/探测/起草成本。off/on 只相差两次验证回合，却多 44 次小模型起草回合。旧 RD=57.317s 是辅助基线，不能代表纯 H 开关。

两项实验沿用原六个合成项目，属于追加归因；路由实验仍用原人工历史，没有把第一项 Target 历史混入第二项。优先审查候选质量提升的归因、与 R 的重叠及 H 初始化/起草成本，而非只看 H 路线本身 150/168 的接受比例。代码已先推送 `8c230ec`；完整结果与报告随本次结果提交发布。本机/服务器最终完整回归均为 40 项通过。

## 本次交付

**2026-10-17 更新：先读 [第二轮 GPU 报告](EXPERIMENTS_V2.md)。** 旧版以下问题与数据保留作第一轮证据，不代表当前实现仍共享 R/H 候选池或只有 19 项测试。当前完整回归在本机/A6000 均为 35 项通过。

第二轮完成原生 D/PLD 对照、独立 R 后缀数组、H dense/hybrid/人工合法历史及等长无关控制，432 次计时方法运行全部 token 一致。合成项目 D_native=1.168×、R=1.487×；公开任务 D_native=1.000×、PLD=1.117×、R=1.056×。D 的目标尚未在全部任务达成。

H 的预定稀疏前缀增量为 +0.375，但为解释在线接受率反向，追加同请求 532 个全前缀检查后仅 +0.045、请求 bootstrap 区间跨零。人工预选片段未明显优于自动检索，R 弱匹配位置的增量接近零；优先审查历史/S/Target 的接续对齐与任务范围，不先假定换检索器就会解决。H 在线接受 966/1245，低于同指令 D 的 972/1239；RHD 慢于 RD。追加缓存消融 144 次全部正确：H 快 1.027×、RHD 快 1.035×，但缓存后仍分别慢于 D/RD 约 2.23%/1.70%。该工程消融沿用原测试项目，不能包装为新保留集验证。

已排查两类不同故障：Qwen3 默认采样参数回填是实现问题，已修复；显式 greedy 后 BF16 在相同已确认前缀的 AR/块验证出现并列 logits 翻转，定点复现证据在 `results/qwen3-pilot/bf16-diagnostic.json`。有效速度表仅用 FP32/eager，不能合并旧错误运行。

复核时运行 `python scripts/write_research_report.py` 即可从 gzip 配对记录重建报告；脚本核对 token、Target prompt hash、snapshot 和配对完整性。来源清单在各阶段 provenance，发布压缩前后哈希在 `results/research-v2-publication.json`。完整 GPU 重跑还需要 README 中的固定模型与服务器路径。

本轮仍不是六阶段研究的最终完成版。成熟 EAGLE B6/B7、自然长期用户历史、采样及并发仍未完成。先读报告，再对照 [底座选择](BASE_SELECTION.md)、[H 与缓存实现](HISTORY_DRAFTER.md)、[进度](PROGRESS.md) 和架构 v0.2。

代码主入口：`memorydraft/backends/transformers.py`、`memorydraft/drafters/candidates.py`、`memorydraft/routing/policy.py`；检索和时间快照位于 `retrieval/`、`memory/`，回放位于 `evaluation/replay.py`。

## 第一轮保留证据

- 本机与 A6000 的既有回归均为 19 项通过。GPU 结果共 212 次配对方法运行，均通过相同 Target 输入下的 token 一致性检查；不是 212 个独立自然请求。
- `results/semantic/` 是最终语义检索配置：AR 9.972s、D 18.819s、H 19.965s、RHD 13.300s。D 接受 322/819 个候选 token，H 接受 320/852；未取得 H 净收益。
- `results/ablation-empty/` 与 `results/ablation-unrelated/` 是同一组 5 请求的消融；相关历史对照取 `results/semantic/` 中相同请求 ID。空历史 H/D 的接受与提议计数相同。
- `results/validation/` 为初版词法诊断，`results/ar-fallback/` 为增加实测 AR 成本退路后的词法运行。不能把跨运行时间差当成同一次配对统计。
- `results/diagnostics/precision.json` 保存 BF16/SDPA 的 docs-3 第 54 个 token 分歧及 FP32/eager 的一致结果。`results/demo/persistence.json` 保存跨进程写回与隔离检查。
- 结果均来自明确标注的合成数据；权重、私有历史和环境凭据不在提交中。逐轮记录是这组小型机制实验的可复核数据。

## 优先检查的问题

1. **D 自身为何慢于 AR？** 普通 0.5B Draft 相对 1.5B Target 的串行成本、Python generate 调度及 FP32/eager 配置是待检验原因。当前 host 分阶段时间受 CUDA 异步影响，尚不足以精确归因 GPU 内核瓶颈。先核对缓存实际复用与验证轮数，再决定是否更换匹配的大 Target/成熟 Draft。
2. **H 为何未提高接受量？** 检查历史与固定 Target 接续的适配程度、历史前缀是否干扰 S、候选块长及短输出覆盖。现有结果不支持“相关历史必然提高接受率”，也不足以否定长期用户记忆在其他分布上的价值。
3. **数值分歧能否充分解释？** FP32/eager 同时改变了 dtype 与 attention；恢复一致是证据，但不能仅凭这一对照断定 BF16 是唯一原因。请审查缓存/位置偏移，必要时仅对失败请求补充单变量检查，不放宽接受规则。
4. **路由是否只是减少损失？** 最终语义 RHD 实际走 R=2、H=21、D=21、AR=242 轮；仍慢于 AR。AR 退路的收益不能被写成 H 的收益，路由探测成本也必须计入。
5. **是否偏离主研究设置？** 当前是设置 A、小模型组合、合成短请求和单次 pilot。架构要求的设置 B、真实合法历史、成熟 B6/B7 对照、可靠重复和更长输出尚未完成；不能用现有结果宣称整个研究目标成功或失败。

建议审阅输出分为：确定的实现问题、已有证据支持的失败原因、仍需实验的假设，以及最多两个优先后续实验。保留所有负结果，不通过修改 Target prompt、放宽匹配或永久关闭 H 制造成功。

## 复现

先按 README 安装/核验固定模型、依赖及路径，然后在已准备好的 A6000 项目目录运行：

```bash
.venv/bin/python -m pytest -q --basetemp .test-tmp-review
OMP_NUM_THREADS=4 .venv/bin/python scripts/run_a6000_suite.py
```

若只审阅报告，可直接核对各目录的 summary、配对 JSONL、provenance 与 `docs/EXPERIMENTS.md`，无需重新下载模型或重复完整 GPU 套件。
