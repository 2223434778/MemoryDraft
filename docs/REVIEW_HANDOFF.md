# Astra 审阅入口

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
