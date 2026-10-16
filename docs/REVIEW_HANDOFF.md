# Astra 审阅入口

## 本次交付

这是首个可运行的 R/H/D 机制原型及负结果报告，不是六阶段研究的最终完成版。先读 [实测报告](EXPERIMENTS.md)、[底座选择](BASE_SELECTION.md)、[H 与缓存实现](HISTORY_DRAFTER.md)，再对照 [进度](PROGRESS.md) 和架构 v0.2。

用户后续要求已形成 [下一阶段实验协议与上游审计](NEXT_EXPERIMENT.md)：分离 R/H 检索、补充原生 D，并用同前缀对照验证 H。本轮还核实了 R 覆盖率及 REST/Qwen 适配问题；没有新增 GPU 性能结果。

代码主入口：`memorydraft/backends/transformers.py`、`memorydraft/drafters/candidates.py`、`memorydraft/routing/policy.py`；检索和时间快照位于 `retrieval/`、`memory/`，回放位于 `evaluation/replay.py`。

## 可核对的证据

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
