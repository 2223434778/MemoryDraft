# MemoryDraft

> OT-R 实验在 `codex/ot-r-probe` 分支进行。此前完整 infra 已独立冻结为
> `codex/infra-interview-v0.3` / `infra-v0.3-interview`，本机另存独立仓库、ZIP 和 Git bundle。
> 见 [面试版本位置与复现](docs/INFRA_INTERVIEW.md)。OT 运行配置为 `configs/runtime-ot.json`，
> 需安装额外依赖 `python -m pip install -e ".[dflash,transport]"`；实验规则见 [OT-R 协议](docs/OT_R_PROTOCOL.md)。
> [首轮结果](docs/OT_R_RESULTS.md)：80次计时严格一致，旧JSON上OT较后缀R降低约1.83%延迟，收益集中在一个旧项目；格式不匹配复制例仍退化。默认检索保持suffix。
> [24个新请求验证](docs/OT_R_VALIDATION_RESULTS.md)：192次计时严格一致，OT总体较suffix慢1.75%，两个预定适用场景均未达继续投入门槛；停止扩大同类OT实验。
> [实际触发位置诊断](docs/OT_R_DIAGNOSIS_RESULTS.md)：24次插桩回放均正确，定位到编辑区误触发、端点漏召回及排序问题；旧正例的OT消歧机制得到解释。插桩数据不作性能证据。
> [DFlash前缀引导验证](docs/OT_GUIDED_RESULTS.md)：新16请求/160计时全部严格一致，主场景引导OT较suffix降低2.38%延迟，引导Edit降低2.55%；未显示OT排序增量，默认仍suffix。显式实验配置`configs/runtime-ot-guided.json`，检索选项`ot-guided`/`edit-guided`。
> [内部锚点拼接开发检查](docs/OT_SPLICE_RESULTS.md)：SQL局部可减少验证轮，但首轮整体慢于Guided-Edit；定位到Draft前缀拒绝触发R冷却。[五请求最小修复检查](docs/OT_SPLICE_COOLDOWN_RESULTS.md)保留局部收益，尾部加权OT与Guided-Edit整体基本持平，仍有保留错误Draft前缀的问题。全部为已观察任务，尚无独立OT增量确认；显式配置`configs/runtime-ot-splice.json`，默认保持suffix。
> [OT完整矩阵三个小改动](docs/OT_MATRIX_RESULTS.md)：135次旧调用重现，14个同状态/39候选；64机制与128计时全部严格一致。局部顺序正则未改变选择，矩阵特征会误拒有用接续，组合仍慢于TailOT；该路线已停止继续优化，旧结果保留。
> [预计收益门控与动态长度](docs/OT_BENEFIT_RESULTS.md)：保留内部拼接，不使用完整矩阵特征。正式56机制/112计时全部严格一致；OT求解候选31→7，但验证65→67轮，比固定TailOT慢1.57%，比Suffix快2.46%，尚未显示新路由收益。14状态的跨项目接受预测误差较大，长度始终选32。`configs/runtime-ot-benefit.json`是基础模型配置；实验runner按项目排除后注入校准策略，默认仍suffix。
> [接受长度校准诊断](docs/ACCEPTANCE_RESULTS.md)：48次严格生成、56个去重Draft状态。用Draft置信度和条件生存估计，原14动作MAE从10.47降至8.42（−19.57%），Brier从0.235降至0.156；关键坏前缀与两处候选排序改善，但好长尾被低估，项目区间跨零，未通过预定门槛。预测器保留为诊断，不接入默认路由，不宣称加速。
> [前缀风险辅助拼接](docs/PREFIX_RISK_RESULTS.md)：已实现显式`edit-risk-splice`，仅在原边界存活概率<0.5且有安全替代边界时改选，固定原长度上限。旧56状态校准、独立6项目/12请求检查；新Target只有11/12完整输出合同正确，按冻结协议在机制/计时前停止。当前尚无风险策略性能结论，保留实现与失败参考。
> [风险拼接完整续测](docs/PREFIX_RISK_CONTINUATION_RESULTS.md)：按用户要求单列任务质量、保留全部12请求，60机制/120计时全部严格一致；本批没有合格拼接候选，风险未触发，无可归因的性能增益。[两个旧训练项目的覆盖检查](docs/PREFIX_RISK_CONTROLS_RESULTS.md)中，payment改选边界14→7、接受7→32、验证8→7轮，但无置信度消融也取得相同行为，仍慢于该例Guided-Edit；不作为泛化或置信度独有收益。

**固定 Target 的推理优化、服务与正确性评测工具链。**

ASD 近似解码实验独立位于 `codex/asd-splice`：[组合报告](docs/ASD_SPLICE_RESULTS.md)。
旧8请求/9方法已完成360次机制及计时运行；自然EOS全部与AR一致，但ASD零例外、零减轮，Splice-ASD较Splice慢0.28%，连续后缀变体慢0.60%。
关键拼接阻断的Target logit差距为26.8–53.9，未显示低差距例外释放长历史尾部的机会。默认CLI/服务仍严格验证；ASD实现允许改变输出，不作为无损加速。

[ASD论文任务小实验](docs/ASD_BENCH_RESULTS.md)已完成：GSM8K/MBPP各8题，官方B=8配置，共320生成。
GSM8K自然时延降低18.31%、固定工作量吞吐提高10.60%，两组最终答案均8/8；MBPP自然时延降低2.35%，公开测试均5/8。
内部拼接未触发，当前正结果归于ASD；本轮不是DSpark复现。MBPP固定128token的大吞吐增幅包含约72%自然EOS后的强制续写，不等同正常代码请求加速。

[ASD实际前缀收益预算验证](docs/ASD_BUDGET_RESULTS.md)已完成：新GSM8K 16题/MATH-500 8题，共288生成。
收益选择组相对原ASD自然时延下降4.37%/9.93%，但GSM8K正确数15/16→14/16；实际后缀组在MATH下降10.22%、GSM8K基本持平。
MATH完整结束5题的自然TPS几乎持平，总体TPS收益主要来自3道上限截断题；尚无稳定质量下的完整长解答每token增益结论。近似策略继续作为独立研究选项。

MemoryDraft v0.3 将原投机解码研究原型整理为可安装的单 GPU infra：统一 AR / DFlash / R+DFlash 运行入口，带有界队列和取消机制的 HTTP/SSE 服务，配对 benchmark、CPU/GPU profiler 与回归测试。复用成熟 Transformers、DFlash 和后缀索引，不重新实现 attention 或推理引擎。

当前支持 batch=1、greedy、FP32/eager，一个模型 worker 串行执行。HTTP 可并发排队；连续批处理、分布式推理和随机采样尚未支持。研究模块 H/D/RHD 与全部实验结果继续保留。

## 三分钟运行 CPU demo

建议 Python 3.11。以下 demo 只演示 CLI 与服务链路，不加载模型，所有报告标记 `synthetic=true`。

```bash
python -m venv .venv-infra
# Linux: source .venv-infra/bin/activate
# Windows: .venv-infra\Scripts\Activate.ps1
python -m pip install -e ".[serve,client]"
python -m memorydraft infer --config configs/runtime-demo.json --request examples/request.json --method AR --output artifacts/demo-answer.json
python -m memorydraft serve --config configs/runtime-demo.json --port 8000
```

另开终端，激活同一环境：

```bash
python -m memorydraft load-test --url http://127.0.0.1:8000 --request examples/request.json --method AR --count 20 --concurrency 2 --output artifacts/demo-load
```

也提供 CPU demo Dockerfile：`docker build -t memorydraft-demo .`，然后 `docker run --rm -p 127.0.0.1:8000:8000 memorydraft-demo`。容器不含模型、历史或 GPU 推理环境。

## 真实 GPU 推理

DFlash 与旧研究后端分别绑定 Transformers 4.57.3 / 4.52.2，使用独立环境。先安装适合硬件的 PyTorch，再安装对应 profile；运行时只读取已准备的本地模型，不自动下载。

```bash
python -m pip install -e ".[dflash,serve,client,test]"
python -m memorydraft infer --config configs/runtime-dflash.json --request examples/request.json --method R+DFlash --output artifacts/answer.json
python -m memorydraft benchmark --config configs/runtime-dflash.json --requests examples/requests.jsonl --methods AR DFlash R+DFlash --repeats 3 --output artifacts/benchmark-01
python -m memorydraft profile --config configs/runtime-dflash.json --request examples/request.json --method R+DFlash --output artifacts/profile-01
python -m memorydraft serve --config configs/runtime-dflash.json --port 8000 --queue-capacity 8
```

配置内路径相对于配置文件解析。默认 Qwen3-8B + DFlash FP32 profile 使用 A6000 48GB；RTX3090 可使用较小 Target 运行 AR，或运行测试。模型准备、API schema、SSE 示例、计时口径、故障排查与边界见 [完整使用指南](docs/INFRA_GUIDE.md)。

## 工程结构

| 模块 | 职责 |
|---|---|
| `memorydraft/runtime/` | 请求/结果协议、配置与模型加载、统一生成、benchmark 与 profiler |
| `memorydraft/serving/` | 单模型 worker、有界队列、背压、deadline/断连取消、HTTP 压测 |
| `memorydraft/backends/` | 成熟推理实现适配、严格验证、KV 回滚与补齐 |
| `memorydraft/memory/` | SQLite 不可变事件、按用户与可用时间选取快照 |
| `memorydraft/retrieval/` | token 后缀检索、记录边界与独立的历史辅助检索 |
| `memorydraft/drafters/`、`routing/` | 原 H/D/RHD 研究模块与路由 |
| `tests/`、`.github/workflows/` | CPU 正确性回归、两个依赖 profile 的 CI 定义 |

服务的 `GET /healthz`、`GET /metrics`、`POST /v1/generate` 可通过 `/docs` 查看。SSE 仅发送已由 Target 确认的 token IDs，最终返回完整文本。队列满返回 429，deadline 包含排队时间，取消在解码边界生效。默认仅绑定本机，可信调用者提供 user_id；身份认证和外部部署由接入环境负责。

## 验证与结果

```bash
# 旧研究 profile：完整 CPU/小模型回归
python -m pip install -e ".[research,serve,test]"
python -m pytest -q --basetemp .test-tmp

# 独立 DFlash profile：工程与解码检查
python -m pytest -q tests/test_runtime.py tests/test_serving.py tests/test_dflash_reuse.py

# A6000 上的真实模型验收，输出目录必须尚不存在
python -m scripts.validate_infra --config configs/runtime-dflash.json --output artifacts/acceptance-01
```

benchmark 对 token、Target 输入哈希与快照哈希逐项检查，遇到首个分歧即停止；报告包含全部在线成本。profiler 结果独立标记，不能当作未插桩性能。小样本不输出 P95/P99。生成物默认保存在 Git 忽略的 `artifacts/`。

本版[工程验收报告](docs/INFRA_VALIDATION.md)：本机146项通过/3项跳过、服务器相关54项通过；真实A6000的12次配对计时与三种方法流式输出通过严格检查，6个HTTP请求完成。该小样本中R+DFlash慢于DFlash，不能据此宣称检索通用加速。

后续[同负载迁移验证](docs/RUNTIME_MIGRATION_RESULTS.md)已完成：原8个配置修订请求、五组交错、120次计时全部正确；新版D42.061s、R+D33.882s，保留约19.4%延迟收益。候选对齐后与旧循环计数相同，未发现明显迁移退化；此前两个短样例的负结果属于不同工作负载。完整回归现为156项通过/3项跳过。

原研究的 19.4% 延迟降低仅来自特定受控 JSON 工作负载；后续归因表明当前请求事实展开即可覆盖几乎全部复用机会，不能作为长期记忆独特价值或通用生产加速的证明。见 [归因报告](docs/REUSE_ATTRIBUTION_RESULTS.md)。

## 项目记录

- [infra 使用与运维](docs/INFRA_GUIDE.md)
- [进度与实际运行记录](docs/PROGRESS.md)
- [保留的研究 README 与实验入口](docs/RESEARCH_README.md)
- [原研究架构](docs/ARCHITECTURE.md)、[审阅交接](docs/REVIEW_HANDOFF.md)
- [第三方来源与 DFlash MIT notice](docs/third_party/DFLASH_LICENSE.txt)

当前未授予整个项目统一开源许可。第三方代码保留各自许可与固定来源。权重、私有对话、凭据、数据库和大型 trace 不进入 Git。
