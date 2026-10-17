# MemoryDraft

**长期用户记忆增强的投机解码研究项目。**

围绕 R（直接复用）、H（历史辅助小模型起草）和 D（普通 Draft），交由固定 Target 严格验证。R 使用合法快照上的独立 token 接续索引；H 使用请求级 dense/BM25 检索及独立 Draft 上下文。

## 当前状态

已实现基于 Transformers 4.52.2 的单请求 greedy 研究原型：独立 KV 的 H/D、R 接续复制、用户/时间快照、词法及可选语义检索、带探测/冷却/AR 退路的路由，以及冻结历史回放。
已在 A6000 上用真实 Qwen2.5-1.5B Target + 0.5B Draft 跑通初始 13 请求、6 方法的一致性检查。首轮 H 未取得净收益；这不是已完成的论文性能结论或生产服务。已补独立成熟 DFlash 基线；R+DFlash 薄适配已实现并通过首轮接入检查，正在做合成开发场景验证。随机采样和并发尚未实现。

第二轮 [GPU 报告](docs/EXPERIMENTS_V2.md) 已完成 432 次配对计时运行：A6000、Qwen3-8B/0.6B、FP32/eager、非思考模式。合成历史任务 D_native **1.168×**、R **1.487×**；公开任务 D_native **1.000×**、R **1.056×**、PLD **1.117×**。D 尚未在全部任务上加速。H 的全前缀接受长度增量约 **+0.045 token**、区间跨零，且在线慢于同指令 D；没有宣称 H 成功。Qwen3 默认采样回填与 PLD 长度边界已修复并有回归测试。

追加的 [H 来源与对称路由实验](docs/H_CAUSAL_RESULTS.md) 已完成：同一 Target 在模拟过去任务上生成的历史，使 H 平均连续接受长度从 1.663 提高到 1.753；但原人工历史上，对称路由的 H-on 比 H-off 慢约 10.15%，缓存后仍慢约 6.47%。来源质量实验与路由成本实验分开控制变量，不能拼成已测的 Target-history RHD 加速。新结果含 108 次计时与 6 次 smoke，源码入口 `scripts/run_h_causal.py`，协议见 `docs/H_CAUSAL_PROTOCOL.md`。

进一步的 [同一 Target 历史在线验证](docs/H_TEACHER_ROUTER_RESULTS.md) 已完成：R/H 共用上一轮冻结历史，72 次计时和 8 次 smoke 全部正确。H-off=49.356s（相对 AR 1.643×），H-on=53.934s，缓存 on=52.043s；启用 H 分别慢 9.27%/5.44%，验证轮数增加 9、小模型起草增加 19。改善历史来源仍未使当前固定 H 策略获得净收益。入口 `scripts/run_h_teacher_router.py`，离线重建报告使用 `python scripts/write_h_teacher_router_report.py`。

最新 [H 蒸馏与 DFlash 报告](docs/H_TRAINING_RESULTS.md)：独立 DFlash 在旧六合成+六公开请求上完成72次严格一致计时，FP32/eager为 **2.213× AR**。新项目按24训练/8开发/8测试分离，LoRA训练后D从62.933s降至46.522s；H=46.099s，但同权重同前缀H增量−0.009、区间跨零，RHD开启H仍无稳定净收益。未训练H在新数据、短历史和k4组合上有正信号，不能概括历史永远无效。H最终168次计时全部正确，两项实验集合不同，不直接比较其加速比。离线复算：`python scripts/write_training_report.py`。

新方向按[场景优先方案](docs/MEMORY_REUSE_RESEARCH_PLAN.md)推进，首轮[运行协议](docs/REUSE_PILOT_PROTOCOL.md)固定R接续、DFlash回退及局部重匹配。入口 `scripts/run_reuse_pilot.py`，复算 `scripts/write_reuse_report.py`。新循环许可和来源见 `docs/third_party/DFLASH_LICENSE.txt` 及文件头。

## 阅读与执行顺序

1. [Codex 启动任务](docs/CODEX_START.md)
2. [架构与六阶段实施计划 v0.2](docs/ARCHITECTURE.md)
3. [架构复核结果](docs/ARCHITECTURE_REVIEW.md)
4. [进度记录](docs/PROGRESS.md)

开发执行模型按用户选择使用 GPT-6.1-sol。底座选择见 [BASE_SELECTION](docs/BASE_SELECTION.md)，H/KV/计时规则见 [HISTORY_DRAFTER](docs/HISTORY_DRAFTER.md)，实测表格见 [EXPERIMENTS](docs/EXPERIMENTS.md)，阶段状态见 [PROGRESS](docs/PROGRESS.md)。

Astra 审阅请从 [REVIEW_HANDOFF](docs/REVIEW_HANDOFF.md) 开始，里面列出失败证据、代码入口和仍需验证的原因。

实验设计和上游核查见 [协议与审计](docs/NEXT_EXPERIMENT.md)；实际执行结果以第二轮报告为准，协议中的“尚未运行”描述保留为运行前记录。

## 安装与检查

建议 Python 3.11。先按硬件安装 PyTorch，再在独立环境中安装项目；不要覆盖其他项目环境。

```bash
python -m venv .venv
# Linux: source .venv/bin/activate；Windows: .venv\Scripts\Activate.ps1
python -m pip install -e ".[test,semantic,research]"
python -m pytest -q --basetemp .test-tmp
python -m memorydraft env --output results/environment.json
```

## 可运行的合成回放

以下旧模型 demo 命令在项目根目录运行。`configs/local.example.json` 固定官方模型 revision，可复制修改设备/本地路径。项目本机 venv 已有 CUDA PyTorch；第二轮 FP32 8B 配置峰值约 33GiB，使用 48GB A6000。FP32/eager 是当前正确性参考；BF16 已发现并复现数值分歧，不视为已验证配置。

```bash
python scripts/make_pilot.py
python -m memorydraft import --db pilot.db --input data/synthetic/history.jsonl
python -m memorydraft replay --config configs/local.example.json --db pilot.db --requests data/synthetic/requests.jsonl --output results/pilot/replay.jsonl --max-new-tokens 64 --block-size 8
python -m memorydraft report --input results/pilot/replay.jsonl --output results/pilot/summary.json
```

服务器已有模型和环境时，使用 `.venv/bin/python` 和 `configs/a6000-validation.json`；语义模式使用 `configs/a6000-semantic.json`。语义模型可通过 `python scripts/download_models.py --embedding` 下载到项目内。下载端点可由调用者设置 `HF_ENDPOINT`，项目不会自动发送私有历史到网络。

第二轮 Qwen3 元数据可用 `python scripts/download_models.py --qwen3 target --metadata-only` 和对应 `--qwen3 draft` 准备；权重按 `configs/qwen3-download-manifest.json` 由 `python scripts/download_verified_weights.py` 下载并校验。已准备资产的 A6000 项目目录中执行：

```bash
OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_qwen3_pilot.py
```

已有提交结果可在本机离线复算，无需模型/GPU：

```bash
python scripts/write_research_report.py
python -m pytest -q --basetemp .test-tmp
```

`results/qwen3-pilot/` 包含配对记录、失败现场与来源清单；JSONL 使用无损 gzip，原始/发布哈希见 `results/research-v2-publication.json`。`qwen3-inherited-defaults` 是修复 greedy 参数前的无效运行，不能并入性能统计。

追加历史向量缓存消融另有 144 次计时运行，检索时间约从 3s 降到 1s，H/RHD 总耗时分别改善约 2.6%/3.4%，仍未超过同批 D/RD。缓存与未缓存路径均保留在 `memorydraft/retrieval/pool.py`；实验脚本为 `scripts/run_embedding_cache_check.py`，明确区分首次与后续访问。

该脚本会保存失败现场、用开发请求选择候选长度，再冻结配置做测试；不会将测试输出写入历史。新方法名 `D_native` 是官方起草控制、`D_adapter` 是相同输入的适配控制、`D` 是 H 的同指令空历史控制，`PLD` 是补齐长度边界的上游 prompt lookup。`--r-scope shared_pool` 可保留旧级联检索消融，默认独立快照索引。

`--methods AR D R RD H RHD` 对应 B0–B5。回放强制加入 AR，逐 token 检查各方法结果，遇到首个差异即保存现场并停止。每个方法共享同一历史快照，不会把测试答案写入过去历史。合成重复场景与真实长期用户分布不能混同。

H 的空/无关历史消融仅用于离线回放：

```bash
python -m memorydraft replay --config configs/a6000-validation.json --db pilot.db --requests data/synthetic/h_cases.jsonl --methods AR D H --history-condition empty --output results/ablation-empty/replay.jsonl
python -m memorydraft replay --config configs/a6000-validation.json --db pilot.db --requests data/synthetic/h_cases.jsonl --methods AR D H --history-condition unrelated --output results/ablation-unrelated/replay.jsonl
```

单请求、最终回答写回与新进程恢复：

```bash
python -m memorydraft generate --config configs/a6000-validation.json --db demo.db --request data/synthetic/request.json --method RHD --max-new-tokens 32 --commit-order 30 --output results/demo/answer.json
```

`order` 是数据集明确给出的**可用顺序**，不是消息创建时间；只读取 `event.order < request.order`。相同时间戳需先映射到稳定顺序。更正以新事件追加，禁止覆盖过去版本；`MemoryStore.purge` 物理删除该用户记录的所有版本，因此会改变后续历史回放。详细 schema 见 `memorydraft/memory/store.py`。私有导入文件放 `data/private/`，不提交到 Git。

## 工程模块

| 路径 | 职责 |
|---|---|
| `memorydraft/memory/` | SQLite 不可变事件、用户隔离、时间快照、最终写回 |
| `memorydraft/retrieval/` | 请求级候选池、CPU embedding/词法诊断、token n-gram 接续 |
| `memorydraft/drafters/` | H/D 独立上下文与 KV、R 提议、即时回滚及懒追赶 |
| `memorydraft/routing/` | 单路选择、成本探测/EMA、冷却、最小驻留与 AR 退路 |
| `memorydraft/backends/` | 版本锁定、词表核验、上游严格验证适配 |
| `memorydraft/evaluation/` | 冻结回放、历史消融、配对总时间比 |
| `tests/`、`scripts/`、`configs/` | 关键回归、可复现命令和配置 |

模型加载/预热与离线安装不计入请求延迟；在线快照、检索、prefill、起草、验证及缓存维护全部计入。主机分阶段计时受 CUDA 异步影响，不可当作独立 GPU kernel 耗时相加。性能结论以同步后的端到端时间为准。

## 复用原则

优先复用现有模型加载、验证、KV 管理和检索实现。初版采用支持独立上下文的小模型，H/D 可共享权重但不能共享不同输入的 KV。SAM-Decoding、RAPID 和 EAGLE 是参考／基线候选，尚未确认互相兼容或导入其代码。第一阶段锁定版本、核验许可和接口，选择一个主底座。

## 评价目标

比较相同 Target 输入下的 AR、普通 D、纯检索、R+D、H 和 R+H+D，并最终与匹配的成熟学习型 Draft 对比。重点看包含检索、prefill 和缓存切换的端到端时间，以及 H 的独立贡献。

私有对话、模型权重、凭据和大型原始 trace 不入库。当前未授予项目统一开源许可；采用第三方代码时保留其许可及来源。
