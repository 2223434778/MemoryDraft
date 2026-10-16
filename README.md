# MemoryDraft

**长期用户记忆增强的投机解码研究项目。**

请求级语义检索构建小型历史候选池，解码级前缀匹配寻找可复制片段；每轮在 R（直接复用）、H（历史辅助小模型起草）和 D（普通 Draft）之间选择，交由固定 Target 严格验证。

## 当前状态

已实现基于 Transformers 4.52.2 的单请求 greedy 研究原型：独立 KV 的 H/D、R 接续复制、用户/时间快照、词法及可选语义检索、带探测/冷却/AR 退路的路由，以及冻结历史回放。
已在 A6000 上用真实 Qwen2.5-1.5B Target + 0.5B Draft 跑通初始 13 请求、6 方法的一致性检查。首轮 H 未取得净收益；这不是已完成的论文性能结论或生产服务。成熟学习型 Draft 的 B6/B7、随机采样和并发尚未实现。

## 阅读与执行顺序

1. [Codex 启动任务](docs/CODEX_START.md)
2. [架构与六阶段实施计划 v0.2](docs/ARCHITECTURE.md)
3. [架构复核结果](docs/ARCHITECTURE_REVIEW.md)
4. [进度记录](docs/PROGRESS.md)

开发执行模型按用户选择使用 GPT-6.1-sol。底座选择见 [BASE_SELECTION](docs/BASE_SELECTION.md)，H/KV/计时规则见 [HISTORY_DRAFTER](docs/HISTORY_DRAFTER.md)，实测表格见 [EXPERIMENTS](docs/EXPERIMENTS.md)，阶段状态见 [PROGRESS](docs/PROGRESS.md)。

Astra 审阅请从 [REVIEW_HANDOFF](docs/REVIEW_HANDOFF.md) 开始，里面列出失败证据、代码入口和仍需验证的原因。

## 安装与检查

建议 Python 3.11。先按硬件安装 PyTorch，再在独立环境中安装项目；不要覆盖其他项目环境。

```bash
python -m venv .venv
# Linux: source .venv/bin/activate；Windows: .venv\Scripts\Activate.ps1
python -m pip install -e ".[test,semantic]"
python -m pytest -q --basetemp .test-tmp
python -m memorydraft env --output results/environment.json
```

## 可运行的合成回放

以下命令在项目根目录运行。`configs/local.example.json` 固定官方模型 revision，可复制修改设备/本地路径。RTX3090 足以容纳本轮模型；本机现有 PyTorch 是 CPU 版，实际 GPU 运行使用服务器。FP32/eager 是当前正确性参考；BF16/SDPA 已发现数值分歧，不视为已验证配置。

```bash
python scripts/make_pilot.py
python -m memorydraft import --db pilot.db --input data/synthetic/history.jsonl
python -m memorydraft replay --config configs/local.example.json --db pilot.db --requests data/synthetic/requests.jsonl --output results/pilot/replay.jsonl --max-new-tokens 64 --block-size 8
python -m memorydraft report --input results/pilot/replay.jsonl --output results/pilot/summary.json
```

服务器已有模型和环境时，使用 `.venv/bin/python` 和 `configs/a6000-validation.json`；语义模式使用 `configs/a6000-semantic.json`。语义模型可通过 `python scripts/download_models.py --embedding` 下载到项目内。下载端点可由调用者设置 `HF_ENDPOINT`，项目不会自动发送私有历史到网络。

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
