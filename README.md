# MemoryDraft

> **项目分工：** 树流的成本树 + ASD + 保真门控 + 反馈调宽、聊天网页与部署模板已迁至独立仓库 [ShuLiu-Decode](https://github.com/2223434778/ShuLiu-Decode)。MemoryDraft 保留原 infra 与历史研究；当前研究分支已清理对应副本，详见[分仓说明](https://github.com/2223434778/MemoryDraft/blob/codex/branch-drift/docs/PROJECT_SPLIT.md)。旧提交、历史分支和 infra 冻结标签仍可追溯。

**固定 Target 的推理优化、服务与正确性评测工具链。**

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
