# MemoryDraft v0.3 使用与运维

## 范围与能力

固定 Target 的单 GPU、batch=1、greedy 运行时。一个模型 worker 串行执行请求，HTTP 层并发接收并进入有界等待队列。支持 AR、关闭 R 的 DFlash 适配器，以及 R+DFlash；原 H/D/RHD 研究接口仍保留。没有连续批处理、分布式推理或采样支持。

架构：`HTTP/CLI → 请求校验 → 有界队列 → 单 worker → 用户/时间快照 → tokenizer/候选索引 → Target 验证 → 已确认 token → SSE/最终结果`。

## 环境配置

建议 Python 3.11，PyTorch 2.5.1（已测）或 2.6；先按硬件安装 PyTorch。

| 安装组合 | Transformers | 用途 |
|---|---|---|
| `.[serve,client]` | 不需要 | CPU transport demo，无模型性能含义 |
| `.[research,serve,test]` | 4.52.2 | 原 H/D/RHD 回放；新运行时的 AR |
| `.[dflash,serve,client,test]` | 4.57.3 | 新运行时的 AR/DFlash/R+DFlash |

`research` 与 `dflash` 必须使用两个虚拟环境，不要同时安装。项目基础包使用标准库；安装推理 profile 后才加载 PyTorch。`semantic`/`training` 是旧研究附加项，不是服务前置条件。

```bash
python -m venv .venv-infra
# Linux: source .venv-infra/bin/activate
# Windows: .venv-infra\Scripts\Activate.ps1
python -m pip install -e ".[dflash,serve,client,test]"
python -m pytest -q tests/test_runtime.py tests/test_serving.py tests/test_dflash_reuse.py
```

CPU CI 覆盖两个 profile；DFlash profile 不运行绑定旧 Transformers 私有 API 的研究测试。GPU 验收由 `scripts/validate_infra.py` 手动执行。CI 配置提交到 GitHub 并不代表工作流已经成功运行，结果以 Actions 页面为准。

## 准备模型与内存

运行时只读取本地模型，所有 `from_pretrained` 使用 `local_files_only=True`。默认配置引用仓库 `models/qwen3-8b` 与 `models/dflash-qwen3-8b`。准备方法及固定 revision 见 `configs/qwen3-download-manifest.json`、`configs/dflash-manifest.json` 与研究文档；运行时不会隐式下载。

加载 DFlash 前校验发布代码、配置、权重 SHA-256，再允许执行其本地自定义代码。审计清单和 MIT notice 随 wheel 打包；Target 的配置/revision 会记录，但本运行时不会重新对 Target 每个权重分片做完整哈希校验，部署方应使用已有校验后的模型目录。

当前可验证 profile 是 FP32/eager。8B Target + DFlash 需要 48GB 级显卡；RTX3090 可运行较小 Target 的 AR 及 CPU 测试。不要把 BF16 的历史不一致配置改名为已验证生产配置。

配置路径相对于 JSON 所在目录解析。可复制 `configs/runtime-dflash.json` 后修改设备、目录和 token 上限。`db` 可选，指向通过原 `memorydraft import` 准备的 SQLite 库。每次执行时固定 `order` 之前的合法用户快照；新接口不自动写入生成结果，因此取消、失败及 benchmark 输出都不会进入历史。

```bash
python -m memorydraft infer --config configs/runtime-dflash.json --request examples/request.json --method DFlash --output artifacts/answer.json
python -m memorydraft benchmark --config configs/runtime-dflash.json --requests examples/requests.jsonl --methods AR DFlash R+DFlash --repeats 3 --output artifacts/benchmark-01
python -m memorydraft profile --config configs/runtime-dflash.json --request examples/request.json --method R+DFlash --output artifacts/profile-01
```

输出文件/目录拒绝覆盖。输出目录 `artifacts/` 默认忽略，其中可能包含 token、文本、数据库及大 trace；分享前检查内容。profiler 的 Chrome trace 可在本地 Perfetto 打开，默认不上传。

## 服务与 API

```bash
python -m memorydraft serve --config configs/runtime-dflash.json --host 127.0.0.1 --port 8000 --queue-capacity 8 --event-capacity 32
```

服务默认绑定 loopback，无访问日志，无遥测。它面向可信本地调用者；`user_id` 是快照选择字段，不是身份认证。需要外部访问时，由部署环境提供认证、请求体大小限制及 TLS。单 GPU 只启动一个 worker，不能用多个 Uvicorn worker 复制权重。

`GET /healthz` 返回 readiness、可用方法及是否 demo；启动完成后才接收请求。`GET /metrics` 返回请求完成/失败/取消/超时/拒绝数、输出 token 数、正在执行数、排队数及容量，不含 prompt、用户 ID 或候选内容。`/docs` 提供 OpenAPI 页面。

`POST /v1/generate` 请求：

```json
{
  "id": "request-01",
  "user_id": "example",
  "order": 10,
  "messages": [{"role": "user", "content": "Explain speculative decoding briefly."}],
  "method": "R+DFlash",
  "max_new_tokens": 64,
  "stream": true,
  "timeout_seconds": 60
}
```

未知字段、非 greedy 参数和不支持的方法拒绝为 422。非流式返回 token IDs、最终文本、prompt/snapshot 哈希、结束原因、GPU 峰值、引擎延迟、排队时间及服务处理时间。配置的输入 token/输出 token 上限在 tokenization 后、模型 forward 前校验。

流式响应使用 SSE：

```text
event: start
data: {"request_id":"request-01","method":"R+DFlash"}

event: tokens
data: {"token_ids":[123,456]}

event: complete
data: {"token_ids":[123,456],"text":"...","finish_reason":"eos",...}
```

`tokens` 只包含已确认前缀，包含触发停止的 EOS，不包含被拒绝后缀。客户端可使用相同 tokenizer 累积解码；最终 `text` 是完整解码结果。逐 token 单独 decode 不保证 Unicode 完整，因此接口没有伪造文本 delta。

队列满在发送流式 headers 之前返回 429 和 `Retry-After`。deadline 从入队开始，包括排队、预处理、推理和慢客户端背压。非流式超时返回 504；流开始后以 `event: error` 携带错误码，HTTP 200 本身不代表生成成功。断连会取消排队请求或发出执行中取消信号。

取消是协作式：AR 在 token 边界、DFlash 在验证轮边界检查。正在执行的 CUDA kernel/prefill 不会被强制打断；worker 在清理和同步完成之前不会处理下一请求。等待队列可立即移除已取消任务；慢客户端的 token buffer 有容量上限。服务关闭时取消待执行/当前请求，等待 worker 停止，避免在前一请求仍执行时释放模型。

## 性能与正确性

benchmark 先用独立短文本预热；每个请求建立 AR 参考，再随机交错各方法计时。对 token、Target prompt 哈希和历史快照哈希全部检查；首个错误立即保存 failure 并停止，不能发布成功汇总。

`latency_seconds` 包括读取快照、tokenization、索引构建、prefill、起草、验证、KV 维护、结果解码及 GPU 同步。加载与预热单独排除；第一 token 时间从上述在线处理开始。服务另报排队与服务处理时间，客户端压测才包含完整 HTTP 传输。TPOT 是请求内 `(总延迟 - TTFT)/(输出 token 数 - 1)`，包含验证块节奏，不是逐 token 间隔分布。

P50 在有样本时报告；P95 至少 20 次观测，P99 至少 100 次才输出。这只是避免极小样本误用，不代表统计稳定；重复执行不能当成更多独立请求。当前 DFlash 方法使用关闭 R 的同一个适配循环，原生上游控制保留在研究脚本，不能把名称当成完整上游 serving 复现。

可在 benchmark 后加 `--baseline artifacts/benchmark-01 --max-slowdown 0.10` 设置工程退化门槛。先核对环境、固定 Target revision、请求、配对、输出和快照；不兼容时拒绝比较。超过门槛写入 `regression.json` 并以非零状态退出。阈值是调用者选择的工程预算，不是显著性检验，不应对 demo 或跨硬件结果使用。

```bash
python -m memorydraft load-test --url http://127.0.0.1:8000 --request examples/request.json --method R+DFlash --count 20 --concurrency 2 --output artifacts/http-load-01
```

压测是固定并发的 closed-loop 客户端，报告成功/失败数、成功请求延迟、客户端 TTFT、墙钟吞吐，并检查流 token 与 complete 结果一致。它不自动证明应用答案正确，也不测开放式到达率；小样本验收不能当成容量规划。

`profile` 单独导出 CPU/CUDA trace 与算子统计，标记 `instrumented=true`。阶段名覆盖 snapshot、tokenize、index_build、target_prefill、retrieval_lookup、draft_forward、target_verify。host 阶段与 CUDA 执行重叠，不能直接相加当成 GPU 时间。

## 故障排查

- Transformers 版本不符：切换 profile 对应虚拟环境，不放宽检查。
- OOM：降低输入/输出预算，或为 AR 使用较小模型；不自动改变 dtype 或 Target。
- checksum 失败：重新检查固定版本资产；不跳过校验运行自定义代码。
- 正确性不符：保留 failure、源码清单与配置，停止比较性能。
- 429：检查 `/metrics`，降低客户端并发或按实际等待预算调整队列。
- 超时后仍占 GPU：允许当前验证轮结束；关注 worker 恢复，不能直接并行启动第二次生成。
- Windows HTTP 测试卡在 asyncio 初始化：检查运行环境是否允许本机 loopback socket；该问题可能发生在应用启动之前。

## 后续扩展边界

v0.3 交付安装 profile、统一接口、服务生命周期、回归评测和 profiler。接下来应根据实际 trace 选择一个瓶颈，补真实负载并优化；连续批处理、BF16、采样和成熟 serving 引擎移植分别需要独立设计与正确性验证。原研究结论和负结果保留在 `RESEARCH_README.md`，不作为生产收益承诺。
