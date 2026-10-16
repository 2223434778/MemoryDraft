# 项目进度

## 2026-10-16 — 架构 v0.2

- 已完成：架构复核、六阶段计划、Codex 启动任务及工程约束。
- 未完成：底座选择、代码导入、模型运行、GPU 性能测试。
- 下一步：执行 CODEX_START.md 第一部分，选择独立 Draft 友好的现有底座，建立 AR/D 基线并尽早实现 H。
- 性能结论：无。本仓库初始内容为设计，不是已运行的推理系统。

## 2026-10-16 — 可运行原型与 GPU pilot

### 已完成的工程

- 选择并锁定 Transformers 4.52.2 / commit `237c7c356cce5f8a6befc0a5f71bc952bb5adc9a`，复用上游验证、KV 回滚、模型加载和独立 Draft 生成。SAM/RAPID 仅审查接口，未复制无明确许可代码；详见 `BASE_SELECTION.md`。
- H/D 共用 Qwen2.5-0.5B 权重，各自固定 prompt、开放 assistant 前缀、独立 KV、即时拒绝裁剪和懒追赶；Target 固定为 Qwen2.5-1.5B。实现 R 接续复制、语义 L1 + token L2、成本路由与 AR 退路。
- SQLite 不可变版本事件，按明确可用顺序恢复快照；用户隔离、未来更正/删除、持久化、最终回答写回已检查。候选库只含请求前的历史，没有把当前测试答案写入过去。
- CLI 提供 env/import/generate/replay/report；冻结历史跨方法回放，相关/空/等 token 无关历史消融，逐 token 一致性检查及配对总时间比。
- 19 项关键测试在本机通过，涵盖验证偏移、全接受/拒绝/EOS/长度、H/D/R/AR 切换和真实上游 KV 使用、检索计时公平性、异常恢复、时间与用户隔离、不可变事件、持久化、路由成本退路和配对统计。

### 环境与运行位置

- 本机 RTX3090 24GB；当前 Python 3.13.5 / torch 2.10.0+cpu，因此 GPU 运行转到服务器，不是本轮模型超出 3090 显存。
- 服务器 Linux，双 A6000 48GB，实际仅用 GPU0。项目路径 `/media/a6000/D/ZhaoBinHe/MemoryDraft`，独立 `.venv`，torch 2.5.1+cu124 / Transformers 4.52.2 / sentence-transformers 4.1.0。复用现有基础 Python，不修改其他项目环境。
- 0.5B 普通指令模型从固定 revision 下载；多语言 MiniLM 从本机官方源下载、SCP 传输，两端压缩包 SHA-256 一致：`f9eec229a244d12246e90fc4a58c1499456be3fe5110829919f5d7d3fedee3c8`。
- 本机 SSH 克隆 GitHub 超时，HTTPS 克隆成功；远端 Hugging Face 直连超时、镜像曾限流，均已绕过为官方源下载后传输，没有把网络失败写成成功实验。

### 实际命令与结果

本机检查：

```powershell
.venv\Scripts\python -m pytest -q --basetemp .test-tmp7
git diff --check
python scripts\bundle_source.py
```

GPU 回放及机制检查（服务器项目根目录）：

```bash
.venv/bin/python -m pytest -q --basetemp .test-tmp4
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONPATH=. .venv/bin/python scripts/diagnose_precision.py --config configs/a6000-pilot.json --output results/diagnostics/precision.json
.venv/bin/python -m memorydraft replay --config configs/a6000-validation.json --db pilot.db --requests data/synthetic/requests.jsonl --output results/validation/replay.jsonl --max-new-tokens 64 --block-size 8
.venv/bin/python scripts/run_a6000_suite.py
```

- `results/validation/`：初版路由，13 请求 × 6 方法 = 78 次，全部与 AR token IDs 一致。
- `results/ar-fallback/`：新成本退路，13 请求 × 2 方法 = 26 次，全部一致。
- `results/semantic/`：真正多语言 CPU embedding、最终路由，13 请求 × 6 方法 = 78 次，全部一致。
- `results/ablation-empty/`、`results/ablation-unrelated/`：各 5 个相同受控请求 × AR/D/H = 15 次，均一致；空历史 H/D 接受和提议计数相同。
- `results/demo/`：最终回答在 order=30 写入；另一个进程验证 order=30 尚不可见、order=31 可见、其他用户不可见，结果全部通过。同步写回单独计时并计入该 demo 的总延迟。
- 共 212 次配对方法运行通过，另有单请求写回检查；这些不是 212 个独立自然请求。初期失败的 BF16 运行不在通过次数内。
- 上述小型原始结果、summary、provenance 已同步到本机。最终套件包含运行源码 SHA-256 清单；初版词法运行只有模型/配置记录，不伪造缺失的 Git commit。

### 实测发现与修复

1. 延迟激活的 H 首轮 mask 未覆盖已确认 Y，会导致 RoPE 维度错误。已修复完整 `C_H+Y` mask，并通过路由切换回归。
2. BF16/SDPA 的 docs-3 在第 54 个 token 与 AR 分歧；诊断发现接近并列的量化 logits。相同请求 FP32/eager 一致，后续参考实验统一使用该配置，未放宽接受规则。
3. 普通小模型起草成本大，H 没有稳定提高接受量。基于真实成本添加 H/D/AR 探测及退路，仍保留 H 实际运行与独立消融。
4. 语义最终配置：AR 总耗时 **9.972s**、D **18.819s**、H **19.965s**、RHD **13.300s**。H 接受 320/852，D 接受 322/819。完整 RHD 实际走 R=2/H=21/D=21/AR=242 轮。当前组合没有净加速。

详见 `EXPERIMENTS.md`，表内数字来自对应结果文件。单次合成 pilot 不作统计显著性、通用性能或“越聊越快”声明。

### 尚未达到的研究完成条件

- B6/B7 成熟学习型 Draft 对照没有完成。已确认 EAGLE 官方 Qwen2-7B 配对，但它不匹配本轮 Qwen2.5-1.5B；不能拿普通 0.5B D 冒充。需要另选匹配 Target，并适配 Target 特征恢复。这个范围尚未实现，而非已经验证不兼容所有模型。
- 当前 pilot 是设置 A，使用明确标注的合成项目流；应用事实记忆设置 B、真实合法用户历史、更长输出及可靠重复仍未完成。
- 采样、并发、服务后端、树融合及训练未实现；CLI 明确拒绝非 greedy 参数。CPU 内存/索引增长成本与精简 trace 的正式测量仍待完成。
- 项目已具备可运行工程骨架与 R/H/D 机制闭环，不能据此宣布六阶段研究或性能目标全部完成。代码与结果保留于本机及授权服务器目录。

## 2026-10-16 — GitHub 审阅交付

- 用户已授权将代码与报告推送 GitHub，供 Astra 检查与分析失败原因。
- 本次整理核心代码、配置、测试、合成数据及小型实验结果；模型权重、虚拟环境、数据库和模型压缩包由 `.gitignore` 排除。
- 新增 `docs/REVIEW_HANDOFF.md`，区分实测事实、可能原因和缺失对照，特别标明 dtype/attention 同时变更的归因限制。
- 发布前检查远端 main、文件大小、敏感内容、结果一致性及 `git diff --check`。本次仅增补审阅文档，不重新运行已通过的 GPU 套件；既有测试与运行记录保留。

## 2026-10-16 — 按用户要求复核研究路线

- 阅读原始论文/官方项目：标准 SD、REST、SAM-Decoding、RAPID、DReSD、RASD、Transformers 官方 assisted/prompt lookup、EAGLE 模型表。新增 `docs/NEXT_EXPERIMENT.md`，架构顶部明确 R/H 独立检索和原生 D/同指令 D 的优先修订。
- 已完成离线日志审计：`python scripts/audit_pilot.py --input results/semantic/replay.jsonl --output results/research-audit/pilot_summary.json`。输入校验 token、历史快照及方法配对；纯 R 4/424 轮起草、接受 12/17 token、贡献 12/436 输出 token。没有重跑 GPU。
- 已执行 `.venv\Scripts\python.exe -m pytest -q tests/test_pilot_audit.py`，**3 passed**，覆盖接受率/覆盖率区分及错误 token/快照配对拒绝。
- GitHub API 检查上游 commit/许可；REST `154d0f0ae45de5d8a9cbd1c696f6ad5da10904bf` 根目录 Apache-2.0、检索子模块 MIT。源码发现 u16 token 持久化与无记录分隔问题，不能直接适配 Qwen。SAM/RAPID 未见 LICENSE，提交与前次审查一致；本轮未复制第三方代码。
- Hugging Face 公开元数据核验 Qwen3-8B/0.6B，固定 revision 和文件 SHA-256 见 `results/research-audit/model_preflight.json`。两个 tokenizer 文件分别逐字节一致、配置词表大小相同；未下载权重或运行推理，尚不能宣称新组合有加速。
- GPU 下一阶段按协议推进：官方 D/prompt lookup 基线 → 独立历史 R → 同前缀 H/D → H 混合检索 → 路由与 B6/B7。长历史数据按原始版本、许可和时间过滤导入；不得用当前测试答案构造历史。
