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

## 2026-10-16 — 第二轮实验执行记录（进行中）

- 新增原生 `D_native`、原生上下文适配 `D_adapter`、上游 prompt lookup `PLD`；旧 `D` 保留为带空 reference 的 H 同指令对照。RD/RHD 的 D 改用原生上下文，避免弱化实用基线。
- R 使用独立快照的 pydivsufsort 索引，uint32 固定宽度编码、对齐检查及记录/特殊 token 边界；动态输入复用调用上游 PromptLookupCandidateGenerator。H 的 dense/BM25-RRF 与 R 分离。新增相同前缀的 H/D 质量诊断与请求成组 bootstrap 汇总。
- 固定 Spec-Bench 提交 `fd2c1cd7d2201ef71db4c5f4e455008f017967bf`，按六类任务的既定顺序选 12 个开发、24 个测试请求，只取首轮。合成设置 B 另有 6 个开发、6 个独立测试项目，每用户 4 条预先存在的历史；详情及许可在 `data/research/`。
- 模型权重使用 `configs/qwen3-download-manifest.json` 中的公开 Qwen 官方镜像，通过完整 HF SHA-256 校验；在授权服务器目录下载。服务器中途断网，用户恢复后断点续传；下载未混入性能有效统计。
- 旧模型对照命令：`OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_research.py --config configs/a6000-validation.json --output results/research-old-bounded --stage smoke --max-new-tokens 48`。8 请求 × 6 方法均正确；AR 9.192s，D_native 13.809s（0.666×），PLD 6.395s（1.437×），R 6.111s（1.504×）。当时后台仍有下载，此结果仅作开发诊断，不作正式稳定性统计。
- 发现并修复 PLD 末尾超长（54 对 48 token）；保留原失败现场。另发现 Qwen3 新版 generation_config 会回填 do_sample=True，导致原标为 greedy 的实验实际采样。现 Target/S 均显式关闭 use_model_defaults，Target 显式 do_sample=False；`test_checkpoint_sampling_defaults_cannot_override_greedy` 禁止 multinomial。错误采样运行移至 `results/qwen3-inherited-defaults`，不计有效结果。
- 无关历史控制改为与 H_dense 等 payload token 数直接注入；人工预选历史强制使用合法选定记录，不再由语义阈值过滤。保存实际前缀长度、来源及少量候选/参考 token，测试空历史 H 与 D_format 提议相同。
- 新套件命令：`OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python -u scripts/run_qwen3_pilot.py`。在 `results/qwen3-pilot/` 保存逐阶段 provenance、SOURCE_MANIFEST、配置、首个分歧及配对记录；所有阶段先核验部署文件哈希。主运行源码清单 SHA-256 为 `6be842aa6afc83a1ed59c8e9129b5eaea384e58cd8b7feec6180e80c1bd95ba5`。
- 修复默认采样后，BF16/eager 在 spec-121 的 index=26 仍有分歧；FP32/eager 的 8 请求 × 6 方法通过。开发校准 D_native 的 k=2/4/8 加速为 1.186/1.125/1.006，冻结 k=2、FP32/eager 后进入测试。该选择不读取测试结果。
- 本机项目 venv 已安装 torch 2.6.0+cu124、NumPy 2.1.3，并确认 RTX3090 可用；系统 Python 的 CPU PyTorch 未改动。主 GPU 实测仍在 A6000 上进行，不混合两种硬件/运行时。显式 greedy 修复后本机完整回归 30 项通过，服务器针对新对照的 8 项通过；报告统计测试另有 3 项通过。最终报告将在套件完成后从结果生成。

## 2026-10-17 — 第二轮主实验完成与成本消融

- A6000 主套件和后续诊断已完成，`results/qwen3-pilot/complete.txt`、`post-checks-complete.txt` 均存在。FP32/eager、k=2、输出上限 128，所有测试配置在开发阶段冻结；原主运行源码清单仍为 `6be842aa6afc83a1ed59c8e9129b5eaea384e58cd8b7feec6180e80c1bd95ba5`，未用新源码伪造旧 provenance。
- 合成设置 B：6 项目 × 8 方法 × 3 次 = 144 次计时运行全部正确。D_native 1.168×，R 1.487×，RD 1.424×，H 1.103×，RHD 1.356×。H 相对同指令 D 多用 3.610 秒，在线接受比例 77.59% 对 78.45%；RHD 相对 RD 多用 2.848 秒，检索累计 2.934 秒。不能把相减的推算当成缓存后实测。
- Spec-Bench 固定测试切分：24 请求 × 4 方法 × 3 次 = 288 次全部正确。D_native 1.000×，R 1.056×，PLD 1.117×。D 在数学任务 1.166×、QA 0.897×；本次不能宣称 D 已普遍快于 AR。
- 主同前缀诊断每 16 token 采样，共 35 位置，H_dense 相对 D_format 平均 +0.375、请求区间跨零。为解释与在线接受率方向的差异，追加同请求全前缀 532 位置检查，增量降为 +0.045、95% 请求 bootstrap [-0.058,+0.149]；静态 R 弱匹配处约 +0.005。H_selected 与 H_dense 相近。保留两份结果，后者标为事后探索；没有重新挑选测试项目。
- `scripts/run_post_checks.py` 在主计时完成后依次执行 `run_dense_prefix_check.py`、`diagnose_saved_mismatch.py`。BF16 的 spec-121 index=26 在相同已确认输入上复现：AR top2 同为 33.75、块验证为 34.00/33.75，最大 logits 差 0.25；query 长度 1/9，支持数值形状差异解释。该诊断不计速度。
- 新增逐文本片段、容量 4096 的 CPU embedding LRU，只给当前合法快照文本打分，查询仍在线编码。未缓存路径保留；H/RHD 使用独立空缓存做一次交错成本消融，不预先离线建库。部署清单 SHA-256：`8d0276f05a9867701466941fe2fa9f28c4bddb7d8aba98b4164e8528948f443d`。本机完整测试 **35 passed in 30.48s**，服务器 **35 passed in 5.40s**。

本机结果整理命令（纯报告，无 GPU）：

```powershell
.venv\Scripts\python.exe scripts/publish_research_artifacts.py --source results/raw/research-stage2 --destination results
.venv\Scripts\python.exe scripts/write_research_report.py
.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-final35
git diff --check
```

追加成本实验（服务器项目目录，主 GPU 运行完成后）：

```bash
OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python -u scripts/run_embedding_cache_check.py
```

结果路径：主报告 `docs/EXPERIMENTS_V2.md`，主配对记录 `results/qwen3-pilot/{memory-test,public-test}/replay-k2.jsonl.gz`，H 诊断 `prefix-test/` 与 `prefix-dense-check/`，成本消融 `embedding-cache-check/`。发布工具仅收集公开/合成实验的 JSON/JSONL/TXT/LOG，JSONL 无损 gzip；逐文件原始与发布 SHA 在 `results/research-v2-publication.json`。权重、数据库与原始下载档不上传。

成熟 B6/B7、自然长期用户数据、采样/并发仍未完成；下一机制重点是历史接续与固定 Target 对齐，而不是在这六个测试项目上继续换检索器挑最优结果。项目未被标记为六阶段完成。
