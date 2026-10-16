# 底座选择与可复用边界

## 已选底座

Transformers **4.52.2**，tag 对应 commit `237c7c356cce5f8a6befc0a5f71bc952bb5adc9a`，Apache-2.0。
复用 `GenerationMixin.generate/_assisted_decoding` 的 Target teacher-forcing、最长一致前缀、纠正/bonus token、停止规则和 KV 裁剪；复用 `AssistedCandidateGenerator` 的独立小模型生成与缓存管理。
项目没有复制这两段实现，也没有修改上游 attention、验证公式或模型权重。

新增适配位于 `memorydraft/backends/transformers.py` 与 `memorydraft/drafters/candidates.py`：

- 请求范围内替换 Target 实例的 `_get_candidate_generator`，结束/异常后恢复；锁限制为 batch=1、单请求。
- H/D 各持有一个 upstream candidate generator，权重相同、prompt/KV 隔离；按实际已确认 token IDs 追加 Y。
- 独立上下文改变长度，因此对 Draft 的 `max_length` 做同等偏移；不修改 Target 的长度。
- 验证后立即裁剪所有暂存候选，再让不活跃路径懒追赶。首次延迟激活也必须把 mask 扩到 `C_route+Y`。
- 只支持 greedy。采样参数明确报错，不冒用缺少 proposal 分布的接口。

接口为版本相关的私有扩展点，运行时严格检查版本。升级必须重新做缓存、拒绝及路由切换回归。

## 上游审查

| 项目 | 核验 commit | 结论 |
|---|---|---|
| [SAM-Decoding](https://github.com/hyx1999/SAM-Decoding) | `aaf939819223147ce544e3a80f65758ac7758076` | `samd/draft.py` 在后缀匹配与 tree model 间选择；`samd/samd_model.py` patch 模型/attention 并管理序列或树缓存。不是现成的任意独立上下文 H 接口。该提交树未见 LICENSE，不复制代码。小候选池先自建 n-gram 接续索引。 |
| [RAPID](https://github.com/real-absolute-AI/RAPID) | `22d41f4113fe862bc80b35f770218360182a1be3` | `src/generator.py` 与 `src/model_wrap.py` 组织独立 assistant 输入；完整算法包含本项目不采用的分布改动。该提交树未见 LICENSE，不复制代码。 |
| [EAGLE](https://github.com/SafeAILab/EAGLE) | `cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b` | Apache-2.0；README 列出官方 `Qwen2-7B-Instruct` ↔ `yuhuili/EAGLE-Qwen2-7B-Instruct`。这不是本轮 Qwen2.5-1.5B 的匹配权重，不能换名作为 B6。EAGLE 依赖 Target 特征，不能直接注入 H prompt。尚未实现 B6/B7。 |

## 首轮模型及硬件

| 资产 | 固定版本 |
|---|---|
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) Target | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`；远端已有快照 |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) H/D | `7ae557604adf67be50417f59c2c2f167def9a775`；普通指令小模型，未蒸馏 |
| [多语言 MiniLM](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`；CPU、可选语义检索 |

上述模型卡标注 Apache-2.0。权重与模型许可保存在模型目录，不进入项目 Git。
加载时比较完整 token-ID 映射、特殊 token、fast tokenizer 后处理配置和输出 head；单看 vocab_size 不够。
Qwen2.5-7B head=152064，1.5B/0.5B head=151936；本版本保守拒绝该组合，未改变 padded token 的 Target 分布来强行兼容。

本机 RTX3090 24GB，但当前 Python 是 CPU PyTorch；实际运行使用 A6000 服务器 GPU0。服务器 Linux，D 盘为 `/media/a6000/D`，项目独立目录 `/media/a6000/D/ZhaoBinHe/MemoryDraft`，GPU1 留空。
远端隔离 venv 复用已有基础环境的 torch 2.5.1+cu124、Transformers 4.52.2，没有修改其他项目环境。执行前后以 `results/environment.json` 为准。

BF16/SDPA 初测出现真实 greedy 分歧；`scripts/diagnose_precision.py` 在同一请求上记录 BF16 分歧及 FP32/eager 一致性。本轮有效参考配置为 `configs/a6000-validation.json`，不把 BF16 的失败运行汇总成无损加速。

## 第二轮执行补充（2026-10-16）

- 主验证底座仍为锁定的 Transformers 4.52.2。增加原生 assistant D、原生上下文适配 D、prompt lookup 控制；RD/RHD 的 D 使用原生上下文，旧 D 保留为 H 的同指令对照。
- 新版 checkpoint 的默认参数可能覆盖 GenerationConfig 的全局默认值；Qwen3 默认为采样。Target 与 S 入口现显式关闭 `use_model_defaults`，Target 另显式传入 `do_sample=False`。采样回填失败记录单独保留，不计 greedy 成功次数。
- 上游 prompt lookup 的候选可能超过剩余输出长度，增加保留 Target 纠正/bonus token 的长度边界；保持验证规则不变。
- R 使用 `pydivsufsort==0.0.19`（MIT）的构建/搜索，增加固定宽度 uint32 token 编码与记录边界薄适配，不自写后缀数组构建。审查的仓库 HEAD 为 `2080e85c28cefca094f9c3a7a304db018f0d3d5b`；运行依赖按发布版本锁定，不把 HEAD 当作该 wheel 的构建 commit。
- H 混合排序复用 `rank-bm25==0.2.2`（Apache-2.0）及既有 MiniLM；审查 HEAD `47aa3ddf8dc1ebeb7ef4e65f2b4536af44594099`，同样与 wheel 发布版本分开记录。
- 模型为 Qwen3-8B/0.6B，固定 HF revision 和 tokenizer 预检见 `results/research-audit/model_preflight.json`。权重镜像 URL、大小和 HF SHA-256 见 `configs/qwen3-download-manifest.json`；逐文件校验后才允许启动新套件。新实测报告由 `scripts/write_research_report.py` 生成。

### 首轮未覆盖的边界（保留）

目前是单请求机制研究适配器，不是生产推理引擎；无树融合、跨卡流水、并发或采样。语义模型需要独立下载和 optional dependency。B6/B7 需要匹配的 Target/EAGLE 组合及特征恢复适配，不能用普通 S 的 D 冒充成熟学习型 Draft。
