# 下一阶段：可加速的 D/R 基线与 H 的独立增益

2026-10-16。本文响应用户的新研究要求，修订 v0.2 中 R 必须受 H 语义候选池限制的决定。本文包含已完成的源码/日志核查与待执行的实验协议；没有新增 GPU 性能结果。

## 1. 明确研究问题和通过条件

投机解码与检索式起草已有论文证据。本项目首先要复现适用条件下 D、R 相对 AR 的真实加速，再检验 H 是否提高同一个 S 对固定 Target 的预测准确性。已有普通 D 失败不能作为投机解码无效的证据，简化的 R 也不能冒充官方 REST 性能。

- 基线通过：同一 Target、输入、精度、停止规则和硬件，在预先固定的测试集合上，D、R 各自的总 AR 时间 / 总方法时间 > 1，且收益在配对重复中稳定。报告全部请求，包括 R 无命中与短输出，不能事后筛选成功子集。特定复用场景的结论须明确限定场景；不能保证任何请求都更快。
- H 机制通过：相同已确认前缀、同一 S、相同候选长度下，H 相对同指令 D 的平均连续接受长度有可重复正增量。进一步检查 R 无强匹配的位置，检验 H 的互补价值。
- H 系统通过：计入专用检索、历史 prefill、KV 追赶之后，H/RHD 相对合理 D/RD 基线仍节约时间。机制通过与系统通过分别报告。
- `D_format` 用于归因，`D_native` 用于实用比较；即使 H 胜过被附加指令拖慢的 `D_format`，仍须报告是否胜过 `D_native`，不能弱化基线制造增益。
- 最终研究还需成熟学习型 Draft 对照 B6/B7。独立 S 的 D、EAGLE 和 H 不能混成一个名字，也不能用 EAGLE 更快来代替 H 相对同一个 S 的证明。

## 2. 论文对应到具体复用位置

| 原始来源 | 采用的依据或组件 | 边界 |
|---|---|---|
| [Leviathan et al., ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html) | 标准起草—验证机制；评估起草成本与接受长度的共同作用 | 论文有效不等于任意模型比或实现必然加速 |
| [Transformers 4.52.2 官方文档](https://huggingface.co/docs/transformers/v4.52.2/en/generation_strategies) | 原生 `assistant_model` D、`prompt_lookup_num_tokens` R 控制组、现有严格验证和 KV 管理 | prompt lookup 只覆盖输入/当前序列，不能称为跨会话历史检索 |
| [REST, NAACL 2024](https://aclanthology.org/2024.naacl-long.88/) / [官方代码](https://github.com/FasterDecoding/REST) | token 后缀查询、历史接续、候选频次排序；官方树验证作为外部对照 | 线性候选适配不是完整 REST 复现；原索引有下述适配问题 |
| [SAM-Decoding, ACL 2025](https://aclanthology.org/2025.acl-long.595/) / [代码](https://github.com/hyx1999/SAM-Decoding) | 静态语料＋当前已确认序列的动态接续，按匹配信号选择路径 | 当前提交未见许可证，不复制实现；论文报告包含与 EAGLE-2 的组合结果 |
| [RAPID, ICML 2025](https://proceedings.mlr.press/v267/chen25s.html) / [代码](https://github.com/real-absolute-AI/RAPID) | 使用检索出的短上下文条件化独立 Draft；其 `src/rag.py` 使用 embedding/FAISS，另有 BM25 路径 | 完整方法调整 Target 分布，不能直接用于固定 Target 的无损结论；当前提交未见许可证 |
| [DReSD](https://arxiv.org/html/2502.15572v1) | 用 Target 上下文化 token 向量检索后续 token，说明 R 不局限于字符串匹配 | 不等同于句子 embedding RAG；需要隐藏状态、专门数据存储及成本评估，留作 R 的后续增强 |
| [RASD, Findings ACL 2025](https://aclanthology.org/2025.findings-acl.320/) | 检索候选与模型候选的树剪枝/融合 | 这是候选融合先例，不是“给 S 历史就一定提高接受率”的证据；不在机制实验前增加树融合 |

“给 Draft 检索上下文”已有先例。拟研究的增量是账户历史、时间版本、独立 H 的接受增益以及与 R/D 的互补，不能把 RAG+Draft 本身作为新颖性主张。

## 3. 本轮已经核实的问题

1. 当前 `TransformersBackend.prompt(messages, "")` 给 D 加了参考说明和空 reference。它可充当 H 的同指令对照，但还缺少直接使用原始请求的原生 D。先区分 `D_native` 与 `D_format`，不把指令差异归因于历史。
2. R/H 都读取同一 `MemoryBundle`。R 的索引只包含语义 top-k 结果；当前 R 还只查询已生成 Y，且没有动态索引当前已确认序列。相对于 REST/SAM，覆盖范围较窄。具体损失要用对照测量，不能仅由源码推断数值。
3. 复算 `results/semantic/replay.jsonl`：每种方法最终输出 436 token。R 在 424 轮中只有 4 轮起草（0.94%），17 个候选接受 12 个（70.59%），仅贡献全部输出的 2.75%。这是“候选接受率尚可、实际覆盖很低”，不是 R 接受率约 1%。D 为 322/819、每轮接受 2.825；H 为 320/852、每轮接受 2.759。D/H 分块位置不同，这些汇总不能代替同前缀因果对照。
4. 现有语义检索每次重新编码候选历史；应预编码不可变历史版本并按合法快照取用，单独计入写入/建索引成本。新请求仍计入查询编码、过滤、搜索和必要的冷加载成本。
5. REST `154d0f0ae45de5d8a9cbd1c696f6ad5da10904bf` 根目录 Apache-2.0，`DraftRetriever` 子目录 MIT。其 [lib.rs](https://github.com/FasterDecoding/REST/blob/154d0f0ae45de5d8a9cbd1c696f6ad5da10904bf/DraftRetriever/src/lib.rs#L106) 将 token 转为 u16 保存，65535 以上会截断；`add_entry` 直接拼接且分隔符被注释。Qwen 不能直接使用该文件格式，历史记录边界也需要保护。此发现来自静态源码审查，尚未编译运行上游。
6. SAM/RAPID 的核验提交分别为 `aaf939819223147ce544e3a80f65758ac7758076`、`22d41f4113fe862bc80b35f770218360182a1be3`，此次 GitHub API 查询未见 LICENSE。EAGLE 为 `cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b`。没有新复制上游代码。

离线审计产物：`results/research-audit/pilot_summary.json`，含输入 SHA-256；模型预检：`results/research-audit/model_preflight.json`。后者不是 GPU 兼容或速度证明。

## 4. 底座与模型选择

先保留 Transformers 4.52.2 为独立 S 的主验证底座，用官方原生 D 与 prompt lookup 校准已有适配器开销。原有 1.5B/0.5B 组合保留为回归样例，不继续作为主要性能结论的唯一配置。

下一组优先候选为 Qwen3-8B Target + Qwen3-0.6B 独立 S，在 A6000 GPU0 测试。两者均为公开 Apache-2.0 模型，已读取固定版本配置及 tokenizer，**两个 tokenizer 文件分别逐字节哈希相同，配置 vocab_size 同为 151936**：

- Target：`Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`。
- S：`Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`。
- 尚未下载权重、执行 runtime fingerprint 或推理；速度不预设。Qwen2/Qwen2.5 的 7B 与 0.5B 输出 head 大小不同，不能未经适配直接替换旧配置。
- [Qwen3 官方说明](https://huggingface.co/Qwen/Qwen3-8B)支持 `enable_thinking=False`。初期统一显式非思考模式、greedy 作为严格 token 一致性实验；这不是官方推荐采样配置的完整质量评测。模板开关必须对所有方法一致，并保存最终 Target prompt hash。
- 首先单变量检查 BF16/eager、BF16/SDPA 的一致性。若失败，保存样例并检查位置/KV/数值，保持所有方法同配置；不靠提高 AR 精度或放宽接受规则产生加速。
- [EAGLE 当前模型表](https://github.com/SafeAILab/EAGLE#eagle-3-weights)列出 Qwen3-8B 的社区 EAGLE-3 权重，标记为非官方。下载前必须再核验 checkpoint 的 base revision、许可、模板和实际加载路径；不能把列名匹配当成 B6 已就绪。官方 Qwen2-7B 配对是另一条已知路径，但其 D/H 小模型 head 问题仍需处理。

R 的实现优先顺序：官方 prompt lookup 校准 → 复用 REST 检索器的明确版本适配（32 位 token 格式、记录边界、无命中等关键测试）→ 接入当前上游验证器。若 Rust 适配超出薄改动范围，采用支持整数数组的成熟索引库，例如 [pydivsufsort](https://github.com/louisabraham/pydivsufsort)，先核验版本/许可证；不能再自写后缀数组构建或 Target 树验证引擎。官方 REST 全树路线应单独标记，速度只与其相同后端/Target 的 AR 配对，不跨引擎挑选较慢 AR。

## 5. R/H 分离后的检索协议

| 项目 | R | H |
|---|---|---|
| 输入 | 当前合法上下文与已确认输出的 token 后缀 | 当前请求、近期可见上下文、项目/任务元数据 |
| 输出 | 原始接续 token 及记录/位置出处 | 少量有边界的原始历史片段或问答示例 |
| 首选算法 | REST 式后缀匹配＋频次；结合当前确认序列的动态复用 | 复用成熟 BM25 与现有 dense encoder，分别检索后做 RRF 排序融合 |
| 可见范围 | 合法用户历史的独立索引；无需通过 H 的语义 top-k | 同一合法快照，经相关性排序并限制上下文预算 |
| 时机 | 每轮廉价查询；只加入已确认 token | 每请求一次，稳定前缀与独立 KV；初期不每轮重检索 |
| 衡量 | 有候选轮次占比、接受长度、被接受 token 的输出覆盖、总成本 | 同前缀 H−D 接受长度、负增益比例、总成本 |

以上 H 混合检索是本项目待验证的工程选择，不是 RAPID 已证明它最优。先用已有 MiniLM + BM25 控制变量；仅在合法人工选择的历史确有收益、自动检索却失败时，再对照 RAPID 的 BGE-M3 或增加重排。避免同时更换小模型、embedding、上下文格式和候选长度，导致无法归因。

R/H 共用用户、时间和版本过滤，而不共用强制 top-k 截断。R 索引保存原始 token 和记录末端，禁止跨记录/特殊 token 边界续写；H 的检索键可包含旧问题，返回值保留其旧回答和来源。不可将未来更正、答案或未来生成的摘要/向量混入过去快照。

过滤还必须作用于排序统计：BM25 的文档频率、R 的接续频次只能来自当时可用的记录；不能先用含未来记录的排名统计，再只过滤最终返回项。

## 6. 两层实验，各自回答一个问题

### 实验 A：D/R 的加速基线

固定 Target 输入与硬件，比较 AR、官方 `D_native`、当前适配器的同输入 D、`D_format`、官方 prompt lookup，以及独立历史 R。先用 8 个短请求检查一致性和缓存，再在开发集仅校准候选长度 2/4/8；固定配置后评测保留集。8、2/4/8 等均为本项目初始预算，不是论文结论。

使用 [Spec-Bench](https://github.com/hemingkx/Spec-Bench) 的代码、摘要、QA 等公开任务校准，以及明确有合法可复用历史的请求评估 MemoryDraft。数据版本、许可、任务抽样和排除规则在运行前锁定；历史不得包含当前测试答案。原 13 条合成请求继续作为独立回归集。

开发集与测试集按项目/历史来源分开；先用小规模集合定位问题，只有通过后才做 3 次交错配对重复。AR 也参与次序随机化，不能永远先跑；使用单独生成的 AR 参考 token 检查正确性。报告请求总延迟、TTFT、解码耗时、峰值显存、CPU 索引内存，以及冷/热索引范围。全部在线索引加载/检索/验证计费，预处理和存储另列。置信区间按请求或共享历史的用户/项目成组计算，不能把 token 或重复运行视作独立样本。

若官方 D 快而适配 D 慢，修适配器；若两者均慢，检查模型成本比和候选策略，再换组合。若 R 候选接受高但覆盖低，修语料/索引/查询范围；若覆盖足够却慢，查搜索与验证开销。没有跨过基线门槛，不继续扩大路由算法实验；H 的离线机制诊断可以独立进行。

### 实验 B：H 是否提高命中

主场景采用架构设置 B：应用预先给 Target 的事实记忆固定，所有方法相同；D 也保留相同应用事实。H 额外取得合法历史原文/表达示例。设置 A 单列，不能暗中给 H 新事实而期望固定 Target 接受。

1. 用固定 Target AR 生成参考轨迹。预先规定每隔固定 token 数抽取前缀，含开头位置；抽样间隔在开发阶段锁定，不按哪些位置 H 表现好挑选。
2. 在相同前缀 Y 上，让同一 S 分别生成候选：`D_native`、`D_format`、H-dense、H-hybrid、等预算无关历史 H、人工选择合法历史 H。空历史 H 应与 `D_format` 一致。人工选择者只看请求及过去历史，不看参考答案；它是诊断条件，不是数学上界或生产检索结果。
3. 固定每组候选块长 k，比较与 Target 参考接续的最长相同前缀 L，记录 `L_H-L_D_format`、首 token 命中、全块接受、按位置生存曲线 `P(L>=j)`，以及 H 降低接受长度的比例。EOS/剩余长度单独处理，同一前缀各方法使用相同有效预算。
4. `E[L] = sum_j P(L>=j)`，因此连续接受长度比不控制块长的总体 accepted/proposed 更直接。该指标是 greedy 下的离线提议质量，不等于随机采样接受概率；也不把离线 teacher-forcing 耗时冒充在线加速。
5. 参考答案仅供评分；检索器、提示、索引和路由绝不读取它。正式端到端运行仍用上游 Target 验证，不能通过与已存答案比对绕过 Target。
6. 按预先标记的“原文复用、局部修改、语义相关但措辞不同、无关/新任务、旧版本冲突”分层；另按 R 查询的可观测强/弱匹配分层。如果 H 只在 R 已能直接复制的位置提高命中，其互补贡献不足。

[LongMemEval](https://github.com/xiaowu0162/LongMemEval) 可补充带时间戳的多会话记忆任务，但它包含模拟历史，不能称为真实用户日志；其 QA 准确率也不是投机命中指标。使用正式带干扰历史版本作为检索输入，不使用 oracle 证据集合/答案字段选取测试候选。长输出收益还需代码修改、文档迭代等预先定义任务，不能只靠短事实 QA。

## 7. 根据结果决定下一步

| 结果 | 下一步 |
|---|---|
| 人工合法历史 H 有增益，自动 H 无增益 | 改检索/片段组织，固定 S 与验证器 |
| dense、hybrid、人工历史均无增益 | 检查 Target/H 可见条件、格式干扰和任务；有限对照后如实记录假设未获支持 |
| H 接受长度增益，但端到端更慢 | 缩短历史前缀、预编码/缓存、降低检索成本；报告机制成立但系统收益未成立 |
| H 在 R 弱匹配区有增益且净成本下降 | 再集成 RHD，最后评估相对成熟 B6 的 B7 增益 |

下一步代码顺序：原生 D/同指令 D 和 prompt lookup 对照 → 独立 R 索引 → 同前缀 H/D 评估 → H 混合检索 → 路由/B6/B7。保留现有存储、快照、严格验证、KV 隔离和回归测试，不整体重写工程。

## 8. 本轮可复现核查

```powershell
python scripts/audit_pilot.py --input results/semantic/replay.jsonl --output results/research-audit/pilot_summary.json
.venv\Scripts\python.exe -m pytest -q tests/test_pilot_audit.py
git diff --check
```

模型元数据读取来源是 Hugging Face `/api/models/<model>` 的 commit 和对应 `/resolve/<commit>/config.json`、`tokenizer.json`、`tokenizer_config.json`；对两个 tokenizer 文件的原始字节分别计算 SHA-256。没有下载权重。固定版本和哈希已存入 model_preflight.json，可独立重取核验。
