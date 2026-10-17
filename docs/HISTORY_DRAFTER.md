# H 的实现与验证

## 第二轮修订（2026-10-17）

以下正文中的 H/D 同指令比较仍保留；实用性能基线另有 `D_native`（官方 generator）与 `D_adapter`（同原生输入适配）。RD/RHD 的 D 路径使用原生输入，而固定 H 与带空 reference 的 `D` 用于隔离历史内容的效果。

R 默认对整个合法快照建立独立 uint32 后缀数组，查询完整已确认 Target 上下文，并用上游 prompt lookup 补充动态复用；不再受 H 语义 top-k 限制。`r_scope=shared_pool` 仅保留旧版消融。H 可用 dense 或 BM25-RRF 检索，同一请求内不重新检索。

Target 和 S 的生成入口显式关闭 `use_model_defaults`，避免 Qwen3 检查点的采样默认值回填；保持严格 greedy，不放宽 token 匹配。PLD 适配器保留最后一个 Target token 的长度预算。

同前缀诊断在生成 AR 评分参考之前确定 dense/hybrid/人工合法历史，保存来源和 payload 长度；空 H 与 D_format 候选完全相同，等长无关控制直接注入。第二轮主测试及追加全前缀结果见 [EXPERIMENTS_V2.md](EXPERIMENTS_V2.md)。H 的全前缀增量小且区间跨零，在线接受比例亦未提高，不能根据少量前缀位置宣布核心假设成立。

`CachedSemanticEncoder` 是显式启用的逐片段向量 LRU，默认 `SemanticEncoder` 保留未缓存路径。每个实例固定 encoder 权重/revision，键是原始片段文本；查询向量仍在线计算，只有 `retrieve()` 从当前合法快照传入的文本被打分。没有缓存由未来上下文汇总出的摘要、排名或答案。缓存不写盘；淘汰与重新编码测试在 `tests/test_embedding_cache.py`。追加实验为 H/RHD 分别建立空缓存，首次编码仍计费，不能拿重复请求命中效果冒充从未见过的新历史收益。

## 初版 H/KV 设计

H 和 D 使用相同独立小模型和完全相同的起草规则，唯一不同是 `<reference>` 区域内容。两者都保留应用给出的原始 messages，包括应用事实记忆。Target 的 messages 在所有方法间保持完全相同。

`TransformersBackend.prompt()` 使用模型 chat template 预构造固定前缀，开放 assistant 回答。每轮直接追加共同的已确认 `Y` 的 token IDs，不 decode/re-encode 当前回答，不添加“继续”消息。H 可以生成新 token，并非复制旧答案；复制单独走 R。

`IndependentDraft` 每条路径持有一个 `AssistedCandidateGenerator`；H/D 共用权重但各有独立缓存。首次使用才初始化、prefill；不活跃路径不计算，切回时由上游 generate 对尚未覆盖的已确认 IDs 做追赶。记录 `context_epoch`、`committed_output_len`、`kv_covered_output_len` 和首次/追赶输入 token 数。首版固定历史前缀，epoch=0。

重要的两处适配：

1. 首次启用 H 可能已经有 Y，因此 mask 从第一轮就覆盖 `C_H+Y`；上游只在已有 KV 时自动调整 mask，不能直接沿用构造器的短 mask。
2. 必须在验证后立即 crop 已使用路径的 rejected suffix。不活跃期间只记录真实前沿；切回时不能依据变长的当前输出长度“保留”以前未接受的 token。

R 在请求池的 token n-gram 索引中寻找最长连续匹配；不跨片段、记录或特殊 token 边界。首轮没有输出前缀时不强行复制历史开头。路由先看 R 匹配，再对 H/D 各做少量计时探测，随后用过去轮次的单位提交 token 成本 EMA 选择；R 连续失败设冷却。没有事后最优路由或每轮同时跑三个提议器。

测试在真实 Qwen2 模型类的随机微型网络上调用上游验证/缓存代码，覆盖全接受、首拒绝、中途拒绝、EOS、长度限制、空历史、R/H/D/AR 切换及异常恢复。微型网络测试只证明适配逻辑，性能结论仅取实际 GPU 权重运行。

## 计时解释

请求总延迟在模型与检索器加载/预热后开始，覆盖 tokenization、请求检索（含当前实现的候选 embedding）、prompt 构造、建接续索引、H 初始化、prefill、起草、追赶、Target 验证和回滚，在请求两端同步 CUDA。回放的 snapshot 读取成本按共享成本添加到每个方法，所有方法使用同一冻结快照。

AR/D 不做 Draft 专用检索；R/RD/H/RHD 才承担增量检索成本。应用事实检索若由外部提供，必须作为共同在线成本另行记录；目前合成 pilot 是设置 A，而不是完成应用 RAG 设置 B。

分阶段 `*_host_seconds` 是主机时间，GPU 异步时不能与 CUDA 内核耗时等同，也不能简单相加。`round_seconds` 包含生成与验证等待，供因果路由反馈；请求总延迟是权威性能指标。没有把首次 H prefill、追赶或缓存切换作为免费操作。TTFT 是首个已确认 token/block 对外可用时刻，speculative 模式可能一次交付多个 token。

当前逐轮保留聚合事件，会有仪器开销，适合机制 pilot。正式性能研究还需要轻量/详细 trace 对照、重复与交错顺序、成熟 Draft 对照及更长输出。
