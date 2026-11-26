# MemoryDraft 项目交接

MemoryDraft 以一个固定 Target 的 greedy 输出为参照，利用历史文本来提出候选，希望减少推理成本。稳定演示使用 [infra-v0.3-interview](https://github.com/2223434778/MemoryDraft/tree/infra-v0.3-interview)，固定提交 `27e36199f8a65acdb9a9c42ccb744ddfa9a9402b`。当前研究分支是 `codex/branch-drift`，保留更多可选策略。树流的成本树、近似门控和网页归独立仓库，分仓前历史标签用于追溯。 [项目分工](PROJECT_SPLIT.md)

## 数据和解码契约

- 请求含 `id`、`user_id`、`order`、`messages` 和输出长度上限。Target 只对调用方提供的 messages 应用当前固定模板，生成比较时 prompt、权重、tokenizer 和数值设置都要一致。
- 历史由 SQLite 不可变事件表示，以用户、记录与可用顺序为键。快照只取严格早于请求 order 的事件，各记录保留此前最新的 active 状态。更新和删除用事件表达，同键内容一致可重试。程序记录 prompt 和快照摘要，方便审计相同对照。 [历史存储](../memorydraft/memory/store.py)
- 新运行时读取预先导入的 SQLite 历史。旧研究 CLI 有 `import` 和 `generate --commit-order`，后者可显式在更晚的可用顺序写回一轮。现在的 HTTP 服务没有历史提交 API。 [命令行入口](../memorydraft/cli.py)

## R+DFlash 运行时

1. 用 Target tokenizer 编码合法 assistant 历史和当前请求消息正文，把每个片段分开索引。由 `pydivsufsort` 建立固定宽度 uint32 字节后缀数组，匹配按 token 对齐，候选不能跨越记录末尾或特殊 token。默认使用全局匹配。 [后缀索引](../memorydraft/retrieval/suffix.py) · [匹配器](../memorydraft/retrieval/reuse.py)
2. 每轮取已确认**输出**的后缀，在合法资源中由长到短查找相同片段。索引窗口为 3–32，默认直接复用门槛为 6 token，候选上限为 32 并受剩余输出预算约束。R 命中时复制后续、跳过 Draft forward。未命中、处于冷却期或待补特征达到阈值时走 DFlash。研究中的引导版可能先跑 Draft 才决定拼接，费用需要单独计入。 [引擎](../memorydraft/runtime/engine.py) · [配置](../configs/runtime-dflash.json)
3. Target 一次验证 anchor 加候选，只接收从首位开始连续相等的 greedy token。首次分歧用 Target 的预测修正，全匹配可追加 Target 预测。正确性比较同时覆盖 EOS 和长度边界。 [验证循环](../memorydraft/backends/dflash_reuse.py)
4. anchor 是已确认但尚未缓存的 token。每轮开始，Target KV 长度等于 anchor 的位置，Draft KV 长度加待补的有效 Target 特征 token 数也等于该位置。验证后裁剪错误分支 KV，仅保留实际路径特征。连续 R 跳过 Draft 并累计特征，回到 DFlash 时一次补齐，恢复独立 Draft KV。 [验证和 KV 回归测试](../tests/test_dflash_reuse.py)

## 原 H/D/RHD 研究路径

H 用请求级词法或语义检索，从合法历史挑选片段写入小模型的参考区。H 与同指令空参考 D 共用权重，各有独立输入前缀、候选生成器和 KV，只追加已确认的 token ID。首次按需初始化，验证后马上裁剪未接受候选，切回时追赶已确认输出。研究后端复用 Transformers 4.52.2 assisted decoding，设有原生 D 和适配 D 对照。RHD 优先用合格的 R，再以历史观测的每提交 token 成本在可用 H/D/AR 路径间选择。旧共享候选池为消融，R 的全快照检索通常不受 H 的 top-k 限制。 [H 实现](HISTORY_DRAFTER.md) · [候选适配源码](../memorydraft/drafters/candidates.py) · [路由](../memorydraft/routing/policy.py)

## 服务和性能

原 infra 支持 AR、DFlash、R+DFlash。一个模型 worker 从有界队列串行执行。HTTP SSE 只流出已确认的 token IDs，最终返回完整文本。deadline 从提交开始，断连取消和背压通过 worker 执行，GPU kernel 不会被强制打断。定位问题可分别查看排队、引擎和客户端时延。基线是单 GPU、batch=1、greedy、FP32/eager。 [服务源码](../memorydraft/serving/) · [使用指南](INFRA_GUIDE.md)

benchmark 预热后构造 AR 参考，交错生成，核对 token、Target prompt、快照摘要，首个分歧会保留失败并停止汇总。完整在线成本含建索引、prefill、起草、验证、状态更新与同步。模型加载、HTTP 和加插桩的 profile 有独立口径。 [benchmark 源码](../memorydraft/runtime/benchmark.py)

| 实验 | 可以据此说明什么 |
| --- | --- |
| A6000 迁移，同 8 个受控配置修订请求 | 相同适配器关闭 R 为 42.060590 s，开启 R 为 33.881685 s。各组 24 条计时，约 19.4% 用时下降，逐 token 一致。120 条是五组和重复的总记录 |
| A6000 短请求工程验证 | 两条不同请求 × 三方法 × 两重复全部严格一致，六条 loopback HTTP 请求成功。该负载 R+DFlash 比 DFlash 慢 10.6% |
| 历史收益归因 | 展开现有请求事实后旧历史独有 oracle 覆盖仅剩 4/2452。该归因只测覆盖，不能用于声明替代方案实测速度 |
| H/OT | 同源历史开启 H 的在线耗时增加；OT 在 24 个新请求上整体较后缀慢 1.75%。实验实现保留 |

详细报告、失败原因、相关协议与证据目录见 [研究索引](RESEARCH_INDEX.md)。线性 ASD、Draft 适配、公开前缀缓存、RMSNorm 等是研究分支中的显式选项。近似生成需单独报告任务质量和输出长度。 [ASD 运行入口](ASD_RUNTIME.md) · [RMSNorm](RMSNORM_RUNTIME.md)

## 复现与待办

先在仓库根目录按 [首页](../README.md) 跑免权重 CPU demo。真机需要本地 Qwen3-8B 与 DFlash b16，依赖按旧研究 Transformers 4.52.2 与 DFlash 4.57.3 分开环境。模型与权重版本见 [Target 清单](../configs/qwen3-download-manifest.json) 和 [DFlash 清单](../configs/dflash-manifest.json)。目录迁移不会改变冻结标签或其历史证据。

待验证的是跨会话历史的独立增量，需比较 DFlash、只查当前请求、准许查历史三组，同时覆盖有用、无关和空历史。服务级显式历史提交仍待实现。交给另一个模型时，先让它复述一次 R→DFlash 切换，再从索引找到至少一份源码、一个回归测试和两组正反实验。原始私聊、数据库、权重、凭据及大型 trace 留在原环境。
