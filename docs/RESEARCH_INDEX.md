# MemoryDraft 研究索引

从 [项目首页](../README.md) 开始。需要完整背景和代码路径时读 [交接说明](MEMORYDRAFT_HANDOFF.md)，需要追溯某次运行时读 [进度日志](PROGRESS.md)。原首页的详细时间线也可在 [整理前的固定快照](https://github.com/2223434778/MemoryDraft/blob/34b1b81cd13b20923ae32a2819c5d8321703a30d/README.md) 找到。

| 主题 | 建议阅读顺序 | 已确认的结论 |
| --- | --- | --- |
| 项目假设与底座 | [原架构](ARCHITECTURE.md) → [底座选择](BASE_SELECTION.md) → [第二轮实测](EXPERIMENTS_V2.md) | 固定 Target，分别评测直接复用 R、历史辅助起草 H 和普通 Draft |
| H 和路由 | [实现](HISTORY_DRAFTER.md) → [因果诊断](H_CAUSAL_RESULTS.md) → [同源历史在线评测](H_TEACHER_ROUTER_RESULTS.md) → [训练和 DFlash 对照](H_TRAINING_RESULTS.md) | H 路径已实现，历史辅助尚未显示稳定在线净收益 |
| R+DFlash 场景验证 | [开发结果](REUSE_PILOT_RESULTS.md) → [合成确认](REUSE_CONFIRMATION_RESULTS.md) → [运行时迁移](RUNTIME_MIGRATION_RESULTS.md) → [归因](REUSE_ATTRIBUTION_RESULTS.md) | 受控配置修订负载较 DFlash 用时低 19.4%，保持严格输出。长期历史的独立作用仍待验证 |
| 更真实的工作负载 | [公开 CI 配置](CONFIG_REVISIONS_DEVELOPMENT_RESULTS.md) → [完整文件合同诊断](CONFIG_REVISIONS_CONTRACT_PROBE_RESULTS.md) → [发布说明筛选](RELEASE_TRIAGE_RESULTS.md) | 工具可复用已有片段，但 Target 输出合同与现有输入资源影响任务是否值得采用 LLM |
| OT、编辑距离与内部拼接 | [新请求验证](OT_R_VALIDATION_RESULTS.md) → [Draft 前缀引导](OT_GUIDED_RESULTS.md) → [拼接与冷却](OT_SPLICE_COOLDOWN_RESULTS.md) → [收益门控](OT_BENEFIT_RESULTS.md) → [风险续测](PREFIX_RISK_CONTINUATION_RESULTS.md) | 保留实验选项，默认仍使用简单后缀匹配 |
| 早期线性 ASD | [ASD 与拼接](ASD_SPLICE_RESULTS.md) → [预算与收益归因](ASD_ATTRIBUTION_RESULTS.md) → [跨任务比较](ASD_COMPARISON_FINDINGS.md) → [可选运行配置](ASD_RUNTIME.md) | 允许近似输出，依赖任务等工作量及质量对照。此路径与严格 R+DFlash 分开 |
| 执行和缓存研究 | [RMSNorm](RMSNORM_RUNTIME.md) → [长输入瓶颈](ASD_LONG_CONTEXT_FINDINGS.md) → [公开前缀](ASD_PREFIX_CACHE_FINDINGS.md) → [共享公共段](ASD_SHARED_PREFIX_FINDINGS.md) | 都有各自的显式配置和评测，需关注冷编译、命中、拷贝成本和样本范围 |
| Infra 交付 | [使用指南](INFRA_GUIDE.md) → [工程验收](INFRA_VALIDATION.md) → [冻结版与归档](INFRA_INTERVIEW.md) | v0.3 演示版本覆盖严格解码、HTTP/SSE、性能分析和真机验收 |

### 原始记录怎么用

| 目录 | 核查内容 |
| --- | --- |
| [`results/infra-v0.3`](../results/infra-v0.3/) | GPU/HTTP 工程验收、请求明细、profile 算子表、发布清单 |
| [`results/runtime-migration`](../results/runtime-migration/) | AR 参考、迁移计时、候选资源与一致性检查 |
| [`results/reuse-confirmation`](../results/reuse-confirmation/) | 合成确认测试的 replay、历史、参考与 provenance |
| [`results/reuse-attribution`](../results/reuse-attribution/) | 当前请求事实展开、oracle 覆盖、归因边界 |

评测报告与原始输入保持原路径，便于复算、比对校验清单和重放。优先使用索引浏览，不必从头读全部进度。分仓前的成本树与网页证据见 [项目分工](PROJECT_SPLIT.md) 和独立的 [ShuLiu-Decode](https://github.com/2223434778/ShuLiu-Decode)。
