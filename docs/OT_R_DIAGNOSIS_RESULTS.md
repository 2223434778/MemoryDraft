# OT-R 实际触发位置的错误定位

2026-10-19。八个已看过请求×三种检索，共24次插桩回放；不是性能测试或独立场景验证。

全部输出与既有AR的token、Target输入和合法快照严格一致。参考答案仅在生成结束后评分。

## 对齐调用

| 方法 | 调用 | 提议 | 对齐接受token | 全部解码轮 |
|---|---:|---:|---:|---:|
| Suffix-R | 0 | 0 | 0 | 68 |
| Edit-R | 9 | 9 | 65 | 71 |
| OT-R | 9 | 9 | 97 | 68 |

## 实际对齐决策

raw/pool是事后最好连续接续长度，>=8只是诊断分层，不是盈亏线。

| 请求 | 方法 | 已生成token | raw最好 | pool最好 | 实际接受 | 分类 |
|---|---|---:|---:|---:|---:|---|
| parcel-queue:0 | Edit-R | 17 | 2 | 1 | 0 | no_long_raw_span |
| parcel-queue:0 | OT-R | 17 | 2 | 1 | 0 | no_long_raw_span |
| parcel-queue:1 | Edit-R | 13 | 1 | 0 | 0 | no_long_raw_span |
| parcel-queue:1 | OT-R | 13 | 1 | 0 | 0 | no_long_raw_span |
| payment-runbook:0 | Edit-R | 19 | 32 | 32 | 32 | long_pool_best |
| payment-runbook:0 | OT-R | 19 | 32 | 32 | 32 | long_pool_best |
| payment-runbook:1 | Edit-R | 11 | 3 | 3 | 1 | no_long_raw_span |
| payment-runbook:1 | Edit-R | 15 | 32 | 0 | 0 | shortlist_miss |
| payment-runbook:1 | OT-R | 11 | 3 | 3 | 1 | no_long_raw_span |
| payment-runbook:1 | OT-R | 15 | 32 | 0 | 0 | shortlist_miss |
| short-settings:1 | Edit-R | 10 | 0 | 0 | 0 | no_long_raw_span |
| short-settings:1 | OT-R | 10 | 0 | 0 | 0 | no_long_raw_span |
| confirm-schema-0:0 | Edit-R | 130 | 32 | 32 | 0 | ranking_miss |
| confirm-schema-0:0 | OT-R | 130 | 32 | 32 | 32 | long_pool_best |
| confirm-schema-0:1 | Edit-R | 131 | 32 | 32 | 32 | long_pool_best |
| confirm-schema-0:1 | OT-R | 131 | 32 | 32 | 32 | long_pool_best |
| copy | Edit-R | 46 | 1 | 1 | 0 | no_long_raw_span |
| copy | OT-R | 46 | 1 | 1 | 0 | no_long_raw_span |

## Suffix与OT的首次分歧

同请求仅比较首个分歧；之后KV/特征累积路径可能不同。advance包含一个Target确认token，EOS处最终发布长度可截断。

| 请求 | 位置 | Suffix路线/接受/推进 | OT路线/接受/推进 |
|---|---:|---|---|
| parcel-queue:0 | 17 | D/15/16 | R/0/1 |
| parcel-queue:1 | 13 | D/5/6 | R/0/1 |
| payment-runbook:0 | 19 | D/12/13 | R/32/33 |
| payment-runbook:1 | 11 | D/3/4 | R/1/2 |
| short-settings:1 | 10 | D/1/2 | R/0/1 |
| confirm-schema-0:0 | 130 | D/15/16 | R/32/33 |
| confirm-schema-0:1 | 131 | D/15/16 | R/32/33 |
| copy | 46 | D/3/4 | R/0/1 |

## 原未插桩计时的逐请求拆分

下表直接汇总原两次重复，不使用此次插桩耗时。新请求在本次诊断前已经观察，不作新正例场景确认。

| 请求 | Suffix秒 | Edit秒 | OT秒 | OT相对Suffix延迟 |
|---|---:|---:|---:|---:|
| parcel-queue:0 | 1.549221 | 1.707249 | 1.740874 | +12.37% |
| parcel-queue:1 | 1.704949 | 1.721436 | 1.758209 | +3.12% |
| payment-runbook:0 | 1.845608 | 1.721428 | 1.757316 | -4.78% |
| payment-runbook:1 | 1.707892 | 2.032325 | 2.085203 | +22.09% |
| short-settings:1 | 0.571894 | 0.693178 | 0.716209 | +25.23% |
| confirm-schema-0:0 | 2.380276 | 2.539917 | 2.153343 | -9.53% |
| confirm-schema-0:1 | 2.379090 | 2.132935 | 2.166700 | -8.93% |
| copy | 1.373279 | 1.494208 | 1.532735 | +11.61% |

## 机制解释与最小改进方向

1. 新五请求的六次OT调用：四次不存在>=8-token原始接续，一次候选池漏召回，一次选中32-token接续。主要损失发生在编辑尚未完成或输出临近结束时，不能归为OT把池内长候选排错。
2. parcel-queue:0在新值72只生成7时提议旧值的8，首token即拒绝；同状态Suffix转DFlash接受15。应先让模型跨过新内容，再寻找恢复复制的位置。仅提高求解精度不会创造不存在的连续span。
3. payment-runbook:1插入Region行后，真实Severity接续可接受32，但现有池全为0。当前倒排只记录端点>=min_context的位置，早期Owner不能作旧记录锚点；稀有新词Region/eu/-west只命中当前修改指令。固定偏移把候选推入Severity中间。
4. 该单点离线反事实shortlist-analysis.json：只把候选数8扩大32仍没有长候选；max_shift从2扩大5后池内出现32-token接续，但OT和Edit仍选择0-token接续。它同时涉及召回和排序，不能靠单项扩容修复。此反事实不修改线上参数、不作速度证据。
5. payment-runbook:0确实恢复32-token并在原计时中更快；项目整体负结果来自另一个请求的较大损失。不能据项目均值否认局部成功，也不能据单请求成功确认稳定适用场景。Edit同样成功且更便宜。
6. confirm-schema-0:0在共同状态下OT接受32，Edit接受0。两端点Edit分数同为0.9375，按位置取更早端点；OT分别约0.934891/0.941865，选对后一个。旧局部正例有真实消歧机制，但不证明OT优于所有便宜排序器。
7. 先改触发决策与编辑后端点召回，再研究接续排序；保留旧正例作回归。不能硬性要求末token精确相同：旧正例成功OT的exact_tail=0，简单提高末尾匹配要求会删掉它。延迟构造索引和求解器加速可降低成本，但不能替代上述机制改动。

## 限制与复算

原未插桩性能证据见OT_R_RESULTS.md和OT_R_VALIDATION_RESULTS.md。本报告不从插桩耗时推导加速。
候选池诊断包含记录/特殊token边界、用户时间快照、历史最小上下文与实际阻塞。oracle只说明该位置存在什么，不能作为线上选择器。
默认suffix和独立infra面试版保持原状；没有在本批答案上修改参数。

执行package SHA：`35603a53734dde44673730c53abacbe5756e27dab24534149742dcd3120bcfc5`。协议SHA：`9f19d8e34a15898816e533e6402d24470a65c2226b5b00b0b89e4748899ba372`。

```bash
python -m scripts.write_transport_diagnosis_report
```
