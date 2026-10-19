# 拼接前缀拒绝：冷却修复检查

2026-10-19。五个已经观察过的请求，事后定位与修复验证，不是新holdout。首轮负结果完整保留于OT_SPLICE_RESULTS.md。

最小修复：尚在保留Draft前缀内拒绝时不触发历史检索冷却；首个历史token或之后拒绝仍冷却。Target验证、候选池/阈值及其他方法规则不变。

30/30机制回放、60/60未插桩交错计时与原AR逐token/prompt/snapshot严格一致；两阶段package/runner相同，全部旧证据包括首轮拼接SHA未变。

| 请求 | 方法 | 首轮总轮 | 修复后总轮 |
|---|---|---:|---:|
| payment-runbook:1 | Splice-Edit | 8 | 8 |
| payment-runbook:1 | Splice-OT | 8 | 8 |
| payment-runbook:1 | Splice-TailOT | 8 | 7 |
| confirm-schema-0:0 | Splice-Edit | 11 | 11 |
| confirm-schema-0:0 | Splice-OT | 11 | 11 |
| confirm-schema-0:0 | Splice-TailOT | 10 | 10 |
| inventory-query:0 | Splice-Edit | 6 | 6 |
| inventory-query:0 | Splice-OT | 6 | 6 |
| inventory-query:0 | Splice-TailOT | 6 | 6 |
| inventory-query:1 | Splice-Edit | 7 | 7 |
| inventory-query:1 | Splice-OT | 7 | 7 |
| inventory-query:1 | Splice-TailOT | 7 | 7 |
| shipment-contract:1 | Splice-Edit | 9 | 8 |
| shipment-contract:1 | Splice-OT | 9 | 9 |
| shipment-contract:1 | Splice-TailOT | 9 | 9 |

两个退化请求下一轮32token机会恢复：

- Splice-Edit：payment-runbook:1=False；shipment-contract:1=True。
- Splice-OT：payment-runbook:1=False；shipment-contract:1=False。
- Splice-TailOT：payment-runbook:1=True；shipment-contract:1=False。

三个正例各方法总轮保持首轮计数：True。

保留Draft的可靠长度仍是剩余问题：以下事后对照不参与线上选择。

| 请求/位置 | 方法 | 保留Draft长度 | 原Draft连续接受 | 拼接连续接受 |
|---|---|---:|---:|---:|
| payment-runbook:1/15 | Splice-Edit | 14 | 7 | 7 |
| payment-runbook:1/15 | Splice-OT | 14 | 7 | 7 |
| payment-runbook:1/15 | Splice-TailOT | 7 | 7 | 32 |
| shipment-contract:1/22 | Splice-Edit | 6 | 6 | 32 |
| shipment-contract:1/22 | Splice-OT | 8 | 6 | 6 |
| shipment-contract:1/22 | Splice-TailOT | 8 | 6 | 6 |

runbook位置15普通Edit/OT保留14个Draft token，原Draft仅前7个正确；TailOT保留7个并恢复32。JSON位置22则Edit恢复32，两个OT保留8个Draft token但仅接受6，依然没到历史尾部。某个内部锚点能匹配，并不说明它前面的Draft都正确；OT相似度尚未解决这项风险。

以下只比较本批五请求、两重复的配对总时延，不将首轮绝对时间跨运行相减，也不外推修复后的16请求总性能。

| 方法 | 秒 | 相对同批Guided-Edit延迟 |
|---|---:|---:|
| Suffix-R | 8.926047 | +5.23% |
| Guided-Edit | 8.482442 | +0.00% |
| Guided-OT | 8.522107 | +0.47% |
| Splice-Edit | 8.540418 | +0.68% |
| Splice-OT | 8.780651 | +3.52% |
| Splice-TailOT | 8.474506 | -0.09% |

| 请求 | Suffix-R | Guided-Edit | Guided-OT | Splice-Edit | Splice-OT | Splice-TailOT |
|---|---:|---:|---:|---:|---:|---:|
| payment-runbook:1 | 1.621153 | 1.512183 | 1.519258 | 1.702480 | 1.708275 | 1.559929 |
| confirm-schema-0:0 | 2.307921 | 2.079389 | 2.102751 | 2.107764 | 2.141076 | 1.981989 |
| inventory-query:0 | 1.494216 | 1.497454 | 1.494343 | 1.381775 | 1.388382 | 1.389449 |
| inventory-query:1 | 1.652484 | 1.659079 | 1.662343 | 1.570164 | 1.572395 | 1.575211 |
| shipment-contract:1 | 1.850273 | 1.734337 | 1.743413 | 1.778234 | 1.970521 | 1.967928 |

60个未插桩观察与机制回放计数一致：True。

解释：允许下一轮继续检索修复了反馈策略，但未保证选对接续；冷却不是全部原因。若仍未恢复32token机会，需离线分析同状态候选与查询预算，不能继续宣称负例已解决。SQL桥接收益仍需与同池Splice-Edit比较；这五个请求由已知故障/正例选取，不能证明OT的独立泛化优势。默认suffix与独立infra面试版保留。

执行提交：`c98bb83e3e50a0ba1c01268317b074a2fcfcbbd4`；package SHA：`89063c6e3007034cb72823d5a42494f26258baf35b7cbf9beaa74dbd0ee2513c`。

```bash
python -m scripts.write_splice_cooldown_report
```
