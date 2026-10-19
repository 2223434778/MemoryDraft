# Draft内部锚点拼接：小成本开发检查

2026-10-19。16个请求的Target答案均已在之前实验中观察，本轮不是独立holdout。

精确suffix优先；失败后保留DFlash到内部4-token锚点的前缀，拼接历史原文尾部。三种Splice共用候选池、预算和严格Target验证。
Splice-OT沿用原评分，TailOT只将整体与末8位置贡献各加权一半；没有实施单调OT或语义检索。

机制回放96/96输出token、Target输入和合法快照与既有AR严格一致。插桩延迟单独标记，不用于性能结论。

| 方法 | 总轮 | 实际Draft前向 | 对齐接受token | 保留Draft接受 | 历史尾部接受 | 超出原Draft接受 | 内部拼接提议 | 查询秒（插桩） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Suffix-R | 128 | 57 | 0 | 0 | 0 | 0 | 0 | 0.000000 |
| Guided-Edit | 119 | 55 | 224 | 0 | 0 | 0 | 0 | 0.004717 |
| Guided-OT | 119 | 55 | 256 | 0 | 0 | 0 | 0 | 0.063273 |
| Splice-Edit | 119 | 55 | 298 | 110 | 188 | 153 | 9 | 0.060457 |
| Splice-OT | 119 | 55 | 298 | 110 | 188 | 153 | 9 | 0.137803 |
| Splice-TailOT | 118 | 55 | 294 | 110 | 184 | 149 | 9 | 0.137193 |

对齐接受量包含保留Draft前缀，不能全部归为历史收益；超出原Draft接受量也不能直接当作相对其他路线的净增量。

| 请求 | Suffix-R | Guided-Edit | Guided-OT | Splice-Edit | Splice-OT | Splice-TailOT |
|---|---:|---:|---:|---:|---:|---:|
| parcel-queue:0 | 7 | 7 | 7 | 7 | 7 | 7 |
| parcel-queue:1 | 8 | 8 | 8 | 8 | 8 | 8 |
| payment-runbook:0 | 9 | 8 | 8 | 8 | 8 | 8 |
| payment-runbook:1 | 8 | 7 | 7 | 8 | 8 | 8 |
| short-settings:1 | 2 | 2 | 2 | 2 | 2 | 2 |
| confirm-schema-0:0 | 13 | 11 | 11 | 11 | 11 | 10 |
| confirm-schema-0:1 | 13 | 11 | 11 | 11 | 11 | 11 |
| copy | 8 | 8 | 8 | 8 | 8 | 8 |
| edge-router:0 | 6 | 6 | 6 | 6 | 6 | 6 |
| edge-router:1 | 6 | 5 | 5 | 5 | 5 | 5 |
| inventory-query:0 | 7 | 7 | 7 | 6 | 6 | 6 |
| inventory-query:1 | 8 | 8 | 8 | 7 | 7 | 7 |
| image-worker:0 | 8 | 8 | 8 | 8 | 8 | 8 |
| image-worker:1 | 8 | 8 | 8 | 8 | 8 | 8 |
| shipment-contract:0 | 8 | 7 | 7 | 7 | 7 | 7 |
| shipment-contract:1 | 9 | 8 | 8 | 9 | 9 | 9 |

机制门槛：同一个Splice方法至少2请求比两个Guided基线都少轮，且内部拼接在同状态比原Draft连续接受更长。
本轮：通过。

- Splice-Edit：2请求；inventory-query:0、inventory-query:1。
- Splice-OT：2请求；inventory-query:0、inventory-query:1。
- Splice-TailOT：2请求；inventory-query:0、inventory-query:1。

未插桩计时192/192严格一致；每请求两重复，全部在线成本计入。仍为已观察任务开发计时。

| 方法 | 总秒 | 相对Suffix延迟 | 相对Guided-Edit延迟 |
|---|---:|---:|---:|
| Suffix-R | 26.000739 | +0.00% | +3.93% |
| Guided-Edit | 25.018151 | -3.78% | +0.00% |
| Guided-OT | 25.153341 | -3.26% | +0.54% |
| Splice-Edit | 25.285204 | -2.75% | +1.07% |
| Splice-OT | 25.421422 | -2.23% | +1.61% |
| Splice-TailOT | 25.258074 | -2.86% | +0.96% |

| 项目 | Suffix-R | Guided-Edit | Guided-OT | Splice-Edit | Splice-OT | Splice-TailOT |
|---|---:|---:|---:|---:|---:|---:|
| confirm-schema-0 | 4.730943 | 4.276754 | 4.315034 | 4.321389 | 4.403559 | 4.242063 |
| edge-router | 2.571793 | 2.468719 | 2.480865 | 2.478866 | 2.482164 | 2.479265 |
| image-worker | 3.373113 | 3.375659 | 3.384914 | 3.386543 | 3.383931 | 3.381974 |
| infra-copy | 1.361144 | 1.362348 | 1.362332 | 1.367317 | 1.368035 | 1.365388 |
| inventory-query | 3.215116 | 3.221949 | 3.228457 | 3.014994 | 3.021188 | 3.026456 |
| parcel-queue | 3.139054 | 3.144099 | 3.174521 | 3.209377 | 3.226965 | 3.229911 |
| payment-runbook | 3.447081 | 3.228640 | 3.245319 | 3.391409 | 3.394270 | 3.396883 |
| shipment-contract | 3.600339 | 3.375245 | 3.396205 | 3.549554 | 3.574037 | 3.571159 |
| short-settings | 0.562155 | 0.564738 | 0.565695 | 0.565757 | 0.567274 | 0.564974 |

未插桩重复与机制回放的轮数/接受量/候选及真实Draft前向计数：192/192相同。

Splice-Edit与Splice-OT的完整引导调用候选token序列在15/16请求上相同；这描述实际路径，不等价于共同状态下的全候选排序能力。

解释与最小修复：

- inventory-query两请求在位置8/10将同状态原Draft接受13/15提升到32，各少1轮；第二个请求的接续来自当前prompt，不能归为历史独特价值。三种Splice均成功，尚无OT独有收益。
- payment-runbook:1在位置11保留11个Draft token但仅接受3；shipment-contract:1在位置14保留9或15个Draft token但仅接受7。两个位置原Draft也分别接受3/7，未尝试到历史尾部。现有R冷却仍被触发，跳过下一轮位置15/22可接受32token的原Guided机会，各多1轮。这是候选来源反馈策略缺陷，严格输出仍正确。
- TailOT在旧confirm-schema-0:0选到不同候选，当前位置接受28而非32，但后续路径总轮11→10；不能将单次接受最大化直接等同于全请求最优。普通OT/Edit总轮相同；TailOT整体仍慢于原Guided-Edit。
- 保存本轮负结果；最小修复只让保留Draft前缀内拒绝不触发历史检索冷却，然后对上述2负例、2 SQL正例及1旧JSON正例作事后开发复查，不扩大新场景。

范围限制：作者受控任务、既有答案、固定8候选cheap预筛、相同数值阈值、短Draft桥接范围。不能据本轮判断自然工作负载泛化或OT最优能力。
Target逐token验证、用户隔离、时间快照、全部在线成本口径不变，默认suffix和独立infra面试版保留。

执行提交：`0000e5c9f78df0d762d2dba981daafa8a59bd2bb`。package SHA：`44f733dea98734d16c18a3b447207a197fba1e7ec4e14255b36e4e376048cd1a`。

```bash
python -m scripts.write_splice_report
```
