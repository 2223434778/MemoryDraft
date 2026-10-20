# ASD 新题与收益来源验证

固定8道新GSM8K、16道新MATH-500；120机制+240完整自然计时+240遵守EOS的128上限前缀计时，共600生成。
完整上限2048；Qwen3-8B+DFlash、FP32/eager、greedy、非思考。同题五组同卡交错，两次计时重复。全部在线成本计入，加载/预热/插桩另计。

Conservative只将原ASD固定gate收紧至0.0625；旧开发状态做近似regret校准，新题不调参、不保证实际预算相等。旧trace缺少未接受EOS位置的限制详见文末审计修订。四个旧策略不改。

gsm8k：Run相对原ASD时延+8.02%、自然TPS-7.65%；输出长度-0.25%、每token轮数+8.37%。相对严格DFlash时延-9.36%、自然TPS+5.50%；相对保守ASD时延-12.97%、自然TPS+3.57%。
math500：Run相对原ASD时延-13.40%、自然TPS-2.10%；输出长度-15.22%、每token轮数+1.73%。相对严格DFlash时延-2.22%、自然TPS-1.29%；相对保守ASD时延-2.96%、自然TPS-4.19%。

自然时延更低可以来自推理路径/长度变化；本报告用严格DFlash、保守门槛及等token前缀检查是否还有执行效率增量。下列全题结果包含错答和截断，质量与终止状态始终单列。

MATH归因重点：Run/Conservative实际总regret为75.527/76.270，例外233/229。Run相对这个保守对照的输出长度变化-7.02%，自然TPS变化-4.19%；共同128token前缀相对原ASD时延+3.65%。因此，完整输出路径变化与验证执行效率需要分别解释，不能仅用全请求自然时延认定实际后缀信息带来独有提速。

旧三题补全曾同时表现为输出减少8.58%、每token轮数减少8.22%、每轮成本约+0.32%。那是三个已观察的长题诊断；新16题的轮密度结果不同，不能把旧局部优势直接推广，也不需要改写旧结果。两轮推理package相同，当前标量/张量决策与重复轨迹审计通过。

本轮判断：没有复现Run/Utility相对原ASD的稳定每token吞吐增量。新MATH上自然时延更低伴随解题路径与质量变化，Run/ASD、Run/保守ASD的跨题时延比区间均包含1；GSM上Run退化，Utility基本持平。保留局部选择器作为实验资产，不把本轮包装成通用后缀提速或稳定质量保证。

后续建议：停止在本批新题扫描门槛、预算和价格。若继续研究，先在旧开发集建立原ASD的门槛/预算速度—质量对照，再用独立任务判断实际后缀信息是否带来额外价值；当前不优先增加更复杂的预测器或扩大同类GPU试验。

## gsm8k 完整答案

| 方法 | 秒（两重复） | token（两重复） | 自然TPS | 主评分正确 | 截断 | 每100token轮 | regret | 例外 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DFlash | 54.298267 | 4244 | 78.161 | 8/8 | 0 | 16.871 | 0.000 | 0 |
| ASD | 45.560562 | 4068 | 89.288 | 8/8 | 0 | 14.553 | 60.268 | 62 |
| ASD-Run | 49.213269 | 4058 | 82.457 | 8/8 | 0 | 15.771 | 16.721 | 26 |
| ASD-Utility | 45.568200 | 4064 | 89.185 | 8/8 | 0 | 14.518 | 36.380 | 45 |
| ASD-Conservative | 56.544874 | 4502 | 79.618 | 8/8 | 0 | 16.482 | 13.063 | 33 |

| 方法/参照 | 时延变化 | TPS变化 | 长度比 | 轮密度比 | 每轮成本比 | 时延比95%区间 |
|---|---:|---:|---:|---:|---:|---|
| ASD-Run/ASD | +8.02% | -7.65% | 0.9975 | 1.0837 | 0.9992 | [1.0180, 1.1675] |
| ASD-Run/ASD-Conservative | -12.97% | +3.57% | 0.9014 | 0.9569 | 1.0091 | [0.7385, 0.9907] |
| ASD-Run/DFlash | -9.36% | +5.50% | 0.9562 | 0.9348 | 1.0140 | [0.8200, 0.9944] |
| ASD-Utility/ASD | +0.02% | -0.12% | 0.9990 | 0.9976 | 1.0036 | [0.9468, 1.0579] |
| ASD-Utility/ASD-Conservative | -19.41% | +12.02% | 0.9027 | 0.8808 | 1.0135 | [0.6448, 0.9506] |
| ASD-Utility/DFlash | -16.08% | +14.10% | 0.9576 | 0.8605 | 1.0184 | [0.7284, 0.9301] |
| ASD-Conservative/ASD | +24.11% | -10.83% | 1.1067 | 1.1326 | 0.9902 | [1.0586, 1.5709] |
| ASD-Conservative/DFlash | +4.14% | +1.86% | 1.0608 | 0.9769 | 1.0049 | [0.9125, 1.1773] |

`时延比=长度比×轮密度比×每轮成本比`严格成立，但这些因素来自不同实现轨迹，不是可独立相加的因果效应。每轮成本含prefill摊销及全部在线开销。

- ASD-Run相对原ASD：逐题时延退化6/8题；主评分损失 `无`，提升 `无`。
- ASD-Utility相对原ASD：逐题时延退化4/8题；主评分损失 `无`，提升 `无`。
- ASD-Conservative相对原ASD：逐题时延退化7/8题；主评分损失 `无`，提升 `无`。

区间按题配对重采样2000次，两次重复先合并为同一题；只是小切片不确定性诊断，不提供质量非劣性保证。

| 五组终止状态 | 题数 | Run/ASD时延变化 | Utility/ASD时延变化 | Conservative/ASD时延变化 |
|---|---:|---:|---:|---:|
| all_eos | 8 | +8.02% | +0.02% | +24.11% |

终止状态分组只作事后诊断，全题表始终是主结果；完整结束不等于答案正确。

128前缀五组全部达到128token的共同集合：8/8题；ID `gsm8k:24, gsm8k:25, gsm8k:26, gsm8k:27, gsm8k:28, gsm8k:29, gsm8k:30, gsm8k:31`。未达到者仍保留在analysis.json的all表中。

| 方法 | 等token前缀秒 | token | TPS | 相对原ASD时延 |
|---|---:|---:|---:|---:|
| DFlash | 30.546760 | 2048 | 67.045 | +15.13% |
| ASD | 26.533284 | 2048 | 77.186 | +0.00% |
| ASD-Run | 27.624276 | 2048 | 74.138 | +4.11% |
| ASD-Utility | 26.484703 | 2048 | 77.328 | -0.18% |
| ASD-Conservative | 30.893342 | 2048 | 66.293 | +16.43% |

在原ASD实际非终止轨迹的288个共同候选块、相同剩余账本上离线重算：

| 选择器 | regret | 例外 | 局部额外proposal | 付费且局部增量≤2的块 | 该部分regret |
|---|---:|---:|---:|---:|---:|
| ASD | 60.268 | 62 | 165 | 34 | 31.482 |
| ASD-Run | 8.251 | 25 | 95 | 12 | 2.418 |
| ASD-Conservative | 10.418 | 30 | 84 | 20 | 7.813 |
- Run与ASD：相同254块、Run多接0块、ASD多接34块。
- Run与ASD-Conservative：相同274块、Run多接5块、ASD-Conservative多接9块。

各块独立借用源轨迹账本，阴影支出相加可能超过每题预算×题数，不能当作替代策略的实际总消耗。局部额外proposal不等于全请求节约token/验证轮；离线反事实不推进模型状态。

逐题完整结果：

| 请求 | 方法 | 秒 | token | 轮 | 正确 | 结束 |
|---|---|---:|---:|---:|---|---|
| gsm8k:24 | DFlash | 6.580029 | 212 | 43 | True | eos |
| gsm8k:24 | ASD | 5.439303 | 194 | 35 | True | eos |
| gsm8k:24 | ASD-Run | 5.306290 | 184 | 34 | True | eos |
| gsm8k:24 | ASD-Utility | 5.312121 | 184 | 34 | True | eos |
| gsm8k:24 | ASD-Conservative | 5.289340 | 184 | 34 | True | eos |
| gsm8k:25 | DFlash | 7.888500 | 323 | 52 | True | eos |
| gsm8k:25 | ASD | 6.760304 | 309 | 44 | True | eos |
| gsm8k:25 | ASD-Run | 7.232267 | 318 | 47 | True | eos |
| gsm8k:25 | ASD-Utility | 7.239099 | 327 | 47 | True | eos |
| gsm8k:25 | ASD-Conservative | 7.194270 | 280 | 47 | True | eos |
| gsm8k:26 | DFlash | 5.675834 | 243 | 37 | True | eos |
| gsm8k:26 | ASD | 5.724434 | 288 | 37 | True | eos |
| gsm8k:26 | ASD-Run | 6.031761 | 260 | 39 | True | eos |
| gsm8k:26 | ASD-Utility | 5.137808 | 261 | 33 | True | eos |
| gsm8k:26 | ASD-Conservative | 6.018285 | 295 | 39 | True | eos |
| gsm8k:27 | DFlash | 6.592388 | 270 | 43 | True | eos |
| gsm8k:27 | ASD | 5.444823 | 287 | 35 | True | eos |
| gsm8k:27 | ASD-Run | 6.204346 | 244 | 40 | True | eos |
| gsm8k:27 | ASD-Utility | 5.903624 | 287 | 38 | True | eos |
| gsm8k:27 | ASD-Conservative | 6.470574 | 277 | 42 | True | eos |
| gsm8k:28 | DFlash | 7.796110 | 278 | 52 | True | eos |
| gsm8k:28 | ASD | 3.895748 | 152 | 25 | True | eos |
| gsm8k:28 | ASD-Run | 5.515482 | 193 | 36 | True | eos |
| gsm8k:28 | ASD-Utility | 4.055599 | 180 | 26 | True | eos |
| gsm8k:28 | ASD-Conservative | 10.492047 | 324 | 70 | True | eos |
| gsm8k:29 | DFlash | 6.600345 | 235 | 44 | True | eos |
| gsm8k:29 | ASD | 7.235898 | 245 | 48 | True | eos |
| gsm8k:29 | ASD-Run | 7.101261 | 245 | 47 | True | eos |
| gsm8k:29 | ASD-Utility | 6.678185 | 253 | 44 | True | eos |
| gsm8k:29 | ASD-Conservative | 7.521510 | 244 | 50 | True | eos |
| gsm8k:30 | DFlash | 7.077075 | 267 | 47 | True | eos |
| gsm8k:30 | ASD | 4.938188 | 257 | 32 | True | eos |
| gsm8k:30 | ASD-Run | 5.681590 | 289 | 37 | True | eos |
| gsm8k:30 | ASD-Utility | 5.544911 | 283 | 36 | True | eos |
| gsm8k:30 | ASD-Conservative | 5.665070 | 289 | 37 | True | eos |
| gsm8k:31 | DFlash | 6.087986 | 294 | 40 | True | eos |
| gsm8k:31 | ASD | 6.121864 | 302 | 40 | True | eos |
| gsm8k:31 | ASD-Run | 6.140271 | 296 | 40 | True | eos |
| gsm8k:31 | ASD-Utility | 5.696853 | 257 | 37 | True | eos |
| gsm8k:31 | ASD-Conservative | 7.893778 | 358 | 52 | True | eos |
## math500 完整答案

| 方法 | 秒（两重复） | token（两重复） | 自然TPS | 主评分正确 | 截断 | 每100token轮 | regret | 例外 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DFlash | 260.130839 | 26336 | 101.241 | 11/16 | 2 | 12.614 | 0.000 | 0 |
| ASD | 293.699300 | 29980 | 102.077 | 8/16 | 3 | 12.368 | 119.549 | 184 |
| ASD-Run | 254.355233 | 25418 | 99.931 | 12/16 | 2 | 12.582 | 75.527 | 233 |
| ASD-Utility | 264.796971 | 26436 | 99.835 | 13/16 | 2 | 12.536 | 101.964 | 236 |
| ASD-Conservative | 262.111812 | 27338 | 104.299 | 10/16 | 2 | 12.144 | 76.270 | 229 |

| 方法/参照 | 时延变化 | TPS变化 | 长度比 | 轮密度比 | 每轮成本比 | 时延比95%区间 |
|---|---:|---:|---:|---:|---:|---|
| ASD-Run/ASD | -13.40% | -2.10% | 0.8478 | 1.0173 | 1.0042 | [0.6668, 1.0492] |
| ASD-Run/ASD-Conservative | -2.96% | -4.19% | 0.9298 | 1.0360 | 1.0074 | [0.7167, 1.2319] |
| ASD-Run/DFlash | -2.22% | -1.29% | 0.9651 | 0.9974 | 1.0157 | [0.8145, 1.1443] |
| ASD-Utility/ASD | -9.84% | -2.20% | 0.8818 | 1.0136 | 1.0088 | [0.7428, 1.0195] |
| ASD-Utility/ASD-Conservative | +1.02% | -4.28% | 0.9670 | 1.0323 | 1.0121 | [0.8251, 1.1764] |
| ASD-Utility/DFlash | +1.79% | -1.39% | 1.0038 | 0.9938 | 1.0204 | [0.7804, 1.2943] |
| ASD-Conservative/ASD | -10.76% | +2.18% | 0.9119 | 0.9819 | 0.9967 | [0.7695, 1.0401] |
| ASD-Conservative/DFlash | +0.76% | +3.02% | 1.0380 | 0.9628 | 1.0082 | [0.8291, 1.3015] |

`时延比=长度比×轮密度比×每轮成本比`严格成立，但这些因素来自不同实现轨迹，不是可独立相加的因果效应。每轮成本含prefill摊销及全部在线开销。

- ASD-Run相对原ASD：逐题时延退化5/16题；主评分损失 `无`，提升 `math500:10, math500:17, math500:18, math500:19`。
- ASD-Utility相对原ASD：逐题时延退化6/16题；主评分损失 `无`，提升 `math500:10, math500:12, math500:17, math500:18, math500:19`。
- ASD-Conservative相对原ASD：逐题时延退化7/16题；主评分损失 `无`，提升 `math500:10, math500:18`。

区间按题配对重采样2000次，两次重复先合并为同一题；只是小切片不确定性诊断，不提供质量非劣性保证。

| 五组终止状态 | 题数 | Run/ASD时延变化 | Utility/ASD时延变化 | Conservative/ASD时延变化 |
|---|---:|---:|---:|---:|
| all_eos | 12 | -8.80% | -17.10% | +0.29% |
| all_length | 1 | -10.97% | -5.77% | -15.03% |
| mixed | 3 | -20.18% | -2.23% | -23.10% |

终止状态分组只作事后诊断，全题表始终是主结果；完整结束不等于答案正确。

128前缀五组全部达到128token的共同集合：16/16题；ID `math500:8, math500:9, math500:10, math500:11, math500:12, math500:13, math500:14, math500:15, math500:16, math500:17, math500:18, math500:19, math500:20, math500:21, math500:22, math500:23`。未达到者仍保留在analysis.json的all表中。

| 方法 | 等token前缀秒 | token | TPS | 相对原ASD时延 |
|---|---:|---:|---:|---:|
| DFlash | 51.485990 | 4096 | 79.556 | +14.67% |
| ASD | 44.898231 | 4096 | 91.229 | +0.00% |
| ASD-Run | 46.536208 | 4096 | 88.017 | +3.65% |
| ASD-Utility | 45.198207 | 4096 | 90.623 | +0.67% |
| ASD-Conservative | 44.819424 | 4096 | 91.389 | -0.18% |

在原ASD实际非终止轨迹的1838个共同候选块、相同剩余账本上离线重算：

| 选择器 | regret | 例外 | 局部额外proposal | 付费且局部增量≤2的块 | 该部分regret |
|---|---:|---:|---:|---:|---:|
| ASD | 119.549 | 184 | 508 | 107 | 74.366 |
| ASD-Run | 23.679 | 94 | 348 | 44 | 5.752 |
| ASD-Conservative | 28.709 | 111 | 324 | 65 | 17.158 |
- Run与ASD：相同1755块、Run多接0块、ASD多接83块。
- Run与ASD-Conservative：相同1799块、Run多接11块、ASD-Conservative多接28块。

各块独立借用源轨迹账本，阴影支出相加可能超过每题预算×题数，不能当作替代策略的实际总消耗。局部额外proposal不等于全请求节约token/验证轮；离线反事实不推进模型状态。

逐题完整结果：

| 请求 | 方法 | 秒 | token | 轮 | 正确 | 结束 |
|---|---|---:|---:|---:|---|---|
| math500:8 | DFlash | 6.598341 | 352 | 43 | True | eos |
| math500:8 | ASD | 5.485878 | 403 | 35 | True | eos |
| math500:8 | ASD-Run | 5.047203 | 373 | 32 | True | eos |
| math500:8 | ASD-Utility | 5.505629 | 403 | 35 | True | eos |
| math500:8 | ASD-Conservative | 6.069165 | 393 | 39 | True | eos |
| math500:9 | DFlash | 41.247559 | 2048 | 257 | False | length |
| math500:9 | ASD | 45.004276 | 1567 | 280 | False | eos |
| math500:9 | ASD-Run | 57.946283 | 2048 | 349 | False | length |
| math500:9 | ASD-Utility | 53.309587 | 2048 | 322 | False | length |
| math500:9 | ASD-Conservative | 36.004881 | 1446 | 224 | False | eos |
| math500:10 | DFlash | 13.392279 | 618 | 89 | True | eos |
| math500:10 | ASD | 28.291721 | 2048 | 177 | False | length |
| math500:10 | ASD-Run | 15.969386 | 870 | 104 | True | eos |
| math500:10 | ASD-Utility | 13.708910 | 628 | 90 | True | eos |
| math500:10 | ASD-Conservative | 11.665427 | 567 | 77 | True | eos |
| math500:11 | DFlash | 42.301470 | 2048 | 260 | False | length |
| math500:11 | ASD | 43.351474 | 2048 | 266 | False | length |
| math500:11 | ASD-Run | 38.596424 | 2048 | 235 | False | length |
| math500:11 | ASD-Utility | 40.851251 | 2048 | 248 | False | length |
| math500:11 | ASD-Conservative | 36.837297 | 2048 | 227 | False | length |
| math500:12 | DFlash | 11.254065 | 609 | 74 | False | eos |
| math500:12 | ASD | 11.512468 | 622 | 75 | False | eos |
| math500:12 | ASD-Run | 10.151430 | 564 | 66 | False | eos |
| math500:12 | ASD-Utility | 10.192301 | 596 | 66 | True | eos |
| math500:12 | ASD-Conservative | 10.742357 | 582 | 70 | False | eos |
| math500:13 | DFlash | 4.620332 | 280 | 28 | True | eos |
| math500:13 | ASD | 5.095488 | 281 | 31 | True | eos |
| math500:13 | ASD-Run | 3.603347 | 222 | 21 | True | eos |
| math500:13 | ASD-Utility | 4.052218 | 230 | 24 | True | eos |
| math500:13 | ASD-Conservative | 4.498612 | 277 | 27 | True | eos |
| math500:14 | DFlash | 11.855603 | 496 | 77 | True | eos |
| math500:14 | ASD | 15.008996 | 657 | 97 | True | eos |
| math500:14 | ASD-Run | 11.675404 | 493 | 75 | True | eos |
| math500:14 | ASD-Utility | 11.532561 | 518 | 74 | True | eos |
| math500:14 | ASD-Conservative | 12.739300 | 570 | 82 | True | eos |
| math500:15 | DFlash | 15.013812 | 836 | 95 | True | eos |
| math500:15 | ASD | 16.114719 | 947 | 101 | True | eos |
| math500:15 | ASD-Run | 15.530143 | 912 | 97 | True | eos |
| math500:15 | ASD-Utility | 15.861592 | 976 | 99 | True | eos |
| math500:15 | ASD-Conservative | 16.298547 | 956 | 102 | True | eos |
| math500:16 | DFlash | 4.627027 | 258 | 31 | True | eos |
| math500:16 | ASD | 3.778327 | 256 | 25 | True | eos |
| math500:16 | ASD-Run | 4.523082 | 257 | 30 | True | eos |
| math500:16 | ASD-Utility | 4.527303 | 257 | 30 | True | eos |
| math500:16 | ASD-Conservative | 4.508628 | 257 | 30 | True | eos |
| math500:17 | DFlash | 27.805174 | 1051 | 177 | True | eos |
| math500:17 | ASD | 22.285896 | 837 | 142 | False | eos |
| math500:17 | ASD-Run | 14.289878 | 619 | 91 | True | eos |
| math500:17 | ASD-Utility | 14.327722 | 620 | 91 | True | eos |
| math500:17 | ASD-Conservative | 16.461315 | 729 | 105 | False | eos |
| math500:18 | DFlash | 12.044967 | 502 | 79 | True | eos |
| math500:18 | ASD | 15.576059 | 670 | 101 | False | eos |
| math500:18 | ASD-Run | 11.886827 | 550 | 77 | True | eos |
| math500:18 | ASD-Utility | 10.212771 | 473 | 66 | True | eos |
| math500:18 | ASD-Conservative | 23.919705 | 1120 | 153 | True | eos |
| math500:19 | DFlash | 15.201793 | 837 | 100 | True | eos |
| math500:19 | ASD | 37.022842 | 2048 | 228 | False | length |
| math500:19 | ASD-Run | 14.136443 | 863 | 92 | True | eos |
| math500:19 | ASD-Utility | 40.838491 | 1892 | 252 | True | eos |
| math500:19 | ASD-Conservative | 37.162933 | 2048 | 229 | False | length |
| math500:20 | DFlash | 3.019046 | 169 | 20 | True | eos |
| math500:20 | ASD | 2.895489 | 170 | 19 | True | eos |
| math500:20 | ASD-Run | 3.046184 | 169 | 20 | True | eos |
| math500:20 | ASD-Utility | 3.047892 | 169 | 20 | True | eos |
| math500:20 | ASD-Conservative | 3.178899 | 166 | 21 | True | eos |
| math500:21 | DFlash | 26.174808 | 1731 | 166 | False | eos |
| math500:21 | ASD | 19.203024 | 1087 | 126 | True | eos |
| math500:21 | ASD-Run | 24.990753 | 1388 | 160 | True | eos |
| math500:21 | ASD-Utility | 15.527478 | 1028 | 101 | True | eos |
| math500:21 | ASD-Conservative | 18.843768 | 1193 | 122 | True | eos |
| math500:22 | DFlash | 17.354515 | 823 | 114 | True | eos |
| math500:22 | ASD | 16.581935 | 848 | 108 | True | eos |
| math500:22 | ASD-Run | 15.705665 | 811 | 102 | True | eos |
| math500:22 | ASD-Utility | 14.493100 | 799 | 94 | True | eos |
| math500:22 | ASD-Conservative | 15.809380 | 801 | 103 | True | eos |
| math500:23 | DFlash | 7.620051 | 510 | 51 | False | eos |
| math500:23 | ASD | 6.490708 | 501 | 43 | False | eos |
| math500:23 | ASD-Run | 7.256781 | 522 | 48 | False | eos |
| math500:23 | ASD-Utility | 6.808164 | 533 | 45 | False | eos |
| math500:23 | ASD-Conservative | 7.371597 | 516 | 49 | False | eos |

## 主评分未通过的输出

主指标沿用保守boxed规范化精确匹配，不宣称一般符号等价；数值题使用冻结GSM解析器。以下全部保留，不将格式假失败视为真实推理错误：

- math500:9/DFlash: {"correct": false, "error": "missing boxed final answer", "expected_answer": "4", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": null}
- math500:9/ASD: {"correct": false, "error": null, "expected_answer": "4", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": "3"}
- math500:9/ASD-Run: {"correct": false, "error": "missing boxed final answer", "expected_answer": "4", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": null}
- math500:9/ASD-Utility: {"correct": false, "error": "missing boxed final answer", "expected_answer": "4", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": null}
- math500:9/ASD-Conservative: {"correct": false, "error": null, "expected_answer": "4", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": "2"}
- math500:10/ASD: {"correct": false, "error": "missing boxed final answer", "expected_answer": "2220", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": null}
- math500:11/DFlash: {"correct": false, "error": "missing boxed final answer", "expected_answer": "\\frac{3}{56}", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": null}
- math500:11/ASD: {"correct": false, "error": "missing boxed final answer", "expected_answer": "\\frac{3}{56}", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": null}
- math500:11/ASD-Run: {"correct": false, "error": "missing boxed final answer", "expected_answer": "\\frac{3}{56}", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": null}
- math500:11/ASD-Utility: {"correct": false, "error": "missing boxed final answer", "expected_answer": "\\frac{3}{56}", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": null}
- math500:11/ASD-Conservative: {"correct": false, "error": "missing boxed final answer", "expected_answer": "\\frac{3}{56}", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": null}
- math500:12/DFlash: {"correct": false, "error": null, "expected_answer": "284", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": "234"}
- math500:12/ASD: {"correct": false, "error": null, "expected_answer": "284", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": "234"}
- math500:12/ASD-Run: {"correct": false, "error": null, "expected_answer": "284", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": "234"}
- math500:12/ASD-Conservative: {"correct": false, "error": null, "expected_answer": "284", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": "234"}
- math500:17/ASD: {"correct": false, "error": null, "expected_answer": "\\pi", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": "-\\pi"}
- math500:17/ASD-Conservative: {"correct": false, "error": null, "expected_answer": "\\pi", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": "-\\pi"}
- math500:18/ASD: {"correct": false, "error": null, "expected_answer": "28", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": "124"}
- math500:19/ASD: {"correct": false, "error": "missing boxed final answer", "expected_answer": "3", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": null}
- math500:19/ASD-Conservative: {"correct": false, "error": "missing boxed final answer", "expected_answer": "3", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": null}
- math500:21/DFlash: {"correct": false, "error": null, "expected_answer": "13535", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": "13536"}
- math500:23/DFlash: {"correct": false, "error": null, "expected_answer": "x=5", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": "5"}
- math500:23/ASD: {"correct": false, "error": null, "expected_answer": "x=5", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": "5"}
- math500:23/ASD-Run: {"correct": false, "error": null, "expected_answer": "x=5", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": "5"}
- math500:23/ASD-Utility: {"correct": false, "error": null, "expected_answer": "x=5", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": "5"}
- math500:23/ASD-Conservative: {"correct": false, "error": null, "expected_answer": "x=5", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": "5"}

## 单列答案等价审计

原始rows和主评分均未改动。人工核对新MATH主评分失败答案，math500:23五组的`5`与参考`x=5`等价；其他数值错误、负的相位参数和未完成输出不提升评分。详情及原始机制SHA见quality-audit.json。

| 方法 | MATH保守主评分 | MATH答案等价审计后 |
|---|---:|---:|
| DFlash | 11/16 | 12/16 |
| ASD | 8/16 | 9/16 |
| ASD-Run | 12/16 | 13/16 |
| ASD-Utility | 13/16 | 14/16 |
| ASD-Conservative | 10/16 | 11/16 |

旧gsm8k:21的Utility真实错误、旧math500:7的Run退化继续保留；本轮不修正旧结论。短前缀样本是完整新题的补充诊断，不是新的独立问题。

离线审计修订：原trace没有保存未接受proposal的EOS位置。本轮gsm8k:30/Utility有一块拒绝EOS后继续生成，EOS裁剪可从记录的可行上限反推；审计兼容此情况，不更改原推理/账本/计时。
同状态反事实限定为原ASD的非终止块：Run与Conservative的门槛都更严，可选前缀必在原ASD实际可行上限内，不会跨越未知EOS。原协议拟报告五条源轨迹，此处因日志可识别性收窄为这一条，其他源不作无界后缀收益统计。
旧开发Run日志同样缺少未接受EOS标记，因此原门槛校准是近似启发式，不能视作精确EOS裁剪后的支出匹配。保持预先冻结gate0.0625，正式比较以新题实际账本为准。

冻结执行commit `38a87cef2d37192719c2fefe282eb574d19ca09e`；runtime package `8924a107833ad260fb0dac41a3ebdfb73c4b747dc5fdfcdf11aa4a09cc200e67`。源码与旧发布证据执行前后SHA一致。
重建：`python -m scripts.write_asd_attribution_report`。配置/数据/协议/源码SHA、逐题原始输出、账本、分片命令和服务器测试见results/asd-attribution。
