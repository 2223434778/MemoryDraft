# ASD 实际前缀收益预算验证

2026-10-20。新GSM8K 16题、MATH-500 8题；96机制+192自然EOS计时，288次生成。

同一Qwen3-8B/DFlash、FP32/eager、greedy、非思考、最多512token。近似组B=8/g=0.25/M=2；Utility价格2冻结，不按本批调参。
同题四组同卡交错；GSM8K物理GPU0、MATH-500物理GPU1。所有在线策略成本计入，机制插桩及加载不计正式时延。

| 任务 | 方法 | 自然EOS秒（两重复） | token（两重复） | 自然TPS | 正确 | 截断 | 机制轮数 | 每100token轮数 | regret总支出 | 例外token | 主动缩短块 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | DFlash | 128.894754 | 9678 | 75.085 | 15/16 | 1 | 845 | 17.462 | 0.000 | 0 | 0 |
| gsm8k | ASD | 118.929300 | 9802 | 82.419 | 15/16 | 1 | 772 | 15.752 | 123.831 | 124 | 0 |
| gsm8k | ASD-Run | 119.216686 | 9462 | 79.368 | 15/16 | 1 | 772 | 16.318 | 31.481 | 66 | 0 |
| gsm8k | ASD-Utility | 113.728458 | 9602 | 84.429 | 14/16 | 1 | 735 | 15.309 | 80.204 | 113 | 129 |
| math500 | DFlash | 69.798354 | 6816 | 97.653 | 5/8 | 3 | 459 | 13.468 | 0.000 | 0 | 0 |
| math500 | ASD | 63.230965 | 6832 | 108.048 | 5/8 | 3 | 411 | 12.032 | 60.822 | 75 | 0 |
| math500 | ASD-Run | 56.771632 | 6560 | 115.551 | 5/8 | 3 | 367 | 11.189 | 16.105 | 41 | 0 |
| math500 | ASD-Utility | 56.953907 | 6502 | 114.162 | 5/8 | 3 | 368 | 11.320 | 34.640 | 68 | 58 |

自然EOS路径长度可能改变；TPS也反映不同生成分布，不能当成相同工作量或逐token等价加速。本轮没有忽略EOS的强制续写。

GSM8K按最终数值评分；MATH-500为最后boxed答案的保守规范化精确匹配，不是一般符号等价。失败、截断、零激活均保留，两重复不支持显著性或质量非劣性保证。

| 任务 | 相对原ASD | 自然时延变化 | 自然TPS变化 | 正确→不匹配 | 不匹配→正确 |
|---|---|---:|---:|---|---|
| gsm8k | ASD-Run | +0.24% | -3.70% | 无 | 无 |
| gsm8k | ASD-Utility | -4.37% | +2.44% | gsm8k:21 | 无 |
| math500 | ASD-Run | -10.22% | +6.94% | 无 | 无 |
| math500 | ASD-Utility | -9.93% | +5.66% | 无 | 无 |

按全部四组的终止原因作事后诊断（保留上面的全量主结果，不用子集替代）：

| 任务 | 终止分组 | 题数 | 方法/原ASD | 时延变化 | 自然TPS变化 |
|---|---|---:|---|---:|---:|
| gsm8k | all_eos | 14 | ASD-Run | +0.28% | -2.71% |
| gsm8k | all_eos | 14 | ASD-Utility | -4.35% | +3.73% |
| gsm8k | mixed | 2 | ASD-Run | +0.07% | -7.70% |
| gsm8k | mixed | 2 | ASD-Utility | -4.48% | -2.76% |
| math500 | all_eos | 5 | ASD-Run | -7.26% | +0.03% |
| math500 | all_eos | 5 | ASD-Utility | -9.00% | +0.25% |
| math500 | all_length | 3 | ASD-Run | -13.60% | +15.74% |
| math500 | all_length | 3 | ASD-Utility | -10.99% | +12.34% |

all_eos=四组均完整结束；all_length=四组均达长度上限；mixed=终止原因不一致。分类只作诊断，未作为在线决策或选题依据。
MATH完整结束5题中，Run/Utility的自然TPS变化+0.03%/+0.25%。完整结束与上限截断分组须分别解释，尚不能据此宣称完整长解答的稳定每token加速。

负时延变化表示更快。Utility额外缩短是在它自身轨迹/剩余预算下相对本轮最长可行前缀；不能当作与原ASD整条轨迹逐块配对。

| 任务 | 方法 | 付费块 | 局部额外候选token | 增量≤2块 | 这些块regret | 每regret局部增量 |
|---|---|---:|---:|---:|---:|---:|
| gsm8k | ASD | 110 | 308 | 71 | 81.368 | 2.487 |
| gsm8k | ASD-Run | 64 | 231 | 27 | 5.118 | 7.338 |
| gsm8k | ASD-Utility | 103 | 361 | 49 | 19.258 | 4.501 |
| math500 | ASD | 70 | 200 | 43 | 39.993 | 3.288 |
| math500 | ASD-Run | 39 | 196 | 13 | 2.114 | 12.170 |
| math500 | ASD-Utility | 61 | 272 | 30 | 11.968 | 7.852 |

预算效率与请求效果：

- gsm8k: Utility regret支出变化-35.23%，验证轮变化-4.79%；正确数15→14。regret不是错误概率，减少支出不保证质量提升。
- math500: Utility regret支出变化-43.05%，验证轮变化-10.46%；正确数5→5。regret不是错误概率，减少支出不保证质量提升。
- gsm8k:21人工核对：题目需由当前年龄37减去生育年龄23，正确14；原ASD输出14，Utility输出23，属于真实推理错误，不是解析漏识别。未做反事实重放，不能断言由哪一个例外单独造成。
- MATH-500有3题在全部四组中均截断且不匹配；这些题没有完整解答质量结论。

局部增量=同一已评分块的近似接受长度−严格接受长度；不等于全请求新增输出或节约验证轮。预算到7的位置分数保存在analysis.json，不用最终长度作为在线输入。

| 请求 | 原ASD正确 | Run正确 | Utility正确 | ASD→Run→Utility token | ASD→Run→Utility轮数 |
|---|---|---|---|---|---|
| gsm8k:8 | False | False | False | 512→512→512 | 75→78→78 |
| gsm8k:9 | True | True | True | 343→347→348 | 55→55→52 |
| gsm8k:10 | True | True | True | 255→275→275 | 38→36→35 |
| gsm8k:11 | True | True | True | 268→284→272 | 26→35→28 |
| gsm8k:12 | True | True | True | 428→393→444 | 99→86→90 |
| gsm8k:13 | True | True | True | 458→384→389 | 53→50→44 |
| gsm8k:14 | True | True | True | 339→313→317 | 42→45→41 |
| gsm8k:15 | True | True | True | 277→275→270 | 45→51→50 |
| gsm8k:16 | True | True | True | 172→160→146 | 32→29→24 |
| gsm8k:17 | True | True | True | 315→292→296 | 48→48→45 |
| gsm8k:18 | True | True | True | 173→231→229 | 31→39→38 |
| gsm8k:19 | True | True | True | 360→368→369 | 67→74→61 |
| gsm8k:20 | True | True | True | 420→426→367 | 59→59→45 |
| gsm8k:21 | True | True | False | 229→154→240 | 35→27→44 |
| gsm8k:22 | True | True | True | 169→148→148 | 31→25→25 |
| gsm8k:23 | True | True | True | 183→169→179 | 36→35→35 |
| math500:0 | True | True | True | 401→347→303 | 48→48→44 |
| math500:1 | False | False | False | 512→512→512 | 48→42→46 |
| math500:2 | True | True | True | 386→366→366 | 30→28→28 |
| math500:3 | True | True | True | 320→334→358 | 38→40→42 |
| math500:4 | False | False | False | 512→512→512 | 88→66→67 |
| math500:5 | True | True | True | 387→333→329 | 58→43→44 |
| math500:6 | True | True | True | 386→364→359 | 49→47→44 |
| math500:7 | False | False | False | 512→512→512 | 52→53→53 |

MATH不匹配的原始解析值（需结合完整输出复查）：

- math500:1/DFlash: {"correct": false, "error": "missing boxed final answer", "expected_answer": "p - q", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": null}
- math500:1/ASD: {"correct": false, "error": "missing boxed final answer", "expected_answer": "p - q", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": null}
- math500:1/ASD-Run: {"correct": false, "error": "missing boxed final answer", "expected_answer": "p - q", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": null}
- math500:1/ASD-Utility: {"correct": false, "error": "missing boxed final answer", "expected_answer": "p - q", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": null}
- math500:4/DFlash: {"correct": false, "error": "missing boxed final answer", "expected_answer": "\\text{Evelyn}", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": null}
- math500:4/ASD: {"correct": false, "error": "missing boxed final answer", "expected_answer": "\\text{Evelyn}", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": null}
- math500:4/ASD-Run: {"correct": false, "error": "missing boxed final answer", "expected_answer": "\\text{Evelyn}", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": null}
- math500:4/ASD-Utility: {"correct": false, "error": "missing boxed final answer", "expected_answer": "\\text{Evelyn}", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": null}
- math500:7/DFlash: {"correct": false, "error": "missing boxed final answer", "expected_answer": "90^\\circ", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": null}
- math500:7/ASD: {"correct": false, "error": "missing boxed final answer", "expected_answer": "90^\\circ", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": null}
- math500:7/ASD-Run: {"correct": false, "error": "missing boxed final answer", "expected_answer": "90^\\circ", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": null}
- math500:7/ASD-Utility: {"correct": false, "error": "missing boxed final answer", "expected_answer": "90^\\circ", "metric": "normalized boxed exact match, not general symbolic equivalence", "parsed_answer": null}

执行runtime package SHA `8924a107833ad260fb0dac41a3ebdfb73c4b747dc5fdfcdf11aa4a09cc200e67`。所有既有发布结果在执行前后SHA一致；冻结源码前后校验见execution-verification.json。

重建：`python -m scripts.write_asd_budget_report`。
