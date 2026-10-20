# ASD 在 GSM8K / MBPP 上的组合小实验

2026-10-20。GSM8K test前8题、MBPP官方test编号11–18；64机制+128自然EOS+128固定128token，共320次生成。

Qwen3-8B/DFlash、FP32/eager、greedy、非思考，ASD B=8/g=0.25/M=2。任务来自ASD使用的benchmark，模型/起草器/提示不等同其原论文复现。

每题独立空历史，拼接只检索当前题目文本/公开测试；参考答案及其他生成不入索引。DFlash strict为本批参考，没有额外生成AR。

| 任务 | 方法 | 自然EOS秒（两重复） | 输出token（两重复） | 正确（8题） | strict一致（8题） | 机制轮数 | 机制例外token | 截断（8题） | 固定128token TPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | DFlash | 66.675068 | 4500 | 8/8 | 8/8 | 436 | 0 | 0 | 62.538 |
| gsm8k | DFlash-ASD | 54.463856 | 3972 | 8/8 | 0/8 | 353 | 53 | 0 | 69.168 |
| gsm8k | Splice | 67.897938 | 4500 | 8/8 | 8/8 | 437 | 0 | 0 | 61.310 |
| gsm8k | Splice-ASD | 55.464486 | 3972 | 8/8 | 0/8 | 354 | 53 | 0 | 67.800 |
| mbpp | DFlash | 10.718594 | 578 | 5/8 | 8/8 | 55 | 0 | 0 | 26.266 |
| mbpp | DFlash-ASD | 10.466526 | 568 | 5/8 | 5/8 | 53 | 5 | 0 | 66.632 |
| mbpp | Splice | 10.861796 | 578 | 5/8 | 8/8 | 55 | 0 | 0 | 25.847 |
| mbpp | Splice-ASD | 10.593452 | 568 | 5/8 | 5/8 | 53 | 5 | 0 | 65.600 |

正确性按自然EOS的最终数值/MBPP公开测试评分；不混合两类质量指标。固定128token忽略EOS，仅报告等工作量吞吐，尤其短代码可能越过正常结束，不作任务质量证据。全部在线生成成本计入，评分/加载/插桩时间不计入正式生成时延。

| 任务 | 比较 | 自然EOS时延变化 | 固定工作量TPS变化 |
|---|---|---:|---:|
| gsm8k | DFlash-ASD/DFlash | -18.31% | +10.60% |
| gsm8k | Splice-ASD/Splice | -18.31% | +10.58% |
| gsm8k | Splice-ASD/DFlash-ASD | +1.84% | -1.98% |
| gsm8k | Splice/DFlash | +1.83% | -1.96% |
| mbpp | DFlash-ASD/DFlash | -2.35% | +153.68% |
| mbpp | Splice-ASD/Splice | -2.47% | +153.80% |
| mbpp | Splice-ASD/DFlash-ASD | +1.21% | -1.55% |
| mbpp | Splice/DFlash | +1.34% | -1.59% |

自然输出长度改变时，总时延变化包含路径/长度变化，不可全归为每token加速；固定吞吐另看。小样本两重复不做显著性或质量非劣性保证。

MBPP固定工作量中，strict有71.8%、ASD有72.3%的token在自然EOS之后；64/64固定行的自然输出前缀与相应自然EOS方法一致。该口径下的大吞吐增幅覆盖了大量强制续写，不能解读为正常代码请求获得同等加速；正常请求以自然EOS时延为准。

| 方法 | 机制R轮 | 内部拼接提议 | 机制历史段提交token | 含例外的块 | Draft桥段例外 | 历史段例外 | 跨桥释放历史块 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DFlash | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DFlash-ASD | 0 | 0 | 0 | 53 | 0 | 0 | 0 |
| Splice | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| Splice-ASD | 1 | 0 | 0 | 53 | 0 | 0 | 0 |

本批内部拼接提议为0，没有验证出拼接与ASD的互补收益。若出现加速，应归于ASD验证策略；普通精确R的少量触发另计在R轮，不能混称内部拼接。

这里的历史段指拼接结构中的复制段，资源本轮仅当前输入；近似接受后的精确匹配基于新前缀，不保证strict轨迹。

GSM8K评分器修正：原提示中的占位符使部分输出包含 `<answer>…</answer>`，初版解析漏识别标签及boxed货币数值。离线修正仅接受已明确输出的数字，未改输入、生成、阈值或标签；原始评分仍在rows中，修正明细在quality-v2.json。

本轮共48条机制/重复记录的评分结构得到修正（重复不增加独立题数）。最终报告使用修正后的评分，grader SHA `56dab84b122f38fc1d6ce2eeb49b21c3a50342d258bc9d494349580b30d1bb94`。

| 请求 | strict正确 | ASD正确 | strict token→ASD token | strict轮→ASD轮 | ASD例外 |
|---|---|---|---|---|---:|
| gsm8k:0 | True | True | 241→233 | 49→48 | 6 |
| gsm8k:1 | True | True | 159→166 | 33→33 | 8 |
| gsm8k:2 | True | True | 400→296 | 75→46 | 6 |
| gsm8k:3 | True | True | 138→129 | 32→25 | 5 |
| gsm8k:4 | True | True | 272→307 | 58→53 | 8 |
| gsm8k:5 | True | True | 321→281 | 67→60 | 5 |
| gsm8k:6 | True | True | 311→208 | 44→28 | 5 |
| gsm8k:7 | True | True | 408→366 | 78→60 | 10 |
| mbpp:11 | False | False | 35→35 | 12→11 | 1 |
| mbpp:12 | True | True | 18→18 | 4→4 | 0 |
| mbpp:13 | False | False | 37→37 | 6→6 | 0 |
| mbpp:14 | True | True | 23→23 | 5→5 | 0 |
| mbpp:15 | False | False | 88→88 | 10→10 | 0 |
| mbpp:16 | True | True | 47→42 | 10→10 | 2 |
| mbpp:17 | True | True | 13→13 | 2→2 | 0 |
| mbpp:18 | True | True | 28→28 | 6→5 | 2 |

执行runtime package SHA `4d92d736d78a5c1962022922443853282ad661edddc9582d91b453b0a8a19e0e`，来源提交/数据SHA/旧证据保护见provenance与source-manifest。

重建：`python -m scripts.write_asd_bench_report`。
