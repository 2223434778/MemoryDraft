# 七任务指标补全：AR参照、实际接受长度和独立奖励评估

2026-10-24。本轮预登记的剩余补测已完成：249个匹配请求、747次新固定生成量计时、120条独立奖励评分。
**继续保留BudgetTree-ASD的数学场景收益；Joint不升级，开放任务保持质量取舍。** 这轮补的是评价证据，没有据新评分改算法、预算、任务或原输出。
原七任务4980计时/249 AR及失败裁判保持，[原结论](ASD_PAPER_FINDINGS.md)继续有效；[补充主表](ASD_PAPER_METRICS_RESULTS.md)包含完整五方法对照。

## AR速度基线

每请求均为严格DFlash→AR→严格DFlash，固定256token、忽略EOS。三个新输出逐token匹配上一轮严格fixed DFlash，Target输入/快照不变。
AR和当前DFlash是同窗直接配对；其他候选复用上一轮计时。原候选TPS除以新AR TPS是跨会话比值，按逐请求新旧DFlash时间比折算的列是估计。两列明确区分，不把估计写成新候选同窗实测。

|任务|新AR token/s|当前DFlash/AR直接倍率|组合/AR跨会话比|组合/AR锚点估计|Joint/AR锚点估计|ASD→组合实际接受长度τ|
|---|---:|---:|---:|---:|---:|---:|
|gsm8k|20.382|3.845×|4.696×|4.709×|4.706×|5.351 → 6.792|
|math500|20.314|4.332×|5.168×|5.175×|5.211×|6.566 → 7.665|
|humaneval|20.148|1.935×|3.663×|3.669×|3.278×|2.442 → 5.044|
|mbpp|20.517|4.106×|5.840×|5.855×|6.050×|7.337 → 8.420|
|mmlu|20.250|3.015×|3.463×|3.466×|3.403×|3.833 → 4.676|
|mtbench|19.435|2.202×|2.892×|2.893×|2.837×|2.710 → 3.827|
|alpaca|20.610|2.106×|2.576×|2.576×|2.613×|2.189 → 3.040|

249个请求中，新严格前后漂移超过3%的有0个，新旧严格跨会话漂移超过3%的有0个；全部保留，未按漂移删样本。逐请求数值见analysis.json/triplets。
原候选之间的增量仍以同轮实测为准：GSM组合较ASD自然时延低9.86%、等128低11.64%，正确集合保持63/64；MATH自然低11.42%、等128低4.70%、正确26→28/32。
固定256在短代码/MMLU含大量EOS后计算；其AR倍率描述计算吞吐，不能替代自然完整请求时延或质量。

## 接受长度定义

统一输出tau_selected（日志中选择的Draft前缀/轮数）和tau_emitted（裁掉末轮没有实际输出的Draft token后/轮数），均不计Target恢复/bonus token。
另列输出token/Target调用、验证节点/输出token与提议量。独立按位置标记Draft与Target来源，3735条旧首重复轨迹全部覆盖且无重叠；避免将最后一轮超出输出上限的部分算成接受收益。
树与线性候选的计算分母不同；更长接受长度需要与节点成本和wall一起看。原生16块与论文DSpark块长不同，跨论文不能仅按τ数值排序。

## 独立奖励模型

[Skywork-Reward-V2-Qwen3-8B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-8B)使用独立偏好训练权重、原始标量reward，逐回答评分，无方法名或A/B位置。
8对独立控制全部通过，反序复算误差0.0；24单元×5方法共120条，去重后87次任务前向，全部可评，没有裁剪长回答。MT使用各方法自身真实双轮。

|方法相对ASD|MT-Bench胜/平/负（8）|MT偏好|Alpaca胜/平/负（16）|Alpaca偏好|
|---|---:|---:|---:|---:|
|DFlash|5/0/3|62.50%|9/2/5|62.50%|
|BudgetTree|5/0/3|62.50%|9/2/5|62.50%|
|BudgetTree-ASD|5/0/3|62.50%|3/2/11|25.00%|
|JointTree-ASD|3/1/4|43.75%|6/1/9|40.62%|

偏好是高reward胜、同分/相同文本平局半分。它不是准确率、官方MT-Bench分数或AlpacaEval胜率；与Target共享Qwen3基础模型，仍可能受训练数据、风格和长度影响。
组合在MT有5胜3负的局部信号，但Alpaca为3胜2平11负；不能据速度宣布开放质量已保持，也不把它当数学收益失效。没有调提示、换模型或删负例来改善分数。

## 对奖励结果做任务级复核

9个有明确可检查属性的单元、45个方法单元、65个属性检查中14项未满足；这是后验诊断，不冒称全体开放任务准确率。

- MT109：ASD把面向方向答成north，其余组答south；组合在此修正事实错误。
- MT130：ASD在约束移除后给出期望O(n+m)哈希方案，其他四组仍排序。奖励模型却把严格DFlash排在ASD之上，证明通过基础控制仍不能替代算法要求检查。
- MT138：只有Joint的第二轮JSON完整闭合，但所有方法都没有将手机名全部大写。语法、指令遵循与偏好需分开。
- Alpaca的被检五个单元满足对应的循环、被动语态、五问、动物分类、十四行结构属性；十四行不等于完整韵律认证。

另复核7个奖励变化较大或有客观对照的任务。bi-annual案例各方法都明确为每年两次，组合−5.125是奖励偏好差异，不能解释成已证实的年限误解或−5.125个百分点准确率。
化学案例中ASD和第一组合都提到额外碳酸盐试剂；Joint给出的直接CO2/水逆过程条件不足。酸性会增加碳酸钙溶解，而补充碳酸根可沉淀回碳酸钙，属于需要区分的过程；这些支持对回答的分项复核，不构成对整篇回答自动判分。参见[OpenStax](https://openstax.org/books/chemistry-2e/pages/15-3-coupled-equilibria)和[Colorado大学演示](https://www.colorado.edu/lab/lecture-demo-manual/general-reactions/s130-chemical-rxns-precipitation-cacl2-na2co3-caco3/s130-prep-notes)。
完整证据及输出SHA在open-contract-audit.json、reward-case-audit.json；旧评分、旧192条失败位置裁判及新120条奖励评分全部保留。

## 执行与复现

GPU源码`4a33ae58c8c35d5f0219ed3366a5150d70d02274`，2259 canonical Git blobs，source SHA`fbde4b6505e814dd8b861eddf66514ae597c5e4300a45b86ac1e26917b137165`，前后及归档复核不变。新root `/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-paper-metrics`，启动2026-10-24 04:20:49 UTC，完成审计2026-10-24T05:06:24.407018+00:00。

|物理卡|正式生成|模型加载秒|冷backend秒|图数/正式新图|峰allocated GiB|compiler cache B|
|---|---:|---:|---:|---:|---:|---:|
|0|375|20.891|3.424|4 / 0|35.094|414726|
|1|372|19.952|3.429|4 / 0|36.055|414726|

四次预热/冷编译另列，cold backend包含在对应预热wall内，不重复相加；本轮普通严格控制不读取构树成本表。全部正式在线成本计入请求wall。
本机奖励评分Python3.13.5/Torch2.6.0+cu124/Transformers4.52.2，RTX3090/BF16/SDPA，校验13.365秒、加载11.284秒、任务前向18.801秒，全进程57.888秒，峰14.332GiB。这些离线成本不混入生成加速比。
模型实际14文件/15152796883B，全部按固定revision和LFS SHA/Git blob校验；准备进度中13文件是计数笔误，manifest和运行校验始终覆盖14文件。成功分块下载阶段1435.318秒，首直连无进度已终止并保留日志，首失败阶段未完整计秒，不估造总下载耗时。权重仅留本地artifacts，不进入GitHub。
本机完整1040 passed/7 skipped/167.98秒，相关28项通过，服务器28项通过。首次完整在后段停滞，无最终XML；核对并终止owned进程后，带逐测试/超时堆栈日志的完整重跑通过，未为此更改生成实现，首次停滞原因未定。

最终结果归档614833B/SHA`523e688cc02d13f50a7de2f1ae44cec2b609b09fdc778b7db277b64f845adfe5`；原始归档SHA`59d2f5dc2abd0cfc43b25a4fce992ccf24890ee575adb65c26452888ea22d636`。发布成员逐字节核验，统计独立重建，旧证据和infra三引用保持。补充原始记录位于[results/asd-paper-metrics](../results/asd-paper-metrics)。

```powershell
.venv\Scripts\python.exe -m scripts.write_asd_paper_metrics_report --input results/asd-paper-metrics
.venv\Scripts\python.exe -m results.asd-paper-metrics.inspect_results
.venv\Scripts\python.exe -m results.asd-paper-metrics.audit_open_contracts
.venv\Scripts\python.exe -m results.asd-paper-metrics.audit_reward_differences
```

其他底座优先使用[文献报告值](ASD_LITERATURE_REFERENCE.md)作横向定位，核心新方法的收益仍由本机同环境ASD/DFlash对照支持。后续优先迁移数学组合，保留工程收益和全部消融；本轮没有启动额外模型配对、训练或新seed。
