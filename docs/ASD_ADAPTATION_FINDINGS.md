# 阶段5：等预算Draft适配的实际结果

本轮完成真实训练及18道开发题的540次配对计时。**ASD实际前缀适配在预指定GSM8K域出现值得继续验证的正信号**：配合原ASD，自然时延比原权重低4.46%，比等预算普通适配低3.43%；三模型共同达到128 token的7题，分别省3.34%和4.96%。最终答案均8/8，没有新增截断。普通适配在该共同128范围反而比原ASD慢2.18%。

结论仍限于一个初始化seed、32训练题/8内部校准和8个已观察开发评估题。不能宣称额外增益已获独立稳定确认，不能由最终答案正确推断全部推理步骤正确；默认服务和面试版保持原权重。下一轮优先检验等生成位置训练对照及独立GSM题，不先增加复杂模块。

## 比较的三个模型

- base：原Qwen3-8B/DFlash b16权重。
- ordinary：在原严格DFlash实际轮4/12到达的Target greedy前缀上，重新生成teacher后续，做普通领域适配。
- asd-prefix：在原ASD同轮到达的实际前缀上，重新生成teacher后续；不复制ASD原后续当标签，不用旧错误上下文的Target logits/KV。

两组均为q_proj/v_proj LoRA rank8/alpha16/dropout0，532480可训练参数、128个AdamW更新/累积2，共256样本反向，lr1e-4、clip1、相同初始adapter/shuffle键。Target及Draft base冻结，embedding/head共享且冻结。仅训练数据分布不同；前缀长度不同，因此等更新不等于完全相同FLOPs。

官方GSM8K **train**固定commit3101c7d5072418e28b9008a6636bde82a006892c，按预定SHA排序选32训练/8内部校准，排除与完整官方test的规范化精确文本重合。保存的问题子集没有参考答案。评估是原suite development：GSM8题，其余五类各2，共18；训练/内部校准/评估ID与问题隔离。validation48不用于训练或选checkpoint，reserve96未生成。精确排重不保证语义近重复或上游预训练隔离。

80条strict/ASD源生成，其中39达到256采集上限、41 EOS；这些截断源不是自然答案质量证据。所有40题都有两个固定源状态，最终80对/160个唯一teacher前缀，2400监督token，无缺失或按答案筛样。训练各64状态/32题；内部混合校准为32状态/8题。features严格截止anchor之前，后续15位置全mask；新teacher标签仅进入位置加权CE，权重exp(-(k-1)/7)。

## 训练确实生效，但离线指标不等于线上性能

|训练组|初始共同校准loss|32步|64步|96步|128步/选中|训练段秒|
|---|---:|---:|---:|---:|---:|---:|
|ordinary|1.643746|1.642513|1.634464|1.617016|1.602276|11.714|
|asd-prefix|1.643746|1.643092|1.636581|1.622042|1.604261|11.648|

两组按相同内部校准池CE选中末128，选中与final权重SHA相同；没有用18题结果选epoch或继续训练。首梯度norm分别0.24556/0.13030，有限且非零；Target无梯度，Target及Draft base parameter pointer/version守卫未变。真实tiny Target另逐位核对权重未变。

普通适配CE下降2.52%、ASD源下降2.40%，但内部校准的严格最长接受长度两组都从3.75降至3.59375。不能仅凭CE较低判定普通组更适合实际推理，也不能声称离线接受代理已经提高。校准使用未合并FP32/eager，线上测的是合并dense/RMSNorm；这两个指标/配置的差异尚未被单独因果分解。

训练源64对中，ASD前缀此前全部发生过近似接受，64对prefix SHA均不同，63对teacher后续不同；**只有3对生成长度边界相同**。内部校准16对也全部不同且有此前例外。故该控制包含实际内容与解码进度分布的共同变化，不能把收益纯归因于纠正“错误token”，还需同生成位置的严格源对照。

## GSM8K：自然结果和固定工作量

模型推理前将LoRA合并到原dense q/v，在线结构仍是同一16块。三模型都评strict D/原ASD，Target输入、FP32/highest/TF32关闭、eager attention、RMSNorm profile和ASD B8/g.25/M2保持一致。

|模型/策略|自然时延/同策略base|自然TPS/base|更快题|正确|输出总token|Target验证调用|
|---|---:|---:|---:|---:|---:|---:|
|base/D|1|1|基线|8/8|1588|291|
|ordinary/D|1.007820|0.992544|4/8|8/8|1588|293|
|asd-prefix/D|0.987270|1.013415|7/8|8/8|1588|288|
|base/ASD|1|1|基线|8/8|1648|276|
|ordinary/ASD|0.991675|1.016000|5/8|8/8|1654|273|
|asd-prefix/ASD|0.955426|1.046117|8/8|8/8|1634|262|

时延/TPS为题级重复均值相对同题同策略base再题等权；调用/token是每题一条稳定轨迹之和，不乘重复数。验证调用不含共同的8次prefill，但全部prefill成本在wall内。所有18题的全部严格输出与旧完整DFlash/RMSNorm逐token/text/EOS相同；base ASD也重现旧完整输出，重复轨迹/账本稳定。

ASD前缀适配的8个自然时延点估计均更低，但1150/885/924没有减轮，其中有接近计时噪声的差异，不把8/8写成8个实质改善。主要减轮来自109、1196、350、374、52。1196输出197→211反而更长，调用37→32、时延比0.872149；52输出347→326，收益包含路径缩短，必须保留两类例子。

|GSM三模型均实际达到128的7题|时延/base|时延/ordinary|
|---|---:|---:|
|ordinary/D|1.016172|1|
|asd-prefix/D|0.994893|0.979483|
|ordinary/ASD|1.021797|1|
|asd-prefix/ASD|0.966559|0.950393|

短题1150保持自然EOS，没有强制续写；所有8题128/EOS结果单独在主表中，不能与7题共同完整范围混用。共同范围的实际计算结果支持“收益不全由输出变短造成”，但不能把自然4.46%与不同集合的3.34%直接相减作因果分解。

GSM自然ASD三组近似接受token数量都为54，regret总支出base/ordinary/asd-prefix为60.24634/58.59283/58.66578，没有靠提高预算或更多例外数获得提速。ASD前缀组验证调用少14次、总输出少14 token（0.85%）；该计数是实际运行效果，不证明唯一的训练因果机制。

## 迁移和不确定性

|家族|asd-prefix+ASD自然时延/base|ordinary+ASD时延/base|新增正确→错误|
|---|---:|---:|---:|
|GSM8K（8题）|0.955426|0.991675|0|
|MATH-500（2题）|0.976204|0.975913|0|
|HumanEval（2题）|1.000768|0.999219|0|
|MBPP（2题）|0.999881|0.998881|0|
|SQuAD（2题）|0.998045|1.002086|0|
|JSON（2题）|0.999186|1.000744|0|

MATH两组收益近似，不能算ASD源独有增量。其余类别基本持平，亚百分点波动不作新性能结论。全18题ASD前缀组自然ASD时延低2.27%、strict低0.57%；GSM占8/18请求，这不是六类等权或普遍任务分布。六臂全部自然EOS，无4096截断；strict均16/18、ASD均17/18，没有新增由对变错，也没有本轮新增由错变对。保留旧SQuAD失败及strict HumanEval失败，近似策略的原有优势不能归给新训练。

事后按请求bootstrap（先平均计时重复，再对8/7题重采样10000次）仅描述已有开发数据：ASD前缀组自然ASD/base的95%区间[0.9253,0.9828]；相对ordinary为[0.9231,1.0040]。共同128/base为[0.9205,1.0071]、相对ordinary为[0.8831,1.0077]，后几项跨1。严格D自然/base也跨1。故“相对原ASD出现窄域正信号”有支持，“ASD源额外增益已经稳定证实”还不成立；bootstrap不消除开发集已观察、单seed及源进度混杂。它没有改变checkpoint或门槛。

## 成本、完整性与未执行项

两卡收集计时段211.516/201.260秒，含模型准备与teacher采集；训练段11.714/11.648秒不含此前加载/特征读取，不能只报12秒称全流程成本。特征1719637760B=1.60154GiB/160文件；每个adapter20张量、532480参数、2132408B。原始hidden/weights只保留服务器，不进Git。GPU allocated收集峰值34.6596GiB，训练34.5305/34.5346GiB，计时评估峰值35.1619GiB。

评估每卡3图/backend3.18786/3.19195秒、cache398054B，正式新增编译0。空cache首base D32 wall4.74335/4.76201秒、TTFT3.44322/3.45287秒，图0→3的编译实际在请求内。实验为交错固定模型所做的离线dense合并总10.8298/10.7387秒单列，不声称线上每请求换权重免费，也不将其从generate计时中事后扣除。

首次PEFT前置2 failed/84 passed：旧系统bitsandbytes缺cuda124库，未开始GPU worker；只对已检查原生FP32 Linear临时跳过未使用量化后端探测并恢复函数，未改环境或精度。失败ZIP及源9bb81be完整保留。修复后真实tiny DFlash/PEFT梯度及tiny合并测试通过。

附加的“完整模型未合并PEFT对dense逐logits/校准重放”没有运行：准备读取PEFT API签名时，自动权限审批连续两次超过截止时间；不是安全拒绝。按工具规则重试一次后停止该可选操作，不把它算作通过。已完成的是tiny数值合并核对、完整选中checkpoint SHA/shape/finite检查和540次实际dense部署评估。这不影响已测部署性能，但限制对离线代理变化原因的定位；见optional-verification.json。

## 来源、审计及复现

GPU执行来源`4d7ac6eca2ab9970e9f71314fc77b9b9de79e140`/1584 Git blobs，source SHA `4b5109f72f1da81084e59b14533594e1ac26c8b684a47156e4774a7c0ead1e84`。本地完整canonical ZIP SHA `c14712655b4b70b88136922ddfa2d3c8c61977663f2653e0d83b8a6382626deb`；实际以旧已校验ZIP+1233414B delta（SHA `11f264bbce26c4390bb9721839a64ef0573e52821e1c7bb74f7da220dfcde091`）在新asd-adaptation-v2重建目标文件逐SHA核验，不声称传输了新完整ZIP。

PID697777于2026-10-22 07:31:56 UTC（北京时间15:31:56）启动；07:48:12完成后处理/末核查，两卡19MiB/0%。成功ZIP889588B/SHA `b4178f0a920155077e00ad2a1633ccf131e9192d21a9dbd399ca9e45fce7f837`；失败82100B/SHA `1032258be91b46cc780f2fea789ed7c4b2b99577bf28d7cb26ea6de935fdc015`。

统计修正版单独冻结`f8e68035f6ae3f01e8e0fa531260a248e03fe7f8`：原报告共同128集合对base/ordinary未统一要求ASD-prefix也达到128，修正符合原协议。GPU执行目录与生成源码未改，没有重跑有效计时；本批修正前后数值差异恰为0，但反例测试防止后续集合不一致。服务器执行修正版精确SHA `20a7d56864df54607429e38e8ac33b766b839a08f9ef40a8dd226878a39f8440`，Python3.11服务器和3.13本地主统计完全一致（差异0）。

本地完整811 passed/7 skipped/114.38s；兼容修复后相关20 passed/2 skipped/8.88s，统计修正12 passed/6.55s；服务器87 passed/10.03s。后审计两份1584源清单、11执行冻结文件/2统计文件、1073旧结果SHA、12批478历史发布文件、80源/160状态和540轨迹前沿通过；两个既有CRLF例外单列，main/面试branch/tag均仍27e3619。原Target、默认runtime和旧项目保留。

主要文件：`memorydraft/evaluation/adaptation.py`；`scripts/prepare_asd_adaptation.py`、`run_asd_adaptation_{collect,train,eval,pipeline}.py`、`write_asd_adaptation_report.py`；`tests/test_adaptation.py`、`test_asd_adaptation_report.py`；协议与数据在docs/ASD_ADAPTATION_PROTOCOL.md和data/research/asd-adaptation。完整主表ASD_ADAPTATION_RESULTS.md、原始gzip/曲线/审计在results/asd-adaptation。

服务器入口（已运行目录不能直接覆盖，复跑需新部署目录）：

```bash
cd /media/a6000/D/ZhaoBinHe/MemoryDraft/asd-adaptation-v2
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_adaptation_pipeline
```

本机纯报告重建（原ZIP/Git历史用于审计）：

```powershell
.venv\Scripts\python.exe -m scripts.write_asd_adaptation_report
.venv\Scripts\python.exe -m results.asd-adaptation.inspect_results
.venv\Scripts\python.exe -m results.asd-adaptation.audit_statistics
.venv\Scripts\python.exe -m results.asd-adaptation.audit_postrun
```

方法依据：[DFlash](https://arxiv.org/html/2602.06036v1)的锚点/位置CE，[AdaFlash](https://arxiv.org/html/2607.19223v3)及[Draft-OPD](https://arxiv.org/abs/2605.29343)的部署轨迹训练思想。本轮是有限离线LoRA/新teacher硬标签对照，没有完整KL或在线刷新，不主张这些思想原创。
