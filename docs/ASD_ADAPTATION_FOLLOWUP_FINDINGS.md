# 阶段5后续：同生成位置控制与独立GSM结果

本轮完成一次新增严格前缀训练及24题的960次正式计时、24次完整AR参考。**本阶段接纳位置匹配严格前缀适配在数学ASD场景取得的实际收益。** 在Qwen3-8B/DFlash b16及固定ASD配置下，新16题的等长128 token时延降低3.23%，自然时延降低7.32%，未新增答案错误；自然值的路径长度因素在下文单列。这是已实现并实测的场景优化成果，用户于2026-10-22明确不要求补不同seed作为接纳条件。

旧ASD前缀方案的开发收益也已重现，作为局部成果保留。原“ASD内容增量”门槛比较ASD前缀与同位置严格前缀，其未通过结果继续如实记录。位置匹配方案自身相对原ASD取得的收益已经获得本阶段接纳。

## 改了什么

保留上一轮ordinary与asd-prefix适配器，逐SHA核对、不重训。新增ordinary-position：对每个ASD轮4/12的已确认前缀，用同题严格轨迹截到完全相同的generated_before，保留同prompt、同上下文长度和anchor位置；内容可以不同。全部80个位置均在严格源EOS前到达，无删样；64训练状态/32题，16额外内部校准状态只采集、不用于选checkpoint。

原ordinary的64训练采样位置均值35.734 token（9–76），ASD组均值47.141（12–111）。新对照逐状态匹配后者，不只匹配平均长度。80对prefix内容全部不同。新严格截点不必是原DFlash真实轮边界，因此这是离线训练前缀对照，不是可直接调用的线上动作。

新严格状态重新运行冻结Target，提取anchor前hidden，以真实anchor+15mask预测新teacher后续；不复制旧标签或错误上下文KV。新臂与原ASD臂保持相同初始化seed20261022、训练问题/配对shuffle键、q/v rank8/alpha16、AdamW1e-4/.01、累积2、128更新、位置CE和原32状态混合内部校准池。四个训练checkpoint按同一内部CE规则选中末128；本轮没有按新题改超参或继续训练。

内部初始CE精确复现1.643746；新臂降到1.592008、严格接受长度3.75→4.0625。原ordinary/asd-prefix末CE为1.602276/1.604261、接受长度均3.59375。这个离线改善没有转化为严格D上的稳定加速，不能把它当线上收益的充分证据。

## 数据、比较与完整性

旧GSM8开发题单列；新16题来自相同固定官方test源，排除既有data/research GSM记录ID/规范化精确问题及source_index<32，再按预定SHA排序取题。附查5个旧JSONL请求文件，16题均未出现。训练32/内部校准8均来自官方train，与评估隔离；suite reserve96未生成。独立仅指项目未选择/生成，不证明上游模型未见或语义近重复隔离。

四模型base/ordinary/ordinary-position/asd-prefix各测strict DFlash与原ASD B8/g.25/M2，固定Target、prompt、native16、FP32/highest/TF32关闭/eager attention/rmsnorm-inductor。每题完整自然4096/EOS三重复、128/EOS两重复，24×8×5=960；另24次AR和18短warmup。每题同GPU、八臂固定seed交错，模型合并准备单列，全部在线生成成本计wall。

所有严格臂逐token与新完整AR相同；旧8题base两策略也重现旧完整text/EOS。全部重复轨迹/账本稳定、输入/快照一致、实际KV前沿通过、ASD预算守恒；本轮八臂全部自然EOS，无4096截断。正式新增编译0。

## 主结果：相对同策略原权重的题均时延

|模型 + ASD|旧8完整自然|旧共同128的7题|新16完整自然|新共同128的16题|
|---|---:|---:|---:|---:|
|base|1|1|1|1|
|ordinary|0.991738|1.021445|1.003074|0.984321|
|ordinary-position|0.980634|1.012067|**0.926838**|**0.967683**|
|asd-prefix|**0.955884**|**0.966203**|0.991731|0.989509|

旧短题1150遵守EOS，共同128集合统一要求全部四模型实际达到128。新16题全部达到128，因而新前缀没有短EOS集合差异。比值先平均同题重复，再相对同题基线，最后题等权；不把960次运行当960道独立题。

旧ASD-prefix自然省4.41%、共同128省3.38%，与上一轮4.46%/3.34%接近，验证旧信号可重复。它在旧题也比同位置组快2.38%/4.02%；**不能把这批旧正结果全部解释成生成位置差异**。

新16题上，ASD-prefix自然仅省0.83%、128省1.05%；相对同位置组反而慢7.77%/2.85%。原ordinary自然慢0.31%、128省1.57%。同位置组自然省7.32%、128省3.23%，但自然总值受错误路径缩短影响，必须结合下节解释。

|新16 + strict DFlash|自然时延/base D|128时延/base D|正确|
|---|---:|---:|---:|
|ordinary|0.995356|0.988952|15/16|
|ordinary-position|0.999334|1.011201|15/16|
|asd-prefix|0.998810|1.004884|15/16|

同位置组严格自然基本持平、严格前缀慢1.12%。目前正信号主要是它与ASD近似策略的交互，不能称普遍提高Target严格匹配或无损Draft加速；本轮没有重复AR计时矩阵，不拼接旧AR速度作新倍速结论。

## 正确性和错误长路径

旧8题全部八臂均8/8；新16题AR/全部strict为15/16，四个ASD模型均14/16，逐题正确集合也相同。训练未新增正确→错误或错误→正确，**并不代表ASD与AR质量等价**。

gsm8k:1075严格和ASD均错。gsm8k:450严格答4正确，base/ordinary ASD答7，新位置/ASD前缀组答3，均错。新适配没有修复该ASD原有质量损失。该题base ASD输出1125 token/221验证轮，新位置385/79、ASD前缀463/92；把错误长路径缩短不能当正确推理效率提高。

全16题自然ASD的token/calls：base5307/871、ordinary5373/889、position4518/704、ASD-prefix4855/784。新位置自然TPS题均仅增2.74%，不将7.32%自然时延下降全算每token计算收益。共同128均2048 token，calls依次331/325/319/326，支持新位置组存在有限等长计算改善。

额外事后敏感性检查（不替换主样本或门槛）：

|范围|position自然/base|position128/base|ASD-prefix自然/base|ASD-prefix128/base|
|---|---:|---:|---:|---:|
|四ASD模型共同正确14题|0.963032|0.965882|0.993637|0.997680|
|仅排除450，保留其余15题|0.965508|0.965428|1.030852|0.993040|

position的正确14题自然仍省3.70%、128省3.41%；这只说明信号不完全靠错答450，不是新独立确认。ASD-prefix去掉450后自然反慢3.09%，并受1075路径变长影响。保留这两道失败，禁止用事后筛题制造主结果。

预算规则没有变化。新16自然base/position/ASD-prefix近似接受token为124/127/121，regret总支出122.1900/121.7184/121.7937；128前缀为105/106/99及114.7191/109.5763/112.4909。与旧8相同例外数的情况不同，新位置组实际多接受少量近似token，但未增加B/g/M，不能把计数变化隐去。

## 已取得的贡献、适用范围与采纳决定

本轮贡献是实现并验证了一套位置匹配的严格前缀适配配方：复用冻结Target和成熟DFlash，以小规模离线LoRA训练改进固定ASD部署；推理时合并为原dense结构。新16题共同128时延降低3.23%、验证调用331→319；全部四ASD模型正确集合相同，适配未新增答案错误。共同正确14题的事后分析仍有自然3.70%、128前缀3.41%的收益。性能改进、实际训练实现和保留完整正负对照共同构成本阶段成果。

预先指定的请求bootstrap10000次结果保留：新position/base自然区间[0.8348,0.9905]、128[0.9355,0.9974]；相对ordinary分别[0.8303,1.0776]/[0.9573,1.0068]。ASD-prefix/base自然区间[0.8855,1.0962]、128[0.9435,1.0428]；相对position为[1.0105,1.1604]和[0.9722,1.1129]。原ASD内容增量点门槛未通过，记录不改；位置匹配方案按已测场景收益采纳。

适用范围为本轮配对模型、执行profile、GSM题集、32训练题和一个训练seed；旧题位置组前缀退化、ASD质量差异及错误长路径均保留。按用户决定，本阶段不再安排补seed，也不以证明所有任务或所有初始化都获益作为接纳条件。后续优先整理固定训练配方、适配器身份和工程使用入口。默认严格服务及独立面试版继续保留；全参训练、完整KL/on-policy刷新未评估。

## 成本与发布来源

收集80新features共892671808B（约0.831GiB），两worker含准备54.246/56.030秒；新臂训练段11.166秒，不含之前加载/读取。532480参数、首梯度norm0.14908、20个adapter张量/2132408B，选中/末128 SHA相同：42755672a816b44b727cbc5bcdbbd2b18dd2a19db42e5706f09d540d678b1f6e。Target无梯度、Target/Draft base版本守卫通过；80features SHA/shape/finite与adapter SHA/shape/finite另CPU核验，原160features/4权重前后SHA全同。权重/hidden不推Git。

评估峰allocated37476341760B（约34.90GiB）；每卡4图/backend3.44966/3.31189秒、cache417205B、正式新图0。首AR32冷wall5.81727/5.66429秒、TTFT3.54100/3.43118秒，编译0→4图在请求内实际支付。离线dense合并总20.0606/20.9785秒另列，不声称在线切权重免费。执行后两卡19MiB/0%。

实际GPU来源411c2211e6cdfe9d20365d40986f41de4590fdf3/1651 Git blobs，source SHAbc433a997c85ded26a40342e90f4ea41b3612a19e0b185ad38660072b9f70190；本地完整ZIP SHA8c6e19822a401428a0a5aec919101a85fbd819fe95caef37ee200e8074f0bfdf。实际逐SHA重建采用旧完整ZIP+2347345B delta，SHA1ee08de7e1fd922d0f3f99e8020f6dcee06bbaceabb0791a9d5c1c207255f054，不声称传输了新完整ZIP。PID852273于2026-10-22 11:44:14 UTC（北京时间19:44:14）启动，12:14:31末核查时pipeline已完成。结果ZIP768701B/SHA1f42320a13b75717747bd7ff5b38e91e52aa9274292c535b076c358fcb7c4cd4，24成员原始字节核验。

运行后修复prepare向另一目录导出时误把自身题集纳入排重的问题，单独commit8ed9ee42eb12f0e77d31b3df7631d12f9890d099；记录/manifest/license重新导出逐字节相同，原records SHAd0f3d12e6f0d92c4a1ed1fbd1e5b86d3785d9d63192cf5e662253b5342cbbfc2。服务器源码、训练/评估及本地主统计均仍411c221，没有重跑或改数。回归先1 failed/3.89秒、修复后相关18 passed/23.46秒，失败证据保留。本地GPU前全量829 passed/7 skipped/128.95秒，服务器104 passed/24.94秒；新增导出测试不冒充已包含在前两者。

服务器Python3.11与本机3.13主统计逐字段完全一致。后审计1651 Git来源、1126旧结果文件（两个既有CRLF例外单列）、80对齐状态、960实际KV前沿、训练/评估隔离及原infra三个引用通过。主表ASD_ADAPTATION_FOLLOWUP_RESULTS.md；公开raw/审计/事后敏感性在results/asd-adaptation-followup，原项目和默认引擎保持。

本机复核（原始ZIP仅在artifacts，用于压缩字节核验）：

```powershell
.venv\Scripts\python.exe -m scripts.write_asd_adaptation_followup_report
.venv\Scripts\python.exe -m results.asd-adaptation-followup.inspect_results
.venv\Scripts\python.exe -m results.asd-adaptation-followup.sensitivity
.venv\Scripts\python.exe -m results.asd-adaptation-followup.audit_postrun
```

服务器原目录不可覆盖；新目录复跑入口及预算见ASD_ADAPTATION_FOLLOWUP_PROTOCOL.md。数据重导出可使用`python -m scripts.prepare_asd_adaptation_followup --source artifacts/asd-attribution-gsm-source.jsonl --output artifacts/followup-reexport-check`，输出目录须新建。
