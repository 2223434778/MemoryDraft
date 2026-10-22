# 阶段5：等预算Draft领域/部署前缀适配（GPU前冻结）

用户授权实际训练。此前16状态必要性诊断没有发现ASD前缀退化，只约束当时局部检查，不能代替训练结果；本轮独立固定小预算，对比普通领域适配和ASD部署前缀适配，不把旧否决改写成训练已失败。

## 文献与实现

- DFlash https://arxiv.org/html/2602.06036v1 的训练以冻结Target hidden features、真实anchor及其后15个mask匹配推理，冻结共享embedding/head；位置交叉熵权重exp(-(k-1)/7)。沿用已审计DFlash@07ebd93及b16权重9b41424，不复制新上游代码，不从头训练。
- AdaFlash https://arxiv.org/html/2607.19223v3 、Draft-OPD https://arxiv.org/abs/2605.29343 支持在部署到达的前缀上纠正Draft。本轮只做冻结原Draft采集一次、teacher新生成硬标签的离线适配；没有完整反向KL/混合损失、在线刷新buffer/长度头或异步serving，不称这些论文的完整复现。
- LoRA复用现有PEFT0.11.1，generic PeftModel forwarding适配DFlash custom target_hidden，不改模型forward。q_proj/v_proj、rank8/alpha16/dropout0/biasnone；冻结Target和原Draft权重，只训练LoRA。推理时合并成原dense投影，不新增在线LoRA分支。资产许可保持现有MIT，未引入新上游源码；数据GSM8K MIT notice随子集保留。

执行前置修复：服务器系统旧bitsandbytes缺少CUDA124库，PEFT0.11.1创建普通FP32 Linear时仍会探测它。仅对已验证type==torch.nn.Linear/dtype==FP32的投影，在get_peft_model调用期间临时禁用两个bnb量化dispatcher availability探测并自动恢复；不修改共享环境/库/精度，不假装CUDA不可用，也不支持量化模型。首次2 failed/84 passed的CPU前置与源commit9bb81be归档保留，GPU worker未启动。预算/问题/主统计不变，新root重试。

Target输入、权重、greedy、非thinking、原native16和验证规则固定；训练/teacher采集FP32/eager，评估统一FP32/highest/TF32关闭/eager attention/rmsnorm-inductor强执行profile。融合norm只在评估启用，训练保留普通可微forward；不混合不同profile的时延比较。ASD继续原B8/g.25/M2/longest，不重定价，不使用树/R/OT，不把近似输出当严格加速。

## 数据隔离及实际轨迹

官方openai/grade-school-math固定commit3101c7d5072418e28b9008a6636bde82a006892c，grade_school_math/data/train.jsonl，SHA17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465。下载<10MB。训练候选与完整官方test及suite题按小写/空白规范化精确排重；这不证明语义近重复或上游预训练数据隔离。按SHA256("MemoryDraft-adaptation-v1:"+sourceID)排序取前32训练、后8内部校准；不读生成结果选题，子集questions.json仅问题/ID/split/source_index/benchmark，没有参考答案。

评估使用原suite development，GSM8题，其余五类各前2，共18题。与训练/校准ID和问题隔离；现有validation48已观察，不当新盲测，也不用于训练、选checkpoint或调超参；reserve96不生成。内部校准来自同一官方train但与32训练问题不相交，不能叫新外部分布验证。

每个训练/校准问题分别用原权重严格D与ASD生成至EOS或256，采实际Draft轮4、12的起草前frontier（on_draft在本轮Draft后、Target验证前观察相同已确认prefix）。strict源是Target greedy轨迹，ordinary学习该源；asd-prefix学习ASD到达源。对应轮不保证同生成长度，分布差异包含前缀和进度，不声称纯单token因果干预。某轮任一源已EOS则两臂都不取该配对，不补题/补状态；缺失公开记录，不能按最终正确性筛训练数据。

每个状态独立从原prompt+实际已确认prefix（含真实anchor）重新运行冻结Target：提取anchor之前的全部目标层features，之后生成最多15个新的严格teacher token，遇EOS停止。输入features绝不含anchor之后真实token/未来hidden；ASD错误上下文的旧Target logits/KV丢弃，ASD后续输出不当监督标签。每个源样本是单个anchor+15mask，padding位置loss=-100，teacher EOS仍计loss。相同prefix按SHA去重teacher缓存；训练分配仍保留每个source状态角色。

记录ASD状态此前spend/relaxed和两源prefix SHA，区分真的分布差异与相同样本。最低覆盖训练>=16问题、内部校准>=4问题，否则明确不可评估，不生成假训练结果。理论上最多80源生成+160个teacher状态；实际源/缺失/唯一teacher状态数全部报告。teacher块不是完整自然答案，训练采集不是在线服务性能。

## 等预算训练与checkpoint

两组同原Draft初始化和seed20261022，使用配对状态键的相同shuffle顺序，batch1/累积2、AdamW lr1e-4/weight_decay.01、clip1、128个optimizer更新=256样本反向；不扫描rank/lr/loss/epoch。每32更新在相同“内部校准严格源+ASD源”混合池测请求等权位置CE、严格最长匹配长度。按四个训练checkpoint中最小混合CE选最终推理权重，保留step0基线和最后128权重的指标；即使全差也选择最好的已训练checkpoint并如实报告，不能偷偷退回原权重称训练有效。

实际校准forward开销和选择成本在训练计时内。两组更新/样本反向数相等，不假装prefix长度不同会产生完全相同FLOPs或wall；记录训练wall、可训练参数、首个非零有限梯度、loss曲线、checkpoint SHA、GPU峰值。optimizer仅持LoRA参数，Target frozen/无grad，Target和Draft base parameter pointer/version守卫前后不变；真实tiny Target另做全部权重逐位核验，不把版本守卫冒充全模型密码学哈希。

每臂训练段<=900秒（parent worker1200含加载），无并发线上推理负载。特征缓存两卡合计<=16GiB/每卡8GiB；每状态上下文<=1024，超限失败不删问题；各阶段peak allocated<=46GiB。收集worker1800秒；评估worker3600秒；CPU前置300秒。只结束本轮创建的进程会话。服务器两卡A6000/125GiB内存/磁盘>400GiB可用已核对；没有新模型下载。

## 配对评估与结论规则

base/ordinary/asd-prefix三种dense Draft × strict D/原ASD，共六臂。18题完整自然EOS/4096上限各3重复=324；128/EOS前缀各2重复=216，总540正式计时；每卡每臂32短warmup，共12单列。固定seed adaptation:<id>:<stage>:<repeat>打乱六臂，同题同GPU。每臂从同CPU base投影重新合并，禁止累积delta；固定模型准备/合并wall单列，完成同步后才进generate计时。此为三种固定部署模型的比较，不宣称在线每请求换权重免费。

所有请求生成wall包含prefill/Draft/Target/KV/策略/EOS/decode与末同步。空cache编译在warmup请求内实际支付、保留冷cost；不从请求里扣编译。独立Inductor/Triton cache，<=8图/backend300秒/4GiB cache，正式新增图记录。打分和模型离线准备不计请求wall，训练采集代价另列，不能把“训练免费”当性能结论。

主单位是请求，三次均值除同题同策略base均值后题等权；GSM8是预指定训练域，其余五类各2只是迁移诊断。strict必须与冻结完整DFlash/RMSNorm逐ID/text/EOS一致，base ASD自然输出也核对旧参考；所有新分歧/重复不稳/错误都保留、阻止相应无损或稳定结论。相同正确数不能当质量等价；逐题列ASD正确→错误、错误→正确、截断与输出长度。

ASD自然输出可能变长/变短，必须同时列时延、TPS、token、调用数和质量；不单凭更短输出称计算提速。128/EOS结果另列全18题和base/ordinary/asd-prefix三者均实际达128的共同子集，不强制续写、不把短EOS当固定工作量。两种训练权重除相对base外还直接比较，避免把普通领域适配收益算作ASD前缀独有收益。

本轮目标是完成这一次等预算实际训练及18题评估，不承诺训练有效，不在评估题上扫参或继续训练。根据实际loss/严格时延/ASD质量和前缀成本，判断是普通适配值得继续、ASD源有额外价值、训练预算不足，还是当前小LoRA无收益；不据此否定全参训练/更大数据/完整OPD。需要扩大时另冻结新数据与预算，不能把本次18题当独立确认。

## 执行与发布

新增模块memorydraft/evaluation/adaptation.py；prepare_asd_adaptation、run_asd_adaptation_{collect,train,eval,pipeline}和write_asd_adaptation_report脚本及相关测试。默认推理引擎、原面试分支/tag/main和模型文件不改。

服务器新目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-adaptation`，入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_adaptation_pipeline`。先真实tiny DFlash/PEFT前置，再双卡收集、双卡各训一臂、双卡分题评估。新源码/问题/协议/主统计在GPU前commit冻结并canonical Git blobs逐SHA部署，前后核查全部旧证据。

原始results/raw/asd-adaptation，发布results/asd-adaptation，主报告`python -m scripts.write_asd_adaptation_report`。权重保存在服务器artifacts/adaptation-adapters，features在artifacts/adaptation-features-*；这些权重/hidden/logits不归档、不推Git，仅发布SHA/shape/数量/预算。只发布公开训练问题、生成token、统计、日志和审计，保留全部失败现场。
