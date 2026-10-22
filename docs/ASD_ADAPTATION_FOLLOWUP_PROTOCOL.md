# 阶段5后续：同位置对照与独立GSM验证（GPU前冻结）

目标：区分上一轮ASD前缀适配的收益是否只是严格源和ASD源的生成进度不同，并在项目未使用的新题上检验原适配器。用户已授权继续；不改阶段5原协议或既有结果，不从测试结果选择训练超参。

## 一次固定控制

保留原32官方GSM train/8内部校准、seed20261022、q/v LoRA rank8/alpha16/0 dropout、AdamW1e-4/.01、累积2、128更新/256样本反向、clip1、位置CE exp(-k/7)。每32步只在原来的32状态混合内部校准池选最低CE checkpoint。原ordinary/asd-prefix适配器字节不变，新增ordinary-position一臂，不重训原ASD组。

新严格源来自上一轮完整严格轨迹，截到每个ASD轮4/12样本的generated_before。全部80个位置都可在严格源EOS之前到达，训练64/内部校准16，无删样。两臂同问题、同prompt、同生成token数、同上下文长度、同shuffle键；prefix内容可以不同，实际80对全不同。截取位置不必是严格DFlash真实轮边界；这是离线前缀控制，不当作新的线上动作。记录这些边界差异，不能把同位置控制写成完全相同运行轨迹。

新严格前缀重新运行冻结Target，提取anchor之前hidden，真实anchor+15mask，最多15个新greedy teacher标签/EOS，不复制旧teacher标签，不复用错误上下文KV。旧ASD前缀、adapter及原混合校准feature只读复用并核SHA；新增同位置内部校准16个teacher状态仅诊断，不加入选checkpoint的原池，以保持选模规则相同。新臂仍64训练状态/32题，训练FLOPs与ASD源长度匹配但不假装两次执行wall相同。

## 独立题及测量

官方GSM test固定commit3101c7d5072418e28b9008a6636bde82a006892c及SHA3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14。排除source_index<32、data/research所有既有GSM记录ID/规范化精确问题（包括suite的开发/验证/保留32题及训练40题），按SHA256("MemoryDraft-adaptation-followup-v1:"+ID)排序取16。仅按题元数据选取，不读生成结果；保留原GSM8开发题单列，不混入独立结论。独立指项目此前未选择/生成，不证明上游预训练未见或语义近重复隔离。suite reserve96不生成。

四固定dense模型：base/ordinary/ordinary-position/asd-prefix；各strict DFlash和原ASD B8/g.25/M2/longest。Target权重/prompt/tokenizer、native16、FP32/highest/TF32关闭/eager attention/rmsnorm-inductor完全相同。不引入Run/Utility/树/OT或新的近似条件。原adapter SHA在模块常量和前一轮记录中双重核对；新adapter只由内部CE选定。

24题×4模型×2策略×(完整自然3重复+128/EOS前缀2重复)=960正式计时，另24次完整AR参考和每卡9个32-token warmup。所有完整生成上限4096，EOS立即结束，短题不强制续写。每题同GPU，按预定seed adaptation-followup:<id>:<stage>:<repeat>交错八臂；离线dense合并/同步在请求开始前单列，不能宣传在线换权重免费。每次fresh KV，wall包括prefill、Draft/Target、判定、KV、EOS/decode和末同步。冷编译留在首warmup请求内；各臂匹配相同执行profile。正式新增编译和冷成本如实记录。

每题先生成完整AR并保留；所有strict逐token、完整text/EOS与AR比较，旧8题base两策略也对旧完整参考。ASD不强求与AR同输出，逐题保存最终答案、正确→错误/错误→正确、截断、输出长度/TPS、验证调用、近似接受数量和regret支出；质量与计时分列。所有重复轨迹/账本必须稳定；若完整性失败保留结果，不宣称无损或稳定。

## 主对比与判断

主单位为请求：各臂重复均值除同题base/ordinary/ordinary-position均值，再题等权；旧8与新16分别汇总，不能把计时重复当更多独立题。完整自然和四模型均实际达到128的共同子集分别报告；共同集合对所有四模型一致，不把短EOS当固定工作量。提前指定请求重采样10000次bootstrap，seed固定，给ASD-prefix相对三个对照的均值和95%区间；小样本区间是本数据的不确定性描述，不是多假设普遍显著性证明。

若独立题上ASD-prefix相对同位置对照的自然和共同128点估计均更快、无新质量失败，且至少共同128时延降低>=2%，保留“ASD内容增量候选”；区间跨1则明确未稳定确认，不自动升级。若新同位置组追回旧收益，则优先解释为生成位置分布效应。若两组都只在旧题快或自然缩短而共同128不快，保留开发正结果但停止宣称独立计算收益。单seed和有限LoRA不能否决全部OPD训练；不追加lr/rank/epoch扫参或按答案删题。

## 资源、完整性、命令

两张A6000当前空闲、125GiB RAM、约403GiB磁盘可用；不下载新模型。teacher收集每worker<=1800秒、context<=1024、新features合计<=16GiB/每卡<=8GiB；新训练段<=900秒/worker<=1200；评估每worker<=5400秒，GPU allocated<=46GiB；Inductor<=8图/backend300秒/cache4GiB每卡。CPU前置<=300秒。只结束本轮owned进程；旧源、160features及4个selected/final adapter运行前后核SHA，禁止修改旧执行目录。权重/features留服务器artifacts，不归档不推Git。

新目录/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-adaptation-followup；旧只读资产在../asd-adaptation-v2。源码、数据、协议和统计先commit冻结；用canonical Git blobs归档逐SHA部署，不依赖Windows换行。入口：

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_adaptation_followup_pipeline
```

结果results/raw/asd-adaptation-followup，公开results/asd-adaptation-followup；`python -m scripts.write_asd_adaptation_followup_report`重建主统计。默认runtime、main、面试branch/tag不改；完整失败现场单独保留。
