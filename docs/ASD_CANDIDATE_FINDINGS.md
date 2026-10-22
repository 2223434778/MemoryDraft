# 阶段4：有限树候选与局部后缀修复结论

本轮完成六类各2题、六方法各3次，共216次完整自然请求，另12次短warmup。全部题都在4096上限内自然EOS，六方法各题输出token、文本、EOS及原Target输入/快照均与冻结参考相同，重复轨迹稳定。两种直接token后缀修复没有净收益；32节点树有局部正线索，但总体略慢，48节点树明显过宽。保留native16，不把本轮候选接入默认服务或直接套入ASD。

预登记的自动扩测条件未通过：总体至少省3%，或家族至少2题均更快且平均省5%，三个重复方向一致、全题严格安全。tree32在GSM8K平均省3.45%，值得保留，不够这次5%工程门槛。这里的门槛不是统计显著性或理论下界，不能把未准入写成该场景无收益；也不将2题的正数写成分布级确认。按原协议停止本固定预算/直接后缀方案的自动48扩测，不临时降低门槛、换题或扫描预算。

## 成本与自然输出

固定Qwen3-8B/DFlash b16、FP32 highest/TF32关闭、eager attention、rmsnorm-inductor、严格greedy、空历史和原prompt。tree32/48保留原16节点链，前4个proposal位置分别加Top-2单替换路径，分支含替代token共4/8节点。repair4/8使用真实恢复token作新anchor，重验其后旧Draft的最多4/8token；一次后重新起草。没有复用受错误前缀污染的Target状态。

|方法|题均自然时延/native|更快请求|Target验证调用|Draft调用|验证节点|
|---|---:|---:|---:|---:|---:|
|native16|1.000000|基线|278|278|4448|
|packed-linear|1.008754|0/12|278|278|4448|
|tree32|1.018533|3/12|265|265|8480|
|tree48|1.188727|0/12|264|264|12672|
|repair4|1.176796|1/12|380|216|4258|
|repair8|1.178011|0/12|379|216|4760|

时延是每题三次wall均值除同批native均值后题等权，重复不是独立题。调用/节点为每题一条稳定轨迹之和，**没有乘3**；Target列不含各方法相同的12次prefill，prefill及全部在线成本仍在请求wall内。验证节点不等于每节点同价FLOPs。原始轨迹所有额外计算、CPU/设备同步、mask、Top-2提取、选路、KV整理、特征追赶、EOS检查和decode均计入。

每方法合计1987输出token，12/12 EOS；不是截断前缀或EOS后强制工作量。质量完全保持旧参考：GSM8K 2/2、MATH 2/2、MBPP 2/2、HumanEval 1/2、JSON 2/2，SQuAD EM 1/2、F1均值0.785714。严格一致不意味着原Target在每题答对，HumanEval:134旧测试失败和四token问答旧EM失败全部保留。

## 适用场景线索：轻量树不是处处无效

|家族（各2题）|tree32/native时延|变化|
|---|---:|---:|
|GSM8K|0.965504|省3.45%|
|HumanEval|0.974818|省2.52%|
|MATH-500|1.049904|慢4.99%|
|MBPP|1.046622|慢4.66%|
|SQuAD|1.014274|慢1.43%|
|JSON|1.060076|慢6.01%|

GSM8K:52时延比0.968742、Target调用50→45；885为0.962267、37→33，两题均更快，家族三个重复比分别0.967290/0.963402/0.965836。这个结果支持“中等接受长度、早期分叉能实际减少轮次”的窄场景线索。它仍只有2个已观察开发题，不能承诺更多GSM8K有效，更不能用任务名称作为已验证在线路由规则。

HumanEval平均改善被单题主导：82为0.933517、调用7→6；134为1.016119、19→18。SQuAD两题原本各只需一次验证，树不能把调用再减到零；JSON两题都没有减少轮次。MATH:448只从97降到96，而新增节点贯穿全部轮次。这些差异解释了为什么不能拿接受数整体上升替代每个请求的真实收益。

## 失败机制与证据边界

**树增加了覆盖，但额外工作大于节省。** tree32有55/265轮选择替代路径，43次只比同状态线性多推进1；265轮中210轮没有树内推进增量。raw tree_gain合计79，tree48为91，但它们可能包含EOS之后已付计算，不能当最终多输出token；所有方法真实输出均1987。逐请求原始advance超出EOS/上限部分在interpretation.json单列。

tree32的验证调用减少4.68%，节点增加90.65%；tree48调用减少5.04%，节点增加184.89%。从32加到48只再少一次验证，时延却明显增加。packed-linear保留相同节点和轨迹，仅走新打包入口，整体开销约0.88%；这不能证明每个kernel的占比，却说明不能把tree48全部18.87%退化都归给Python选路。证据支持固定宽度验证代价压过有限分支收益；未做独立kernel占比归因。

**旧后缀经常紧接着再次错。** repair4有164次尝试、115次接受0（70.12%），合计只接受112个旧token；repair8为163次、112次接受0（68.71%）、接受140个。每次零接受仍推进真实恢复token，也仍支付一次Target验证。Draft调用278→216（省22.30%），Target却278→380/379（增36.69%/36.33%），导致整体约慢17.7%。加长后缀不能越过第一个再次拒绝的位置。

这些计数符合“旧候选在修正前缀后仍不够可靠”的解释，没有证明所有信息复用都失败。我们只复制token；没有实现重新条件化的Draft、错误后缀表示融合、训练修正或完整树搜索。`R`轨迹在本实验表示复用matcher薄接口，不代表检索了用户历史；请求历史始终为空。

尚未做“只传递fresh候选、从不修复”的独立matcher开销控制，所以不能把约17.7%的每毫秒退化都分配给额外Target。CPU接口成本确实在wall内；新增Target调用及高零接受比例是已直接记录的结构性代价。

## 与论文的关系及后续取舍

[BASTION](https://arxiv.org/abs/2605.29727)已经把DFlash位置分布转为动态树，并联合考虑路径覆盖和验证成本。本轮固定前四位置、单次替换的树不是它的best-first拓扑/预算控制实现，负结果不能否定它的结果；若以后继续树路线，优先核验并薄适配成熟成本约束构造，避免只增加分支长度。

[ReTrace](https://arxiv.org/abs/2608.29748v2)利用拒绝后缀的表示、对齐与修正融合辅助新Draft，本轮直接旧token重验是更便宜也更弱的对照。当前证据支持停止无条件直接重验，不支持断言训练或表示层复用无效。[OPT-Tree](https://aclanthology.org/2025.tacl-1.8/)提供树共享和拓扑优化依据，但其AR Draft成本不能照搬到并行DFlash。

现阶段保留严格native16与显式RMSNorm执行profile。轻量树GSM线索、可用树验证/KV组件和全部负结果保留，作为未来更有针对性的拓扑或阶段5实际轨迹适配的起点。未运行完整48新方法自然输出、未复现完整BASTION/ReTrace、未做ASD树账本/近似分支选择、没有新独立确认；validation48已观察、reserve96仍未生成。

## 工程、审计与复现

- GPU前冻结`2949b9ed7dda32678ef0484194554cbc43365d74`，1540 Git blobs，source SHA `3fbbfe1da9e33f728d542f6ea9835eca1e0fb1323337fa344e79d3ed251eedce`。本地完整canonical ZIP SHA `e2503bf149c76b5918dc5300bbb0363b5f5ecaba1f0e1cf5eb1cb95e89f1626c`。
- 部署为旧已校验ZIP+819188B delta（SHA `ac40da7d4e45da56a49f9bbd98d03ef21c12890a753427552adde1fa66a9a4be`），在新`asd-candidate-pilot`重建全部目标blob逐SHA核验，不声称服务器接收到新完整ZIP。前后1540源SHA一致，旧目录不改。
- PID618342于2026-10-22 05:23:43 UTC（北京时间13:23:43）启动；约13:30完成及末核查。成功归档307959B/SHA `61753a35323d691f87f33b1418b05a47b65a2f01802a7d5f3fedc166d7d9ca36`。结束两卡各19MiB/0%；一次即时观察1%利用率随后恢复0，completion.json保留两次观察，没有因此重复生成。
- 每卡3图，backend3.27113/3.24606s，cache398883B/卡，peak allocated35.1599GiB，正式新编译0。空cache首次native32 wall4.83495/4.82669s、TTFT3.52471/3.51903s，图0→3，backend在请求内；不是从warmed时延扣掉冷编译，warmup逐行单列。没有AR，不能把旧同profile AR计时与本批混合计算新加速比。
- 本地首轮81 passed/2 skipped/12.45s，相关94 passed/2 skipped/12.28s，完整791 passed/5 skipped/109.86s；随后补全全题安全准入检查14 passed/6.74s。服务器121 passed/9.47s。真实tiny Target核对树/独立路径logits及KV、分支切换后的feature追赶、stream/EOS/预算和ties。只增加可选研究入口，默认runtime不启用树或修复。
- 后审计通过1540 Git blobs、10核心冻结文件、1044旧结果SHA（仅两个既有CRLF例外）、11批434旧发布文件、所有轨迹前沿/树祖先路径及原infra三引用。Python3.11服务器与3.13本地纯统计重建完全一致，差异0。

主要代码：`memorydraft/backends/{draft_tree,candidate_repair,dflash_reuse}.py`；运行/统计：`scripts/{run_asd_candidate_probe,run_asd_candidate_pipeline,write_asd_candidate_report}.py`；新增测试：`tests/test_{draft_tree,candidate_repair,asd_candidate_report}.py`。协议、生成主表、机制解释、后审计和原始gzip记录分别见ASD_CANDIDATE_PROTOCOL.md、ASD_CANDIDATE_PILOT_RESULTS.md及results/asd-candidate-pilot。发布不含权重/logits/hidden/DB/大trace/cache。

服务器执行：

```bash
cd /media/a6000/D/ZhaoBinHe/MemoryDraft/asd-candidate-pilot
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_candidate_pipeline
```

本机重建报告和审计（已下载原ZIP，后审计需完整Git历史）：

```powershell
.venv\Scripts\python.exe -m scripts.write_asd_candidate_report
.venv\Scripts\python.exe -m results.asd-candidate-pilot.inspect_results
.venv\Scripts\python.exe -m results.asd-candidate-pilot.audit_statistics
.venv\Scripts\python.exe -m results.asd-candidate-pilot.audit_postrun
```
