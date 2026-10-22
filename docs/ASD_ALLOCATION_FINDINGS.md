# 阶段3完成：保留原生16块，局部选择空间尚不能被廉价特征利用

本轮完成固定模型下的四动作成本、有限oracle及留题预测检查，按生成前协议不进入闭环选择器扩测。请求级几乎没有自适应空间；局部MBPP/短问答存在机会，但预登记单层选择器未通过任何合格场景的门槛。结论限于本轮原生b16模型、四动作和四个廉价特征，不表示所有计算分配算法都无效，也不是ASD近似策略的完整新评测。

## 实际范围

六类已观察开发题各2，共12题。12次native512源运行逐ID/prompt/snapshot重现旧完整DFlash/RMSNorm相应前缀；144次请求级四动作×3重复；27个实际状态×四动作×3重复=324次保留KV的局部分支，共480主生成，另10短warmup。计划轮0/4/12中9处因源提前结束而缺失，未补位或换题。validation不作为新独立，reserve96未生成。

请求比较native16、实际Draft8、Draft16/只验证7、原生AR，最多128token且遵守EOS；局部比较同一真实缓存前沿的一次动作，之后恢复严格native，最多64个新token且遵守EOS。局部Target单步保留真实hidden features，后续Draft实际追赶，不假装免费切换。当前起草/验证/提交及未来计算都实际支付；共同重放与prefill不计局部clock，所以局部比值不能当成完整请求收益。

144请求全部逐token相同；27状态的四动作完整同范围输出全部相同，三重复各自输出/轮次精确重现。后审计还核对所有共同前缀、全部过去轮次/缓存前沿和动作之后恢复native。没有靠缩短答案获得收益；128/64/512上限仍不能冒充完整自然答案质量检查。源保存旧完整质量，本轮没有新部署策略或新增完整ASD质量结论。

## 真实成本与oracle

下表为请求等权的native归一成本均值，局部先在每请求内对状态等权；与前轮RMSNorm几何平均统计口径不同。

|动作|完整请求128/EOS|一次局部动作及后续64/EOS|
|---|---:|---:|
|原生Draft16/verify15|1.000000|1.000000|
|实际Draft8/verify7|1.224244|1.016271|
|Draft16/只verify7|1.182876|1.015223|
|原生AR／轮内Target单步|3.748638|1.076463|

实际缩短Draft导致请求耗时平均多22.42%，只缩短验证多18.29%。Target调用数相对native的题级中位比为1.1952/1.1752，每调用摊销总成本中位比为0.9703/0.9782：单次略省，但更多调用抵消收益。两组中位数不可直接相乘解释总体，逐题精确分解在interpretation.json；每调用成本包含起草等全部在线开销，不是独立Target kernel时延。

|范围|事后oracle/最优固定|跨重复选择/固定|留题预测/固定|
|---|---:|---:|---:|
|请求，总体12题|0.990718|0.991221|1.041159|
|局部，总体27状态|0.951273|0.951369|1.045080|
|局部，MBPP两题|0.928661|0.928661|1.003124|
|局部，SQuAD两题|0.836896|0.836896|1.298426|

请求事后oracle仅省0.93%，没有范围通过事后>=5%且跨重复>=3%的空间门槛；SQuAD请求oracle相对其最佳固定half省4.15%，保留但未过门槛。局部总体省4.87%，MBPP/SQuAD分别省7.13%/16.31%，两个局部场景通过空间门槛。跨重复结果相近，机会不只是一次抖动选赢家；样本仍小，不能称新题泛化或统计显著。

预登记单层stump仅见已确认长度、prompt长度、过去最多4轮平均推进及上一轮推进；每叶至少3状态，阈值只用训练题，按请求留一。家族表现使用全体其他请求训练的同一模型，不按该族两题重新挑模型。按其选择回查实测动作成本，请求/局部总体分别多耗4.12%/4.51%，合格局部MBPP多0.31%、SQuAD多29.84%，无场景通过留题再省>=2%的门槛。这是尚未支付在线选择开销的离线估计，不是已运行闭环选择器的端到端时延。训练全部请求后的请求模型直接退化为常量native；不把训练内最优模型当作部署证据。

## 机会来自哪里，预测为何失败

局部正例集中在是否即将结束：mbpp:262源轮4只剩EOS，Target单步0.05022秒，而native0.07204秒，少付一次Draft且不增加后续验证。两token的squad:5734025d4776f419006616c6也只剩EOS，单步0.05003秒对native0.07568秒。

另一道四token问答squad:572651f9f1498d1400e8dbef尚余3token：native一轮完成，单步先推进1个后还须Draft/验证，0.13517秒对native0.08687秒，反而慢约55.6%。留题模型恰好在该例错选Target单步；它在两个问答源轮0看到相同generated=1/recent=1/last=1，仅prompt长度不同，不能可靠分辨哪个下一步就是EOS。MBPP的结束位置前几轮推进并不低，也无法仅用近期接受长度识别终止。

这是对本批记录的解释：真正有用的动作有时取决于尚未知晓的剩余生成/终止状态，而现有特征学到的prompt长度分割缺乏留题稳定性。不是“最优动作永远native”，也不能把知道未来EOS的oracle包装成免费在线信号。终止状态之外的一次短动作还可能改变后续Draft块边界，须保留真实KV及未来成本；不能只看当前推进或把所有局部oracle节省串成全局上界。

## 阶段决定

按协议本轮阶段3结束：保留原生16块和既有RMSNorm执行选项，不启用动态长度/AR选择器，不继续扫四特征阈值、不加复杂预测头、不消费保留集，也不因为oracle局部有正数就启动昂贵完整自然生成。保留四动作、真实KV分支和分析工具，以及局部终止机会，供以后有新在线信号或独立数据时提出新的明确假设。

本轮严格基线上的成本结果不能直接解释为ASD/Run/Utility策略收益；ASD预算、质量风险和旧负结果继续保持。下一研究阶段按路线表进入多候选/局部Draft修复的成本预检，本次不声称该方向已经实现。

## 复现、失败与审计

核心动作在默认无回调时保持旧trace/输出；真实tiny Target测试混合动作、Target积累后Draft追赶、EOS/输出上限/stream和默认等价。原生模型与上游MIT DFlash保持既有revisions，没引入上游新代码。理论依据见协议中的[DISCO](https://proceedings.mlr.press/v262/mamou24a.html)、[DFlash](https://arxiv.org/html/2602.06036v1)、[BlockPilot](https://arxiv.org/html/2606.31315v1)，不声称动态长度首创或论文复现。

冻结cea9ebb92db4879144b79b0338cd1272b38b0a35/1491 Git blobs，source SHA5dc11ff081b1c81798e2f6ee4bd5c8e227bc5b8d828b178948261538a9af3bf0。本地完整ZIP SHA1e0cd79e5476e1c1284e80da4be82f99550c6313e3926e4882a7003657ac28cc；部署为旧已校验完整ZIP+404622B delta（SHA6b5b909f89a0e9cf1258fbf629cf6450db43aad2e61d0937234cebdcf0b0479c），1491目标blob全部验证，不声称服务器收到了新完整ZIP。独立asd-allocation-v2/PID568165，2026-10-22 04:02:59–04:09:43 UTC，ASD ALLOCATION COMPLETE，两worker0退出，两卡恢复19MiB/0%。

每卡4图，正式无新编译；outer3.804/3.824秒、backend3.426/3.432秒（嵌套不相加），cache各414716B，最大记录allocated35.158GiB。warmup单列，不将其wall全部当编译；局部oracle排除共同prefill/重放，不据此算生产摊销。

本地相关60 passed/2 skipped/11.58秒、完整731 passed/5 skipped/112.80秒；后续资源/部署报告相关13 passed/6.65秒。首次报告合成夹具每侧3题，留一后不足预定每叶3状态，改为每侧4题保持规则，原失败XML保留。首次服务器81 passed/8 errors/9.94秒，因新部署没有artifacts父目录导致pytest临时目录失败，GPU未启动；仅pipeline补mkdir及回归用例，原失败目录/ZIP保留。修复后服务器90 passed/9.76秒，生产动作、数据、主统计和门槛未改。

成功结果ZIP264854B/SHAf3c835e254d39259771824e1488a65f34891391adfc5ee426c44b56731140684；原前置失败ZIP73598B/SHA1bf52dc8e04065b162ba0ed7d08638e9e3a5bbbf669524d14569b7d2de4d0f45。后审计1491冻结Git blob、8个当前冻结文件、1006旧结果SHA、10批387个已发布文件、所有真实KV共同前沿和后续native，以及面试infra三个27e36199引用通过，只有两个既有CRLF例外。服务器Python3.11从原未压缩行重建主统计，与本地Python3.13仅11浮点字段有最大3.33e-16末位差，选择/模型/判定相同，记录statistics-rebuild-server.json。主统计/评分/执行代码GPU后未改，新增interpretation为只读事后描述，不改变门槛。

运行：`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_allocation_pipeline`；报告`python -m scripts.write_asd_allocation_report`，描述`python -m results.asd-allocation.inspect_results`，审计`python -m results.asd-allocation.audit_postrun`需Git历史及原ZIP。结果results/asd-allocation；发布小型公开记录、测试、报告，排除权重、完整hidden/logits、数据库与编译缓存。
