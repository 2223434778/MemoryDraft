# 项目进度

## 2026-10-19 — 接受长度条件生存校准（完成，局部改善未达门槛）

- 冻结执行提交`36798f32c2992511ea89479be7fd70279dfc4f5f`，671个Git原始文件SHA核验；源zip SHA`2aae5e51ae81c41bd9842296c586c1c27efb4f7511d5a8d42abecfdb1979580f`，实际package SHA`b6ec034a21ab76ca2d57912a8599214d9943aa22b792377610f68fd6b7c5a435`。独立服务器`/media/a6000/D/ZhaoBinHe/MemoryDraft/ot-r-acceptance`，原文件执行前后均SHA一致。
- 首次漏建models软链接，load_engine在生成前退出；失败日志保留。补`models -> ../models`后同源码运行完成，服务器相关45 passed/5.72s（首次无模型缓存时43 passed/2 skipped）。生成进程2327090已结束、GPU恢复双19MiB/0%。
- 实际运行协议中的`python -m scripts.collect_acceptance_probe --config configs/runtime-ot-benefit.json --output results/raw/acceptance/collection`和`python -m scripts.evaluate_acceptance_probe --input results/raw/acceptance/collection --output results/raw/acceptance/evaluation`，环境OMP_NUM_THREADS=4/HF_HUB_OFFLINE=1/PYTHONPATH=.；完整脚本保存`results/acceptance/run-pipeline.sh`。48/48生成严格匹配AR token/prompt/snapshot，24/24旧matrix路径计数一致，全部旧证据递归SHA不变。
- 56去重Draft状态/69shadow候选/9项目，完整匹配14原状态；每次排除当前整个项目，固定特征、正则与参数。旧/无置信度/有置信度MAE分别10.472736/13.917830/8.423701，Brier0.234529/0.276227/0.155672；新比旧MAE降低19.57%，未达预定20%。按项目等权MAE改善0.962952，5000次bootstrap区间[-5.042062,6.926308]跨零，开发门槛失败，不接入默认路由、不调参凑过线。
- 有效信号：56状态Draft MAE无置信度5.042924→2.122956；payment/11预测32→5.57实际3；shipment/14预测30.89→8.03实际7；shipment/22预测30.56→7.34实际6。payment/15恢复TailOT优于Edit、shipment/22恢复Edit优于TailOT的候选排序。
- 尚未修复：原14动作平均偏差+4.18→−7.50，对原本正确的长接续明显低估。Draft概率仅覆盖前15位置，16–32尾部依赖少量跨项目条件标签；样本、正则和尾部特征各自作用尚未因果区分。保留前缀风险模块，未来优先独立项目风险排序/条件尾部校准，不将本次离线改善当作在线加速。
- 原始归档116833字节/SHA`1479a35f1c04cee9cc61005c1dcc463d1009f7d91fa606688c2bfcfc2cb1921a`，紧凑证据在`results/acceptance`，报告`docs/ACCEPTANCE_RESULTS.md`。不含数据库、权重、完整词表logits；报告按服务器已执行系数重算，不跨SciPy版本重新拟合。报告/审计10 passed/6.92s，重建命令`.venv\Scripts\python.exe -m scripts.write_acceptance_report`。
- 最终完整回归351 passed/3 skipped/48.27s：`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-acceptance-final --junitxml=artifacts/acceptance-tests-final-local.xml`。此前一次自动审批因额度耗尽未启动该命令，用户恢复额度后运行完成；不是测试失败。原infra三引用仍为`27e36199f8a65acdb9a9c42ccb744ddfa9a9402b`。

## 2026-10-19 — 接受长度条件生存校准（执行准备）

- 针对旧14状态MAE10.47，新增 `retrieval/acceptance.py`：Draft前缀生存 × 边界通过后的历史条件生存，候选分别预测，固定正则、逐项目排除；不使用完整OT矩阵，不更改默认路由。
- decoder新增默认关闭的只读 `on_draft` 诊断接口；读取现有logits，不额外Target前向。`collect_acceptance_probe` 固定16旧请求×3旧路径48次，全部strict检查；旧matrix24路径计数对照，原14状态须完整Draft和候选匹配。shadow候选越过线上调用预算仅作离线诊断，标签strict通过后添加。
- `evaluate_acceptance_probe` 对比旧预测、无置信度、有置信度；原14固定动作MAE/Brier、全部状态、候选排序、按项目bootstrap。协议 `ACCEPTANCE_PROTOCOL.md` 预先冻结通过条件，不按结果反复调参。本轮无速度声明，无独立holdout。
- 首轮模型/decoder测试41 passed/2 skipped/17.02s：`.venv\Scripts\python.exe -m pytest -q tests/test_acceptance.py tests/test_dflash_reuse.py --basetemp .test-tmp-acceptance-unit`。将使用新服务器目录 `ot-r-acceptance`，保护全部旧证据和独立infra版本。
- 冻结前相关43 passed/2 skipped/9.83s；完整347 passed/3 skipped/60.79s，命令`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-acceptance-full --junitxml=artifacts/acceptance-tests-full-local.xml`。两个runner通过py_compile，diff检查通过，14旧Draft重建无歧义。完整回归有一条既存python_multipart弃用警告。

## 2026-10-19 — 预计收益OT门控/动态长度（完成）

- 按用户要求保留Draft内部拼接，停止完整矩阵可靠性路线，新策略只沿用既有TailOT标量排序。正式执行冻结`7c10c5dad39ffd0d909ae577b87a8026d39e2523`，635个Git原始文件SHA核验；源zip SHA`fa8e5f843b3956ed51b662ff8be0b9aa1905c5f6559b478b280b8f562ca48d44`，实际package SHA`1d7947ea6371e8c54c63ff382f46f6eee9de413cd87531c3cf6ccbc6eabbdd43`。独立服务器`/media/a6000/D/ZhaoBinHe/MemoryDraft/ot-r-budget-v2`，流水PID2305668已结束，双GPU恢复19MiB/0%。旧结果含matrix全部递归SHA不变。
- 实际依次运行`python -m scripts.profile_benefit_costs`、`probe_benefit_offline`、`run_benefit_probe --stage mechanism`、`run_benefit_probe --stage timing`，完整配置/命令见`OT_BENEFIT_PROTOCOL.md`与`artifacts/run-benefit-pipeline.sh`（服务器）；输出`results/raw/ot-benefit/{cost,offline,mechanism,timing}`。共享基础配置`configs/runtime-ot-benefit.json`；跨项目校准数据由runner注入，不自动启用旧训练数据。
- context256/768/1536、候选15/16/24/32、2预热/7重复核测量；Draft约12.48–13.86ms，Target验证约59.64–89.97ms，共享shortlist/policy中位约3.368ms、每候选TailOT2.260ms。14状态/6项目的旧数据逐一排除当前整个项目：OT调用2，缩短0，回退Draft0；预计接受376.53，实际318，平均绝对误差10.47。策略/阈值/长度未按这些结果调参。
- 正式56/56机制、112/112未插桩交错计时均严格匹配原AR token/prompt/snapshot，112/112计时路径与机制计数相同。总时延Suffix14.177480s、Guided-Edit13.515785s、Splice-Edit13.631645s、TailOT13.614736s、Gate-OT13.828701s、Budget-Edit13.633580s、Budget-OT13.828517s。Budget-OT比Suffix低2.46%，比TailOT高1.57%，比Guided-Edit高2.31%；不把基础拼接收益归因于新策略。
- 门控OT候选求解31→7、调用9→2，但总验证65→67轮，实际Draft前向仍31；Budget-Edit与Edit相同66轮，三个收益组均没有真正缩短候选。shipment/22预测OT接受30.56实际6而Edit32；payment/15选Edit仅7，漏掉TailOT32。减少求解成立，条件收益预测不准；动态长度未体现额外收益。下一步应先补独立的廉价Draft前缀风险/接受生存预测校准，而非扩大当前邻居策略或在同批换阈值。
- 证据`results/ot-benefit`，报告`OT_BENEFIT_RESULTS.md`。原始归档170881字节/SHA`f2cac0ac41b755e5fb9ce62b67391b2579f481e8df2bedeaed05bce607a9298d`，包括首版250ab31的56行汇总失败现场；不含权重、DB、大型矩阵或私有历史。报告重建`.venv\Scripts\python.exe -m scripts.write_benefit_report`。跨Python3.11/3.13浮点sum出现1e-14量级差异，报告统计重算采用1e-12容差并保留原始记录字节；路线、长度、token IDs仍精确检查，Target验证完全未放宽。
- 正式服务器相关92 passed/8.16s；本机策略/报告15 passed/7.24s；最终完整340 passed/3 skipped/42.53s，命令`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-benefit-final --junitxml=artifacts/benefit-tests-final-full-local.xml`。首版审计失败与一次新增单测夹具错误均保存/说明，修复不改推理策略。
- `main`、`codex/infra-interview-v0.3`与标签`infra-v0.3-interview^{}`仍为`27e36199f8a65acdb9a9c42ccb744ddfa9a9402b`；默认suffix不变。本轮为开发检查，未增加独立holdout或通用性能声明。

## 2026-10-19 — 收益路由汇总口径修复

- 首版执行提交`250ab31a9f10dcf6f9c7ba470d3d5a9e5bc7c1bb`，服务器 `ot-r-budget`，634文件SHA核验；91项相关测试通过。成本/14状态离线阶段完成，56次生成均通过逐请求strict检查，但机制最终audit将拼接归因恒等式错误应用于Guided-Edit，抛 `Invalid benefit accounting`，未进入计时。
- 最小修复仅限定该恒等式检查到真正拼接组，新增包含Guided-Edit非零接受量的审计回归。策略特征、邻居、阈值、长度和运行范围不变。保留首版失败现场；以新冻结版本/独立 `ot-r-budget-v2` 重跑正式小检查，不把首版汇总失败混入成功计时。

## 2026-10-19 — 保留内部拼接，预计收益调用/动态长度（执行准备）

- 用户决定舍去完整矩阵可靠性路线；旧实验不删除，新路由不使用矩阵特征。新增 `retrieval/benefit.py`，用历史已验证接受长度预测 Draft/Edit/TailOT 三个动作的推进量，并按实测成本比较预计rate；OT需要比最佳廉价动作高2%才调用，单候选不调用；拼接长度16/24/32。保留精确suffix优先与原始Draft/历史span、严格Target验证和预算。
- 校准固定14个已观察状态；每次排除当前整个项目，其他每项目最多一个邻居，最近3项目加权。不读取当前答案作在线决策、不动态调参。因数据稀少且全部已观察，只是交叉项目开发验证，不称独立holdout。协议 `OT_BENEFIT_PROTOCOL.md` 写明局部rate/成本档/pending近似限制。
- 新增成本微基准、离线检查、七组GPU runner，以及runtime显式可选入口；默认suffix和独立infra版本不变。首轮发现缺失校准抛TypeError及pytest默认临时目录不可写，已改明确配置检查/项目内basetemp。随后15 passed；相关86 passed/2 skipped/13.88s，命令 `.venv\Scripts\python.exe -m pytest -q tests/test_benefit.py tests/test_runtime.py tests/test_dflash_reuse.py tests/test_splice_reuse.py --basetemp .test-tmp-benefit-2`。
- A6000双卡19MiB/0%，使用新独立 `ot-r-budget` 目录；先冻结源码、完整回归及成本测量，再固定八个旧请求×七组机制56次、交错计时112次。不用本轮结果反复换参数。
- 冻结前完整回归334 passed/3 skipped/38.90s：`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-benefit-full --junitxml=artifacts/benefit-tests-full-local.xml`。三份新runner通过py_compile；`git diff --check`通过。

## 2026-10-19 — OT完整矩阵三个小改动（完成）

- 冻结执行提交`81a0de103f95cf3c990f9fcbd4947276e70723c0`，601个Git原始文件SHA全通过；源归档SHA`7f89cbd431503d11d6c27feac409a6261456ae19b8f044b47447aae0db8445c0`，实际package SHA`a5e135401d0f7558a100d3cd1c4fa19a83e70c43c05e6099183c228df6011c33`。服务器独立`/media/a6000/D/ZhaoBinHe/MemoryDraft/ot-r-matrix`，相关94 passed/8.26s；原始已发布结果含两轮拼接保持递归SHA不变。
- CPU命令`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.probe_matrix_offline --tokenizer models/qwen3-8b --output artifacts/offline-v1`。135次旧调用及提议全部重现，14个去重可评分状态/39候选。相对原OT，B选择变化0，C4，组合2；同状态连续接受总和Edit344、OT318、Tail339、B318、C301、组合314。该总和不是在线新增token或速度，不作独立样本统计。
- 按`OT_MATRIX_PROTOCOL.md`实际运行64次机制检查、128次未插桩交错开发计时，192/192输出token/prompt/snapshot严格匹配原AR；128/128计数路径与机制回放相同。流水PID2285057已完成，双GPU恢复19MiB/0%；CPU/GPU阶段相同package，未按新评分改参数。所有任务早已观察，不是新holdout。
- 总时延Suffix14.164274s、Guided-Edit13.496111s、Splice-Edit13.627771s、原OT13.912373s、TailOT(A)13.609542s、OrderOT(B)14.088728s、PlanOT(C)14.051844s、All13.890790s。B相对原OT+1.27%、C+1.00%、All-0.16%；组合相对TailOT仍+2.07%、相对Guided-Edit+2.92%。新三个组没有超过已测TailOT，保留局部正负结果而不继续在本批换阈值。
- B总轮与原OT同为67但多一次求解；C提议9→5、选中却未接受历史的状态4→1，却误放弃两处池内可用接续，总轮67→69。具体旧JSON可接受32的候选连续性0.469被0.5阈值拒。runbook正确32-token候选连续性0.397被拒，错误候选保留14个Draft但仅接受7、连续性反而0.704。既有未校准阈值误拒，也有上下文对齐与Target接受的排序错位，不能只靠降低阈值宣称解决。组合总轮66，仍高于Tail65；C在JSON负例减少部分浪费但未胜过便宜Edit。
- 结果`results/ot-matrix`，报告`OT_MATRIX_RESULTS.md`。原始zip99659字节/SHA`f956951905b89cb2339117a45375d5324c1aba1fbfca4edbf6ea55b7f8dad532`，仅发布紧凑特征、调用与来源，不含大型矩阵/权重/数据库/私有历史。重建`.venv\Scripts\python.exe -m scripts.write_matrix_report`。
- 本机完整314 passed/3 skipped/34.15s；追加报告/状态审计16 passed/13.83s；补充逐候选解释后最终报告8 passed/10.70s，含真实CLI跨hash-seed证据字节保留。默认suffix、独立infra保持原版本；本轮未实现预计收益调用或动态长度，不把固定启发评分称为接受概率。

## 2026-10-19 — OT完整矩阵三个小改动（执行准备）

- 用户授权测试接续可靠性、完整矩阵与三个小改动。新增`retrieval/matrix.py`：A沿用已有尾部加权；B用初始正确匹配流的相邻行1–4步顺序支持修正真实token代价，再求解一次；C用锚点前末8行支持度/集中度/连续性/dustbin质量排序或放弃；组合A+B+C独立。均为固定词面OT启发，不是校准接受概率/硬单调约束/OPW复现。参数先写入`OT_MATRIX_PROTOCOL.md`，不按新评分结果调参。
- 旧`splice.py`仅抽出可覆盖的`candidate_score`，原Edit/OT/Tail公式不改；新增显式`ot-order-splice`、`ot-plan-splice`、`ot-all-splice`和`runtime-ot-matrix.json`。候选池、长度32、预算4/top8及严格Target验证不变，未实现按预计收益调用OT。
- CPU脚本先恢复原16请求及五请求修复的三种Splice实际状态，核对旧提议/用户时间快照后才进行同池逐候选评分；只保存紧凑矩阵特征，答案只用于事后连续接受评价。若新组产生选择/放弃差异，再固定八个既有请求×八组=64次GPU机制检查；达门槛才做128次交错未插桩开发计时。
- 首批48 passed/15.69s；相关91 passed/2 skipped/12.95s，命令`.venv\Scripts\python.exe -m pytest -q tests/test_matrix.py tests/test_matrix_probe.py tests/test_splice_reuse.py tests/test_runtime.py tests/test_dflash_reuse.py --basetemp .test-tmp-matrix-related --junitxml=artifacts/matrix-tests-related-local.xml`。新增同质量逆序/锚点自证排除/两次求解边际/同池Edit记录评分测试。完整回归正在完成。
- A6000双卡19MiB/0%，将用独立`ot-r-matrix`目录，全部已发布旧结果含冷却修复递归SHA保护，独立infra不覆盖。
- 冻结前完整回归314 passed/3 skipped/34.15s：`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-matrix-full --junitxml=artifacts/matrix-tests-full-local.xml`。未发生本轮测试失败；下一步以冻结Git原始文件部署CPU检查。

## 2026-10-19 — 报告跨平台重建的字节保留修复

- 首次最终交付在Linux重建首轮报告后，修复报告被旧证据SHA保护拒绝。确认诊断JSON语义完全相同，但字典键顺序变化：原SHA`cd81e60688775121a25ca2e4cffd57fa9557bd7b1f0acbf46303d4957e73b118`，重建SHA`bc3ab55d18d00ff1e68c7dac01ffd17d16e7662631e07c35e66938f7f047cd14`。不是模型输出或数值结果变化；原GPU目录和Git原证据不受影响。
- 两个报告命令现在对语义相同的诊断保留已有原始字节，新文件采用固定排序JSON；不改历史出版物。增加实际CLI子进程、不同hash seed/等价JSON序列化的回归。`python -m pytest -q tests/test_splice_report.py tests/test_splice_cooldown_report.py --basetemp .test-tmp-splice-rebuild --junitxml=artifacts/splice-report-rebuild-tests.xml`：12 passed/15.24s。只修复报告写入，不改已计时package、runner或协议；无需重跑GPU实验。

## 2026-10-19 — 拼接冷却最小修复（五请求检查完成）

- 修复执行提交`c98bb83e3e50a0ba1c01268317b074a2fcfcbbd4`，572个Git原始文件SHA全部通过，归档SHA`01568feab9c0d830900ec0bae95e8ac8f5b94feaa710e9645f50d803986bc545`。服务器`/media/a6000/D/ZhaoBinHe/MemoryDraft/ot-r-splice-cooldown`，预执行90 passed/8.20s。实际package两阶段同为`89063c6e3007034cb72823d5a42494f26258baf35b7cbf9beaa74dbd0ee2513c`；全部旧证据含首轮拼接SHA保持不变，不覆盖原执行源码清单。
- 实际命令见`OT_SPLICE_COOLDOWN_PROTOCOL.md`，使用`--cooldown-check --stage mechanism/timing`和独立`artifacts/mechanism-v1`/`timing-v1`。流水PID2194458完成；30/30插桩检查、60/60未插桩计时严格匹配既有AR，60/60轮数/接受/候选和Draft前向计数匹配插桩。双GPU恢复19MiB/0%。没有新holdout或进一步GPU扩展。
- 冷却边界修复正确，但只恢复部分机会：payment-runbook:1中TailOT总轮8→7，普通Edit/OT仍8；shipment-contract:1中Edit总轮9→8，两OT仍9。三个正例各方法轮数保持。剩余问题有直接记录：runbook位置15普通Edit/OT保留14个Draft但仅前7个正确，TailOT保留7个并接受32；JSON位置22 Edit接受32，两OT保留8个但只接受6。可匹配内部锚点不保证其前面的Draft正确，不把剩余失败继续归于冷却。
- 五请求两重复总时延Suffix8.926047s、Guided-Edit8.482442s、Guided-OT8.522107s、Splice-Edit8.540418s、Splice-OT8.780651s、TailOT8.474506s。TailOT相对Guided-Edit仅-0.09%（基本持平），普通OT+3.52%。SQL两请求均保留各少1轮的局部收益，Edit同样有效；旧JSON TailOT保持单例少1轮。不能外推修复后的16请求整体性能或宣称稳定OT优势。
- 结果`results/ot-splice-cooldown`，报告`OT_SPLICE_COOLDOWN_RESULTS.md`。原始zip65857字节/SHA`01c55d562ad25f7418265a68e6280adba3a8582c7a6abfbe5f3cc07a7eab2370`；报告重建`.venv\Scripts\python.exe -m scripts.write_splice_cooldown_report`。追加两个报告/来源审计18 passed/5.77s；补充逐位置解释后最终10 passed/5.51s，原完整290 passed/3 skipped记录保留。main、infra分支和tag仍均`27e36199f8a65acdb9a9c42ccb744ddfa9a9402b`。
- 本阶段结论：保留内部桥接和尾部加权为显式研究选项，默认suffix；若继续，应先离线解决保留Draft前缀的风险和同状态排序，而不是再扩大同类GPU负载。

## 2026-10-19 — 拼接前缀拒绝的冷却修复（五请求检查准备）

- 首轮代码/原始证据已独立提交`0000e5c`/`bc6b39e`。共享解码器只改一行冷却条件：`draft_prefix <= accepted < len(proposal.tokens)`时才冷却；尚未尝试历史的Draft前缀拒绝不触发冷却，普通R/原Guided默认prefix=0仍保持规则。匹配器此前已避免把前缀拒绝映射成历史端点失败。
- 协议`OT_SPLICE_COOLDOWN_PROTOCOL.md`固定两个负例、两个SQL正例、一个旧JSON正例；runner新增`--cooldown-check`限定这5个已观察请求，六组30次机制/60次未插桩开发计时，不做新场景确认。首轮`results/ot-splice`也加入递归SHA保护；服务器新目录`ot-r-splice-cooldown`，证据独立`results/ot-splice-cooldown`。
- 新增冷却0/1、Draft前缀0/1位拒绝、首个历史token拒绝、EOS等真实小Target回归，相关56 passed/2 skipped/10.78s；完整290 passed/3 skipped/25.00s。命令`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-splice-cooldown-full --junitxml=artifacts/splice-cooldown-tests-full-local.xml`。仅相关策略被修复，未改检索阈值或Target验证。

## 2026-10-19 — Draft内部锚点拼接（首轮完成，定位冷却缺陷）

- 冻结执行提交`0000e5c9f78df0d762d2dba981daafa8a59bd2bb`，550个Git原始文件逐项SHA通过；源归档SHA`c09491b515eddd94b4895a0e5b6b1a8daf28e20c353a9ff51ff7482c1900fa96`。服务器隔离目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/ot-r-splice`，执行前相关72 passed/7.52s。实际机制/计时package同为`44f733dea98734d16c18a3b447207a197fba1e7ec4e14255b36e4e376048cd1a`，runner/协议/全部旧证据SHA未变。
- 16个已观察请求×6组一次机制96/96严格一致。三个Splice均在inventory-query两个请求各少1轮，达到预定小成本门槛；收益只有一个项目，不是独立场景确认。按同一提交进行192次未插桩交错计时，192/192输出token/prompt/snapshot严格一致，192/192轮数/接受量/候选与机制回放相同。实际命令见`OT_SPLICE_PROTOCOL.md`；机制PID2185996、计时PID2187541均完成，双GPU恢复19MiB/0%。
- 总时延Suffix26.000739s、Guided-Edit25.018151s、Guided-OT25.153341s、Splice-Edit25.285204s、Splice-OT25.421422s、TailOT25.258074s。三个Splice分别比Guided-Edit慢1.07%/1.61%/0.96%。SQL项目两重复四观察：Guided-Edit3.221949s，Splice-Edit3.014994s；三种拼接都恢复32token，第二请求来源是当前prompt，不能归为历史独特价值。普通OT/Edit总轮/接受量相同，15/16请求完整候选token调用序列相同；TailOT在一个旧JSON请求少1轮，总轮119→118，但整体仍无净优势。
- 逐轮定位退化：payment-runbook:1位置11保留11个Draft token但仅接受3；shipment-contract:1位置14保留9/15但仅接受7，均在Draft前缀内拒绝。原D也接受3/7，但Splice被记为R触发1轮冷却，跳过下一轮位置15/22的32token机会，各多1轮。保存首轮原始负结果，下一步只修复此反馈策略并检查5个既有关键请求，不改Target或调检索参数。
- 证据`results/ot-splice`，报告`OT_SPLICE_RESULTS.md`；原始zip80948字节/SHA`cf741ecfd7088c84de6f319a405b6472b3bd9098b4b56d7d8fb913109ca7f17f`。报告/归因审计14 passed/5.41s，追加解释后纯报告6 passed/4.91s；完整272 passed/3 skipped和原始XML均保留。重建`.venv\Scripts\python.exe -m scripts.write_splice_report`。

## 2026-10-19 — Draft内部锚点拼接（小实验准备）

- 用户授权小成本实验。新增`retrieval/splice.py`与`edit-splice`、`ot-splice`、`ot-tail-splice`显式选项，复用现有DFlash post-draft hook；保留Draft到内部4-token锚点的前缀，再拼原始历史尾部，最多32token。三个Splice使用同一Edit去重/top8池、4次查询预算、0.65阈值；TailOT仅末8位置加权，不声称单调OT。原检索/解码算法和默认suffix不改，独立infra版本保留。
- `OT_SPLICE_PROTOCOL.md`预定16个已观察请求×6组一次插桩机制检查；同一个Splice至少2请求比两个Guided基线都少轮且存在内部连续接受增益，才运行相同实现的两重复未插桩开发计时。无新holdout，不据此宣称泛化。参考仅生成后评分，全部旧证据递归SHA保护，不写回答案。
- 相关测试首批62 passed/2 skipped/35.00s；加入真实小Target拼接反馈、EOS/KV及来源/归因审计后70 passed/2 skipped/13.21s。命令`.venv\Scripts\python.exe -m pytest -q tests/test_splice_reuse.py tests/test_splice_probe.py tests/test_dflash_reuse.py tests/test_runtime.py --basetemp .test-tmp-splice-final --junitxml=artifacts/splice-tests-final-local.xml`。
- 服务器已连通，双A6000各19MiB/0%。将部署独立`/media/a6000/D/ZhaoBinHe/MemoryDraft/ot-r-splice`，沿用父目录模型和`.venv-dflash`，先完整测试再冻结执行源码。尚无本轮GPU结果。
- 冻结前完整回归272 passed/3 skipped/25.25s，命令`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-splice-full --junitxml=artifacts/splice-tests-full-local.xml`；新增模块/脚本compileall与`git diff --check`通过。

## 2026-10-19 — DFlash前缀引导的OT/Edit（完成）

- 执行提交`6f9303625ddf01b1c82938b2f1184a08e7c83141`，517个Git原始文件部署SHA逐项通过；原始归档SHA`3799e0d8649a8e0d85dbe0b3010afbbef7dbea965955ba8e515fe5572dbdb9de`。服务器隔离目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/ot-r-guided`；开发/新验证实际package同为`58dc6fdbd73ca3d404b521147ad3e8227df5f05bc022fda79084fe79ae88a466`，runner相同，未在新答案上调参。任务SHA`0097a10eaa102db6c866326ca145c3842657103ee400c892c110b99e4cf2c2b4`，协议SHA`0f1eca2375b6e5bf19023cabd87d403913565d541597445b33b569eb4a1cdc80`。
- 开发命令`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_guided_validation --stage development --config configs/runtime-dflash.json --output artifacts/development-v1`。PID2136237退出，32/32输出token/prompt/snapshot严格一致；两旧长正例Guided-OT/Edit均保留32token接受。旧插入负例的长接续恢复；短输出/未完成新内容误调用减少。开发计时不当独立确认。
- 新验证命令`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_guided_validation --stage holdout --config configs/runtime-dflash.json --development artifacts/development-v1 --output artifacts/holdout-v1`。PID2137526退出，先16/16新AR契约正确/EOS，再160/160五组两重复严格一致。全部在线成本计入，历史不写回，旧结果SHA不变，两GPU恢复19MiB/0%。
- 新主场景：DFlash16.784681s、Suffix12.724529s、原OT12.928572s、Guided-Edit12.400227s、Guided-OT12.421961s。引导OT比Suffix低2.38%、比原OT低3.92%，比引导Edit高0.18%；对Suffix仅2/4项目更快，项目bootstrap比值[0.9483,1.0025]，预定工程门槛未通过。新全部请求：引导OT较Suffix低1.25%、较原OT低3.77%；control较Suffix高0.74%。这些是作者受控任务，不宣称生产泛化。
- 主场景总轮120→114，实际Draft前向50→50；引导R已经支付Draft，不能把D路线44说成只算了44次Draft。较长接续减少Target验证轮。引导OT/Edit在新32/32请求-重复配对的轮数、接受量、调用/候选/提议和Draft前向相同，均256对齐接受token；主场景各192。没有新证据支持OT排序增量；TOML/JSON两个项目的局部收益为事后格式拆分，不能升级为独立场景确认。
- 本机最终完整回归242 passed/3 skipped/24.53s，服务器执行前相关61 passed/7.16s；追加报告篡改/配对/来源审计最终11 passed/5.27s。原始失败和修复XML一并保存。报告命令`python -m scripts.write_guided_report`，Windows/Linux正文SHA均`b1924812d8d974940f27bbbad96f3131c35e473d981b0b3dddd792d79c9aaaca`，diagnostic SHA均`41bc4d7b55c3a6ab326b5cf7991a0e8aa784de28ca7839fef72cd8e943dbf000`。
- 结果`results/ot-guided/{development,holdout}`，报告`OT_GUIDED_RESULTS.md`；原始下载zip65728字节/SHA`4b85b66e58f1837a5da8e3e8334a8c028fbc237e4893df9e453cf027b1e58d64`。只发布紧凑受控任务结果、来源与测试，不含权重/数据库/私有历史。保留前缀引导为显式工程实验选项，默认suffix，独立infra面试版不改；不扩大同机制OT试验或把引导收益归为OT独有贡献。

## 2026-10-19 — DFlash前缀引导的OT/Edit（执行准备）

- 用户授权进行下一步。新增`retrieval/guided.py`，精确suffix仍优先；未命中后复用已有DFlash前4个候选作为接续锚点，最多64后缀项/8候选/4查询，候选需长于现有Draft块。OT/Edit使用同池同预算，未改Target输入或逐token验证。`draft_forward_rounds`单列实际Draft计算，避免把引导R计为跳过Draft。旧`transport.py`不改，默认仍suffix，独立infra保留。
- 机制参考LogitSpec（arXiv:2507.01449v3）预测token引导检索与SAM-Decoding模型/检索组合；没有复制新增上游代码。固定协议`OT_GUIDED_PROTOCOL.md`，先旧8请求×4组一次开发，再新8项目16请求×5组两重复验证。新任务在首个新Target答案前生成，全部方法共享给定源文，不写回当前输出。任何新AR契约不符或截断均停止正式计时。
- 首轮相关测试51 passed/2 skipped/2 failed，发现suffix库无匹配返回空位置，已补`count==0`处理；随后53 passed/2 skipped/11.78s。新增fixture/audit检查曾因tuple/list直接比较失败，已仅在测试中标准化JSON表示，任务字节不变；对应6 passed/0.65s。首次完整回归241 passed/3 skipped/1 failed为同一fixture测试，修复后将运行完整检查。原始XML在artifacts/guided-tests-*.xml保留。
- 服务器双A6000空闲；下一步部署独立`ot-r-guided`目录，不覆盖原诊断/实验/infra。完整在线成本纳入，hint只是未确认候选，不作参考答案或已确认上下文。

## 2026-10-19 — OT-R 在线错误定位（完成）

- 固定执行提交`8a40def7e0b87a3d5090a5d11201ff90c2fd4d33`，491个Git原始文件部署逐项SHA通过；归档SHA`cd8286531f1d6cc9f9d72f90fdc4ec6e0789611aa7bd4fe6daa0b70695aa1e4e`。服务器新目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/ot-r-diagnosis`，仅使用既有模型与`.venv-dflash`，原实验和infra不覆盖。首次部署检查遇系统Python不支持`Path.is_relative_to`，尚未解包/启动模型；改用既有Python3.11后校验通过。
- 实际命令`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.diagnose_transport_online --config configs/runtime-dflash.json --output artifacts/diagnosis-v1`。PID2116743已退出，24/24输出token/prompt/snapshot严格一致；全部计数与原未插桩两重复逐项一致。执行package SHA`35603a53734dde44673730c53abacbe5756e27dab24534149742dcd3120bcfc5`，原GPU package文件全保持原SHA。参考答案仅在生成后离线评分，生产检索/验证代码未修改。
- 新五请求六次OT调用：四次合法原始span最好也不足8token、一次pool漏召回、一次32token全接受。parcel-queue新值/新字段尚未完成即复制旧文本；同状态DFlash能推进更多。payment-runbook插入Region行后的长Severity接续不在池内，三个稀有新词只命中修改指令，早期Owner锚点受当前倒排最小上下文范围限制，固定±2偏移未能恢复端点。没有将该组问题简单归因于OT池内排序。
- 单点事后召回反事实：候选数8→32仍无长span；偏移2→5后池中出现32token，但OT/Edit仍选0token。该CPU诊断不改变线上参数，也不作速度/独立验证证据。可复算`python -m scripts.diagnose_transport_shortlist --source results/ot-diagnosis --tokenizer models/qwen3-8b --output artifacts/shortlist-analysis.json`，服务器重建JSON语义完全相同。
- 原未插桩逐请求拆分：payment-runbook:0确实OT较suffix降低4.78%延迟（Edit也成功），另一个插入请求增加22.09%，整体平均掩盖了局部成功。confirm-schema-0:0的Edit在端点129/130分数同为0.9375，按位置取早端点接受0；OT约0.934891/0.941865，选130并接受32。旧正例有真实消歧，但不证明OT优于更好的便宜平分规则。不能用强制末token精确匹配作简单保护，该成功OT的exact_tail=0。
- 本机完整回归216 passed/3 skipped/35.09s；服务器执行前22 passed/2.37s；追加报告/比较/篡改拒绝后本机17 passed/5.32s。命令和XML见`results/ot-diagnosis/tests-*.xml`。报告复算`python -m scripts.write_transport_diagnosis_report`，Windows/Linux正文SHA均`f7088bd4203c82120a23bc5e6ed2b9396b721463bcbac7ff9da46cbc0a8c976e`、diagnostic SHA均`2019028e0829c1a946fb300d2e967d8b6f21f66cb8b0924a9a190c4b38442003`。压缩的紧凑诊断与来源清单保存于`results/ot-diagnosis`，不含权重、数据库或私有内容。
- 结论：先减少编辑区/结束处无收益调用，再处理插入后端点召回及接续排序，保留旧正例回归；求解器提速排在后面。本阶段完成错误定位，未实现新路由，也未重新宣称性能提升。两张GPU已恢复19MiB/0%。

## 2026-10-19 — OT-R 在线错误定位（执行准备）

- 用户要求先定位错误。新增独立诊断模块/脚本和协议`OT_R_DIAGNOSIS_PROTOCOL.md`，只包装既有解码入口捕获实际提议与轮记录；生产检索/解码文件不改。范围为新验证五个实际触发请求、首轮两个局部正例及copy负例，八请求×Suffix/Edit/OT一次，共24次插桩运行，不作性能测量。
- 输出先与已保存AR的token/prompt/snapshot严格核对，再离线重放同预算、阻塞状态下的实际提议；分开枚举合法原始span、合格端点及实际候选池。参考答案只用于事后评分，>=8 token是诊断分层而非收益阈值。
- 本机命令`.venv\Scripts\python.exe -m pytest -q tests/test_transport_diagnosis.py tests/test_transport.py --basetemp .test-tmp-ot-diagnosis --junitxml=artifacts/ot-diagnosis-tests-local.xml`：22 passed/14.70s。服务器双A6000均19MiB/0%，将用新隔离目录运行；结果尚未产生。

## 2026-10-18 — OT-R新场景验证（完成，按门槛停止）

- 修复后在`/media/a6000/D/ZhaoBinHe/MemoryDraft/ot-r-validation-v2`运行，冻结提交`45ec2ef995bd7521c579c2fe998fcd35c9c20c31`，PID1658928已结束，两GPU恢复19MiB/0%。24个新AR输入/快照核对、192/192计时逐token严格一致；两次24个AR的tokens/text/prompt/snapshot/finish_reason/contract逐项相同。任务22/24正确，两个格式控制失败；两个候选适用组均8/8正确，无截断，不剔除失败请求。
- 四组总秒数DFlash43.902004、Suffix35.750338、Edit36.243279、OT36.375919，OT相对Suffix+1.75%、相对Edit+0.37%。local-edit分别13.357009/13.752354/13.904043（Suffix/Edit/OT），OT+4.10%；ambiguity15.294088/15.276154/15.249713，OT-0.29%，但对齐调用0且解码计数相同，不能归因于OT；control7.099240/7.214771/7.222163，OT+1.73%。两个预先指定候选场景均未通过工程门槛，停止扩大同类测试/在本批答案上调参，保留显式实验选项，默认suffix。
- OT/Edit的48个请求-重复配对中D/R轮数、R接受量、对齐调用/候选/提议/接受量全部相同；两者对齐均接受66token，查询0.211308s对0.005951s。仅parcel-queue/payment-runbook/short-settings三个项目触发对齐。Suffix→OT：D146→142、R180→190、总326→332，R接受4366→4388。66个对齐接受token不能当成相对Suffix新增66token，更不能据此声称加速；新增验证与查询没有换来净收益。
- 实际命令`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_transport_validation --config configs/runtime-dflash.json --output artifacts/ot-validation-v2`。实际package SHA`7201d6615cd38402d74aefb82c6d941e9048863c65bd8271fa5b1eca2a3ce03c`，首轮42个既有package文件SHA全部保持；本轮只新增评测模块。461文件部署逐项核验。运行前任务/协议来自226399c，修复仅评测字段；失败目录和原执行清单均保留。
- 本机修复后完整`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-ot-validation-final --junitxml=artifacts/ot-validation-tests-final-local.xml`：205 passed/3 skipped/20.71s；服务器87 passed/8.03s。之后纯报告与失败参考审计新增检查，本机23 passed/2.20s，实际XML另存。冻结修复提交CI成功39141490756。原始下载ZIP103252字节，SHA`40f9e589e238980593dbd19c1c1bef200c883e18e662d83ac3cdef84455e81df`。
- 原始计时/参考/失败记录、执行清单、测试XML和发布SHA保存在`results/ot-validation/`，报告`OT_R_VALIDATION_RESULTS.md`，复算`python -m scripts.write_transport_validation_report`。这些是12个作者编写受控项目，不能当生产流量或宣称OT普遍无效。完整infra面试版本423文件及三个固定ref重新核验仍为27e3619，研究不覆盖main或原infra服务器目录。
- 最终报告审计两平台各23 passed（本机2.14s、服务器1.41s），独立重建正文SHA均`c9572d3f26ac99d6526d2fee018c5b97fef59036ff09ff74aa4f4607c6da0545`，diagnostic SHA均`7071844474062ce4c6f77f581e794feb43852e05fe586b317820e8001380c322`。发布24份小型证据加publication.json，总约296KB；最终package与实际GPU package逐文件完全相同。之后只有报告/测试/交付文档更新，执行源码包、原失败包和最终交付清单分别保留。

## 2026-10-18 — OT-R新场景验证（运行前冻结）

- 用户授权开展上一轮提出的24新请求验证。新增12个作者编写的受控项目，每项目两次独立v1修订，分local-edit/ambiguity/control各4项目；不是自然生产流量或12个真实外部仓库。24请求、13条合法旧记录，未生成Target答案、没有把新答案写入历史。
- `python -m scripts.make_transport_validation --output data/research/ot-validation/cases.json`已导出固定任务，SHA`6dea703af88c9e428d130dd967d81a21662713acac77f50dff97531a1c34e802`。保留全部当前源文及明确编辑指令，expected_text只用于任务评分，不进入检索。512预算、24新AR、四组两重复共192计时；原OT/edit参数、Target输入/验证保持固定。
- 协议`OT_R_VALIDATION_PROTOCOL.md`在输出前冻结：适用候选组至少5%优于Suffix、2%优于Edit，分别3/4项目更快，8/8任务正确且无截断，才通过继续投入工程门槛；区间单独报告，不当统计显著性。先测试/audit，再部署隔离研究目录。默认suffix、面试分支/标签/main及服务器infra-v0.3均不改。
- 本机新评测审计21 passed；完整`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-ot-validation-full --junitxml=artifacts/ot-validation-tests-full-local.xml`为204 passed/3 skipped（21.92s）。最初一项测试把Python tuple直接与JSON list比较而失败，规范化比较后通过，任务文件未改。首轮相关44项中其余43项当时已通过。原Starlette multipart弃用提示未改。运行脚本另在首个严格分歧保存failure.json并停止；启动时核验首轮全部package文件保持原SHA。
- 首次冻结`226399cb4c7ad14cf2fe85a891f6b25d8f78307f`，461文件服务器SHA一致，86项服务器测试输出通过（SSH在测试打印结束后的10秒边界超时，XML已保存），GitHub CI成功39141060160。A6000 PID1654831在24个AR参考后、第一条计时结果审计时暴露新评测字段误用：`GenerationResult`真实字段是`finish_reason`，不是`stop_reason`。该运行status=failed、有效rows=0，目录`ot-r-validation/artifacts/ot-validation-v1`完整保留，不计入性能结果。仅修正评测字段并增加真实运行时数据结构回归；已看过AR的22/24任务正确（两格式控制失败），但任务/输入/参数/门槛均不修改，重新冻结并在新目录运行全部AR与计时。

## 2026-10-18 — 独立保存 infra 并接入 OT-R（首轮完成）

- A6000 PID1622185已结束，两GPU恢复19MiB/0%。10/10新AR与旧token/prompt/snapshot一致，独立耗时130.186340s；随后80/80配对计时全部严格一致。旧JSON：DFlash28.028885s、Suffix-R22.584828s、Edit-R22.479903s、OT-R22.171767s；OT比Suffix降低1.83%、比edit降低1.37%。收益集中在事后拆分的confirm-schema-0（两请求各两重复）：4.759366→4.320043s、-9.23%；其余三项目没有明确改善。不能把已看过数据上的子组当独立确认。
- 旧JSON里Suffix/Edit/OT的Draft轮数82/76/72，R轮数164/168/166，对齐接受token分别0/64/128，OT额外查询0.117405s计入总时间。copy负例Suffix1.373279s、OT1.532735s（+11.61%），两组Draft均10轮，OT额外2个提议接受0token；不能仅凭减少Draft轮数或离线覆盖宣布加速。全部十请求总计Suffix27.747403s、OT27.489606s（-0.93%）。默认仍suffix，不在旧任务上继续调参或扩大重复。
- 运行前代码/协议已提交推送`f30b60a7e908307c5f398b88ffb28a1c5eead712`，433文件部署逐项SHA通过。实际GPU package SHA`27070f0f5d720641ef8f250c59188e33c3ab4bc9a8c5ac5f4169d0dc2367df43`，与本机离线分析及最终package相同；后续只补报告/测试与交付文档。服务器命令：`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_transport_probe --config configs/runtime-dflash.json --output artifacts/ot-r-v1`。隔离目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/ot-r-probe`，原`infra-v0.3`未覆盖。
- 本机最终完整`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-ot-final --junitxml=artifacts/ot-tests-final-local.xml`：183 passed、3 skipped（19.18s，原Starlette multipart弃用提示）。服务器相关65 passed（6.55s）。报告LF输出补充检查两平台各11 passed，原始XML分别保存。`python -m build --no-isolation`因本机未安装build前端失败；替代pip wheel首次遇Windows沙箱临时目录权限，正常临时目录下`.venv\Scripts\python.exe -m pip wheel --no-deps --no-build-isolation . -w artifacts/ot-wheel`成功，wheel SHA`82ca3f6f07566eaf16795e2abe7631de74e62f8e37168c3f7fa0cefc459e0727`。没有因此改代码或安装额外build前端。
- 结果发布在`results/ot-r/`，报告`OT_R_RESULTS.md`。两平台`.venv-dflash/bin/python -m scripts.write_transport_report`独立重建正文SHA均`6bb035227c3e4764996e2a78d5df79846f870ed5a7a297da8a792d04b7d862e6`、diagnostic SHA均`c84ae2a673686e547901aec192b5714ae1dbea2477624ac52db32cd25bfde832`。首次JSON字节核验暴露Windows CRLF，数值与正文已相同；仅报告生成器统一LF后核验通过。原始计时/模型输出未改；旧输入所有SHA保持原值。模型、数据库和大型trace不提交。下一步若继续，是未参与本轮调参项目的独立机制验证和退化保护，不宣称OT已能普遍替换R。

- 用户授权尝试 OT-R，并要求单独保留面试项目。infra 固定提交 `27e36199f8a65acdb9a9c42ccb744ddfa9a9402b`，独立分支 `codex/infra-interview-v0.3`、标签 `infra-v0.3-interview`，研究分支 `codex/ot-r-probe`。本机 `artifacts/infra-interview-v0.3/project` 为独立 Git 克隆；ZIP/bundle/逐项 SHA 清单另存，423 文件原始 Git blob 字节核验通过。复现命令 `.venv\Scripts\python.exe -m scripts.archive_infra --output artifacts/infra-interview-v0.3`；归档位置与面试范围见 `INFRA_INTERVIEW.md`。服务器既有 `infra-v0.3` 不覆盖。
- 新增带预算的 `retrieval/transport.py`，沿用精确后缀路径，失败后调用 lexical OT 或同池 Levenshtein；默认运行配置仍为 suffix。OT 复用 MIT POT 0.9.6.post1，上游标签 commit `85113e9a380f5fcf684c50c73c1ff6a164a7366e`，没有复制上游源码。32-token 窗口、8 候选、双 dustbin、位置先验、4 次调用上限在首轮输出前固定；不是完整 OPW 或 SpecTr，不声称语义检索或新验证算法。实际短锚点保留在 `matched`，新增上下文合格标记只决定是否提议，Target 接受条件仍为逐 token greedy 相等。
- CPU 首轮 `.venv\Scripts\python.exe -m pytest -q tests/test_transport.py tests/test_dflash_reuse.py tests/test_runtime.py --basetemp .test-tmp-ot-first`：50 passed/2 skipped；补充运行配置与配对审计测试后相关 60 passed/2 skipped，XML 在 `artifacts/ot-tests-ready.xml`。含真实 tiny Target 的 OT 拒绝/KV/流式正确性及用户、未来历史、特殊 token 边界检查。
- `.venv\Scripts\python.exe -m scripts.diagnose_transport --output artifacts/ot-r-offline-v1` 已完成，只加载 tokenizer。对旧 8 JSON 与 2 infra 请求的所有长度>=8前缀作独立决策，不模拟在线4次预算/路由。旧 JSON 60 个后缀缺失位置，原始/候选池 oracle 均有48个>=8-token机会；OT 找回48，edit找回47，但 OT 词法总正确前缀1443、edit1491，OT查询更贵。copy 的22个缺失位置原始/池均有12个机会，OT找回0、edit6，OT零接受提议19/22。离线不是在线接受率或加速，参数保持原样，后续只进行协议限定的80次GPU计时。
- 服务器已仅新增 POT 0.9.6.post1，NumPy/SciPy 等原依赖未升级。GPU试验将运行独立 `ot-r-probe` 目录，同权重/FP32/eager/4线程、10个旧AR先核验、四组两重复交错、全在线成本计入；协议 `OT_R_PROTOCOL.md`。未将 CPU 机会结果冒充实测性能。

## 2026-10-18 — 旧正结果向infra迁移的配对验证（完成）

- A6000正式运行完成，PID1596891已退出，两GPU恢复19MiB/0%。原8个完整新AR逐token/prompt/snapshot一致，单列125.609s；之后120/120计时全部一致。Legacy-D=42.030658s、Legacy-R=33.879012s、Runtime-D=42.060590s、Runtime-R=33.881685s、Runtime-R-aligned=33.875867s。新版D/默认R=1.241396，[1.198093,1.291931]，延迟降低约19.45%；旧收益保留，不支持此前短样例退化是工程化破坏旧负载的解释。
- Legacy-D/Runtime-D累计均498轮Draft；三个R组均246轮R、123轮Draft、5418接受token，逐请求/重复计数24/24相同。对齐候选的tokens/记录边界/来源指纹8/8相同；默认system资源增量在本批没有改变这些计数，默认/对齐耗时比1.000172、区间跨1。Runtime-D/Legacy-D=1.000712（约+0.071%）；对齐Runtime-R/Legacy-R=0.999907、区间跨1；预设10%工程预算均通过，不宣称零开销或统计等价。
- 实际命令（服务器infra-v0.3目录）：`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_runtime_migration --config configs/runtime-dflash.json --legacy artifacts/runtime-migration-legacy --output artifacts/runtime-migration-v1`。在新输出前冻结协议/代码的提交为`06c3c4e3effb804eefce9ee7368b0138144aa255`；部署411文件逐项哈希一致。GPU实际package SHA`94340a2485f8ab5778a58da4bdda113ae374e8de3b0026be4de73437790ab738`，脚本/协议/旧源码逐文件SHA保存在provenance及execution-source.json；原确认11文件运行前后SHA不变。
- 新增离线报告及参考/资源/汇总篡改拒绝测试，未修改生产解码/检索策略。最终本机`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-migration-final --junitxml=artifacts/runtime-migration-final-local.xml`：156 passed/3 skipped/17.60s；服务器报告与原审计相关16 passed/3.03s。Windows/Linux独立重算diagnostic逐字一致，报告SHA均`2f42fcf242db68fa5acdb221b78958abd92e2af7a9bbd52c0041f65f1cd319fb`。实际测试XML发布，不把测试模拟数据当性能证据。
- 结果`results/runtime-migration/`，报告`RUNTIME_MIGRATION_RESULTS.md`，复算`python -m scripts.write_runtime_migration_report`。保存原始/压缩SHA，不提交数据库、权重或大trace。该批仍是原合成工作负载，不能重新证明历史必要性；之前infra两短请求的负结果不改写。后续退化保护应独立开发/验证，不根据本批事后改策略冒充本次迁移结果。

- 用户授权继续验证。本轮先隔离工作负载与迁移开销，不修改R路由。冻结协议`RUNTIME_MIGRATION_PROTOCOL.md`：原4项目8请求/原合法历史，五组Legacy-D/Legacy-R/Runtime-D/Runtime-R/Runtime-R-aligned交错3重复，共120计时，先新AR8例严格匹配原参考。
- 发现新版候选索引额外包含system文本，加入只对齐候选资源的控制组，Target输入始终相同。旧解码从`dd0077f`恢复，SHA`e8a697c2397b127c78db8db7dab10b42f9b1cf9f17a65e10c6bd2e43522a7099`与原确认清单一致。索引差异最初误判为换行，恢复校验失败；进一步核对原服务器归档与Git，确认后续新增`include_user_records=False`，旧`f43239a`索引SHA`9d15c30963a1194b9ad7f58c27579d6dbb68396e49a73bebc500715080bea9be`严格匹配原运行。已在GPU运行前修正协议，并加入逐例资源指纹核对；共享matcher与请求定义SHA相同。
- 新增`evaluation/runtime_migration.py`、`run_runtime_migration.py`与快照/配对/篡改审计测试。初步命令`.venv\Scripts\python.exe -m pytest -q tests/test_runtime_migration.py tests/test_reuse_report.py --basetemp .test-tmp-runtime-migration --junitxml=artifacts/runtime-migration-tests-local.xml`：13 passed/7.33s；脚本编译检查通过。A6000两GPU空闲，模型及TF4.57.3环境可用。GPU结果将在实际完成后追加，不用旧耗时冒充新结果。

## 2026-10-18 — v0.3 单GPU推理 infra（工程交付）

- 最终GitHub CI已确认：提交`c37f3206cd8136f6db321c9fa9325294e25ceb2f`的research和DFlash两个job均success，见[Actions 39129849841](https://github.com/2223434778/MemoryDraft/actions/runs/39129849841)。Linux原冻结输入针对性2 passed/4.18s。服务器完整404文件发布SHA逐项一致，独立重建`INFRA_VALIDATION.md`字节相同，package source与delivery.json逐文件相同；两张GPU空闲。该最终文档补记不改变已验证运行代码。

- 首次GitHub Actions实际执行：DFlash profile通过测试、构建和隔离wheel运行；research profile为145 passed/3 skipped/1 failed，定位旧冻结history/request输入在Linux checkout被规范成LF，而原GPU审计SHA绑定CRLF。仅将这两份原工作区输入按原始字节固定为`-text`，不改JSON内容或既有审计期望。首次发布归档也受Windows Git的换行导出影响，改为从Git blob原始字节构建归档并核对全文件SHA；不忽略错误或放宽哈希条件。

- 用户明确授权将科研原型转为完整infra。新增`memorydraft/runtime/`的请求/结果协议、配置、模型加载、AR/DFlash/R+DFlash统一生成、配对benchmark、可比性检查与退化阈值、CPU/CUDA profiler；旧H/D/RHD与研究证据保留。运行时DFlash是关闭R的同一审计适配器，原生控制仍在研究脚本。未改变Target prompt或接受条件。
- 新增`memorydraft/serving/`：单执行worker、有界FIFO队列和token buffer、429、deadline、断连/排队取消、慢客户端背压、异常恢复及健康/指标接口。SSE只发送Target已确认token，最终统一解码文本。取消不提前释放worker；异常清除traceback/context/cause，避免保存GPU KV等局部资源引用。测试覆盖失败后下一请求恢复与资源释放。
- 基础包不依赖GPU库，`research`/`dflash`分别固定TF4.52.2/4.57.3；服务声明FastAPI0.111.1、Uvicorn0.30.2、Pydantic>=2.10。加入GitHub双profile CPU CI、CPU demo Dockerfile、示例、安装/运维/API/计时文档。README改为工程入口，原README保留为`RESEARCH_README.md`，未改写原实验结论。项目版本0.3.0，无统一新开源许可。
- 最终本机命令`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-infra-final-full`：**146 passed、3 skipped、17.19s**；服务器`.venv-dflash`相关**54 passed、6.79s**。测试计数摘自实际工具stdout，原始测试log未落盘。Windows受限环境阻止asyncio内部loopback，曾在TestClient启动前挂起；在允许loopback的执行环境通过，不以跳过HTTP测试规避。只清理了本任务挂起的pytest进程。
- wheel命令`.venv\Scripts\python.exe -m pip wheel --no-deps --no-build-isolation . --wheel-dir artifacts/wheels-final`。在`artifacts/wheel-env`干净环境仅离线安装wheel，无torch/transformers/fastapi；基础infer在仓库外运行，最终demo benchmark成功，明确synthetic=true。Docker CLI不可用，未实际构建Dockerfile；CI定义提交后状态以Actions为准。
- 服务器隔离目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/infra-v0.3`，模型通过只读使用原模型目录，未重复下载。正式命令`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.validate_infra --config configs/runtime-dflash.json --output artifacts/acceptance-v1`。初次归档漏带`scripts/__init__.py`，启动在加载模型前失败；补齐后正式运行完成，首次错误见工具输出，发布日志为后续正式运行，不计失败尝试为GPU计时。
- 真实Qwen3-8B+DFlash、A6000 GPU0、FP32/eager：2请求×3方法×2重复，**12/12**计时token/prompt/snapshot严格一致；AR9.230410s、DFlash4.489956s、R+DFlash4.966973s，分别为AR的2.056×/1.858×。本轮R+DFlash反而比DFlash慢约10.6%，不宣称检索净收益或独立自然工作负载确认。加载、预热和参考生成不并入计时；全部在线处理计入。
- 真实HTTP/SSE每方法16token均与AR一致，直接取消R+DFlash后再次DFlash生成正确。真实loopback HTTP压测6/6成功、concurrency2、wall2.998s、P50约0.982s、客户端TTFT P50约0.556s；小样本不输出P95/P99，最后queued/running均0。HTTP断连、慢流及429的系统场景另由CPU测试覆盖，不夸大为GPU高并发验证。
- GPU profiler单独插桩：218550事件、20602 CUDA kernel事件，含7个命名阶段。索引/查找较短，矩阵乘与Target验证具有显著设备成本；完整trace保留服务器`artifacts/acceptance-v1/profile/trace.json`，不提交大型trace。不据插桩时间宣布新性能优化。
- GPU实际package源码SHA`52f5845362006bc8aacebf96f85a8274c4b305b1e43953ffe49c611dfe2801f3`；最终package源码SHA`bec65fec2c1524fabec9d2b4dbbb01fb5c4259d41a96b48828e34f926b63f64a`。逐文件差异和生成核心不变核查见`results/infra-v0.3/delivery.json`；不以最终来源覆盖原计时来源。GPU验收后新增门槛/资源清理等代码已完成最终CPU/小模型回归。
- 小型公开验收证据在`results/infra-v0.3/`，报告`INFRA_VALIDATION.md`；复算`.venv\Scripts\python.exe -m scripts.write_infra_report`。无私有对话、模型、数据库、wheel或完整trace上传。下一步从真实服务负载和profile选择一项深优化；连续批处理、BF16、采样、分布式和CUDA/Triton算子均不作为本版已交付能力。

## 2026-10-18 — 已有合成正结果的归因检查（完成，CPU阶段结束）

- 用户授权落实归因对照。新增`evaluation/reuse_attribution.py`、`diagnose_reuse_confirmation.py`、离线报告生成器、7项测试和`REUSE_ATTRIBUTION_PROTOCOL.md`。候选构造仅接收当前request，明确限于原合成benchmark语法，不读取gold、当前答案、过去回答或生成器隐藏元数据。旧4项目/8请求/96计时通过原audit，并绑定全部旧发布文件SHA，原推理来源仍为`2518e677cde7d1ee4d7c4201b560d14f7a44bd75500c95e2497c0c072e2b46c2`。
- 首轮离线结果见忽略缓存及发布`offline.json`：Structure100/2452、Static1832/2452、Global2336/2452、Compiled2396/2452；Compiled文本/token/语义全8/8，历史在其外覆盖0。随后透明追加Expanded中间对照：仅展开当前标识和默认值、故意不执行override，保持同schema两轮一致；该对照在首轮诊断后提出，不称预注册。`offline-expanded.json`保留中间结果，`offline-final.json`为最终证据。
- 最终Expanded覆盖2332/2452，历史独有4/2452（0.16%）；原Global实际接受5418个R token的位置中，Expanded也可oracle覆盖5409个（99.83%）。Expanded与历史首轮4/4逐字相同，全两轮4/8；Expanded任务0/8正确，Compiled8/8正确。Expanded/Compiled额外候选token数均与历史8/8配平；Static稍短，组合条件更多，逐例长度公开。oracle是6锚点/32接续/至少8连续正确的乐观枚举，来源/特殊token边界独立，不是速度、接受率或19.4%收益的因果百分比分摊。
- 原DFlash41.991595s、Global33.862142s（1.240×，延迟降低19.4%）和D轮数498→123保留。新证据表明这批结果可由当前明确事实展开成目标格式解释，未证明长期历史必要性。按协议在CPU阶段结束，不新增GPU生成或Static/Expanded/Compiled计时，也不宣称替代候选实际达到旧速度；H、原CI确认及新任务搜索未恢复。
- 本机命令：`.venv\Scripts\python.exe -m pytest -q tests/test_reuse_attribution.py tests/test_reuse_report.py --basetemp .test-tmp-attribution-final`，13 passed/5.66s；`.venv\Scripts\python.exe -m scripts.diagnose_reuse_confirmation --output results/raw/research-stage2/reuse-attribution-complete/offline-final.json`，最终结果复制至规范发布源目录，旧中间结果保留。完整`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-attribution-full`：105 passed、3 skipped、18.61s。分析本身仅CPU/tokenizer，无Target或Draft权重加载。
- 服务器审核包`reuse-attribution-audit.tar.gz` SHA `63d84631dc1a06ad494c74b3769eb97a9dd2cabf2c966ba1a43323455e918df9`，18个部署文件SHA通过。`.venv-dflash`相关13 passed/2.78s；用同一旧发布证据执行`OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv-dflash/bin/python -m scripts.diagnose_reuse_confirmation --source data/raw/reuse-attribution-original --output results/raw/research-stage2/reuse-attribution-server/offline-final.json`。两平台summary/逐例/资源/代码SHA/原证据SHA全部一致，报告字节SHA均`d15186f34306b97663fc7fe748ca7bee905560e3e397f79a60c22691fa7365f5`。初次服务器日志重定向因`results/raw`尚不存在失败，创建目录后完成；该次没有执行分析或模型生成。
- 最终本机分析SHA `16c4ce5db2b8afba80bda46e592e567fdb42e15a3428086a52fc97cb8033ca35`；服务器分析SHA `d50d686819aebadfba57f6875eb92ef766161e81ba2bf44e36b165f74eef92dd`（差异为tokenizer绝对路径和离线耗时）。分析脚本SHA `23f61c710a898beb418953e722d20ce5ff94a1dfe43637875327e7a68e8fe9c0`，模块SHA `6a66ad93d8b0520285d16816455208ba02ad059a0dddb668b906a3daf2cbb2a1`。服务器两GPU仍19MiB/0%。结果见`results/reuse-attribution/`及`REUSE_ATTRIBUTION_RESULTS.md`，原发布161文件保持原内容。
- 发布命令：`.venv\Scripts\python.exe scripts/publish_research_artifacts.py --source results/raw/research-stage2 --destination results`，新增8份公开CPU诊断/决定/测试证据，总169份；原161份manifest条目及发布文件SHA逐项不变。`.venv\Scripts\python.exe -m scripts.write_reuse_attribution_report`可从公开证据离线重建报告，不需要tokenizer。代码、报告及8份新证据提交GitHub供审阅。

## 2026-10-18 — 真实发布说明初筛：小成本可行性检查（完成，按门槛停止）

- 数据/协议/标签/代码在模型输出前提交并推送 `d45f942`；实际推理源码SHA `83f7e6f6b8f43965b9e916c8edd91e36cc484f85e933f25b9224763bef7cc302`，归档 `source-release-triage.tar.gz`。服务器部署全文件哈希通过，前置13 passed；原Qwen3-8B五权重分片SHA逐一验证后，A6000 GPU0执行18次正式Target AR，全部结束、0截断，累计请求217.819s，加载13.761s单列。固定预热、权重审计和离线评分不并入请求时间。作业PID1388481已完成，两GPU均空闲。
- 实际命令：`OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv-dflash/bin/python -u scripts/run_release_triage.py`。首评分：`OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv-dflash/bin/python -m scripts.write_release_triage_report`。冻结质量门槛失败：严格合同3/18、合同合格7/18、严格类别30/90。18份实际都可解析JSON且版本/类别顺序正确；整体合同失败会将五类全计错，不能把该严格得分称为纯分类准确率。
- 原预定oracle机会通过：12个有历史请求1633/2597额外独有覆盖（62.9%），6项目均达到项目门槛；它使用答案枚举最优接续，未计在线成本，不是接受率或加速比。质量未过，因此未启动最多48次DFlash/Context/Global/Arctic计时，原CI12项目确认和H训练亦未重启。
- 事后只读诊断 `scripts/diagnose_release_triage.py` / `posthoc-diagnosis.json`：只看status为88/90类、16/18请求正确；11请求证据超1–2条、9请求含不支持相应判断的证据行（重叠），关键status漏判0/误报1。两实际status错误为FIPS兼容误作漏洞修复、注解/内部typing导入误作API迁移。原始门槛/得分/标签/输出未改。
- 新增探索性静态schema资源对照：两个虚构JSON、323 token，不用当前事实/标签/答案。历史在prompt+静态资源之外独有覆盖仅96/2597（3.7%），解释大量固定格式重复。它在看过输出后追加，格式选择有事后性、资源未逐请求配平、没有在线计时，不能当新门槛或实测速度。两平台独立重算逐例诊断与oracle一致；补充回归本机15 passed（7.47s）、服务器15 passed（2.91s）。
- 结果 `results/release-triage-feasibility/`，主报告 `RELEASE_TRIAGE_RESULTS.md`，解释及结束决定 `RELEASE_TRIAGE_DIAGNOSIS.md`。完整运行前协议与标签保留。发布新增12文件（无权重/数据库），总161份，原149份逐项不变。参考原JSONL SHA `f151438d2d31d2fd84aadfcc0773c93edbfaacb58dd85c54204c7ddb0b93d7ed`；原门槛SHA `a21fe1416869abc0f0f1f0c0d38fb4514a5db3245c3e6ca9cc3012ff60050d10`；Windows/Linux主报告SHA均 `d8ddb168079eb8f65b0bec8ac386d7e5568d157c407bafd990593326fcc5d8c2`。
- 本轮候选检查结束，保留R+DFlash与负结果；不自动继续换数据集、扫提示或增加H/路由。既有静态格式就能解释大部分机会，尚无已论证的新方法贡献。用户未来若提供独立真实应用需求及明确瓶颈，再据其重新评估；不是声明所有自然结构化任务都无历史价值。

- 用户授权执行最后一个小成本候选。按 `RELEASE_TRIAGE_PROTOCOL.md` 固定6项目×3版本18请求，先Target质量/合法历史独有机会，任一门槛失败则停止加速计时。没有启动原12项目CI确认或H训练。
- 已通过只读GitHub连接获取每版真实changelog，tag逐一解析commit并校验tag/commit文件相同；保留各版完整章节及六份许可。模型当前材料充分，历史只来自同项目更早已完成回答。逐例来源标签在任何Target输出前审阅；没有独立专家或Target自评，限制明确记录。
- 推理前复核收窄三组过宽证据行范围，剔除纯类型/内部重构行；最初未运行标签草稿和准备manifest保存在忽略缓存。最终manifest SHA `e5e0b6ac9ddd1a33da4b3dadb5d11e1f3211aab93b3b34006f566c9b9c3415e7`。不按任何模型输出更改标签。
- 新增冻结采集器、分类评分/离线oracle模块、Target-only回放、审计报告和对应回归。本机针对性命令 `.venv\Scripts\python.exe -m pytest -q tests/test_release_triage.py --basetemp .test-tmp-release-triage`：11 passed。权重manifest结构已按现有文件核对，使用原Target五分片SHA校验。后续实际部署/推理来源及结果按完成情况追加。
- 补充未来答案/跨项目快照与标签篡改回归后，本机完整 `.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-release-full`：96 passed、3 skipped（41.39s）；脚本语法及 `git diff --check` 通过。数据/协议/评分代码将在任何GPU输出前提交并推送。

## 2026-10-18 — CI 负结果归因与处理建议（离线分析完成）

- 用户要求分析原因、提出可行解决办法，并明确没有可靠办法的部分。新增 `CONFIG_REVISIONS_DIAGNOSIS.md`：区分 Target 原文保真、当前输入与历史重复、成熟基线贡献和实际应用必要性。新方向仅为有停止条件的候选，未实施或宣称成功。
- 本轮再次核对开发 90/90、合同诊断 4/4 的保存 token/文本与各自 AR 一致。逐例查看原 7 个失败完整文件：5 个存在文件头/未改内容缺失，2 个改变正文空白；Requests 第二轮是可解析列表而非完整 YAML 映射，不误称为语法异常。v2 两例主体恢复后的注释/末尾 LF 问题另行解释。
- 用现有 `quality(..., mode='diff')` 在隔离临时目录直接应用开发请求给定的 12 个 patch，12/12 得到期望文件（沿用末尾 LF 容差）。未计时、未执行工作流、未新增 GPU 推理。实际命令为 PowerShell here-string 管道到 `.venv\Scripts\python.exe -X utf8 -`，完整复算代码及结果在新报告末尾；首次 GBK 打印失败以 UTF8 重跑，原结果未变。
- 建议停止当前 CI 扩展和 H 调参：工程上已知 patch 直接应用；自然语言编辑可另行评估最小修改操作；研究上最多保留一次真实材料重复结构化处理的小规模可行性检查。若没有相对当前输入的独立复用机会或可解释的场景贡献，结束新方法主线，保留实现与负结果。未启动此前停止的 12 项目确认。
- 仅修改三份文档，检查 `git diff --check`；原报告、发布结果和推理代码均保持原证据。无需为文档分析重复完整 GPU/回归套件。

## 2026-10-18 — 公开CI配置修订回放（开发与单次探测完成，确认按门槛停止）

- 单次contract-v2探测已完成：2个开发首轮、4次DFlash/Global配对全部与新AR严格一致；完整文本0/2正确，YAML结构1/2相等。深入逐行及叶子诊断表明Ruff删一行注释，Requests改写一句注释；Ruff唯一YAML值差异为末尾script块少终止LF，不能自动解释为业务逻辑错误。两例均文本正确的预定门槛未通过，**12项目确认未启动、未生成任何确认Target参考**，未继续扫提示或放宽保真要求。
- 探测实际命令：`OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv-dflash/bin/python -u scripts/run_config_revisions.py --split development --contract-version 2 --contract-probe`。源码清单 `077a04368b75f50be7a5d8a4ad2c0fd94e15763dfcc8c53e7f9e19dd4fe27be8`，与开发c6来源分开。GPU0/GPU1均已空闲。原模型输出/计时未改；两个输入版本都只在同批方法内比较。
- 报告 `CONFIG_REVISIONS_CONTRACT_PROBE_RESULTS.md`，原始探测 `results/config-revisions-contract-probe/`，包含输出SHA绑定的 `fidelity-diagnosis.json` 和预定门槛 `gate-decision.json`。本机最终完整83 passed、3 skipped；服务器`.venv-train`完整85 passed、1 skipped、1 warning（原有toy LoRA保存元数据提示），DFlash环境前置35 passed。最终服务器测试来源 `11bff98ee8bdfcb5536e61cef6c1ec6b86485a00c076e2e900bb968d1da97d30`，代码文件SHA与本机最终代码一致；测试来源与c6/077推理来源分开。
- 最终新增26份公开结果（开发13、探测13），统一发布清单149份；原123份保持不变。开发/诊断两报告在Windows/Linux复算逐字节相同，SHA分别 `770bc42af38fd7f2ae1a822ed3ab3469a251d9019c749deea508e95fa80e5297` / `f458c25f2835730f440641b0c78c19a3912b8cffadcc3c7662d7e1cc235d1fae`。原始构建日志保留终端空格并标记-whitespace，原始/压缩及Git blob SHA核验后发布。

- 开发90/90严格一致，完整组AR318.002s、DFlash60.753s、Adapted60.819s、Global36.220s、Context36.414s、Arctic37.306s。Global为DFlash的1.677×，Context/Global1.005、项目区间[0.997,1.019]，没有稳定独立历史贡献；Arctic/Global1.030仅6开发项目/单重复，不能作全面方法优势。
- 原Target合同v1完整文件仅5/12文本及YAML结构正确，10/12可解析；diff修正终止LF评分后3/6可正确应用，0/18参考截断。全部失败保留，性能不代表可用任务已经成立。报告 `CONFIG_REVISIONS_DEVELOPMENT_RESULTS.md` 已从原始/发布数据离线重建：版本/用户快照、token、KV前沿、任务检查、配对、summary全部核对。运行源码仍为c6清单，不用后补报告冒充原运行来源。
- 探测前预先声明：Ruff/Requests开发首轮，尾部重申完整文件/注释/未修改行，2个新AR参考+4次DFlash/Global配对。无答案或新事实；当时16项本机审计/合同/前置阶段测试通过。两例均正确才进入确认；不能把输入版本改变的效果归为R。结果与停止决定见本节顶部。
- 开发结果11份白名单JSON/无损gzip JSONL/LOG已发布，含原始v1评分和独立quality-v2.json更正、失败g++与成功clang构建日志、事后环境捕获；统一发布清单现134份。权重、C++二进制、SQLite和公开API原始缓存未入库。

- 用户授权进入真实项目工作负载。新增采集器、`evaluation/config_revisions.py`、`run_config_revisions.py`、离线审计报告和测试。固定6开发/12确认仓库、2025-09-01之前三版同一工作流及两轮修改，新增diff边界；这是公开项目衍生编辑API回放，不冒称自然用户对话。
- 全部18项目已冻结；数据SHA256 `944f0f77a41b08126411e8558a7b451e45341a412d2cee0460fb7c96b12f351e`。未按模型结果选取文件。API限流/服务器读取超时通过已连接GitHub只读工具缓存公开响应解决，保留原规则和跳过候选；手动复核mypy MIT/NumPy BSD三条款及scikit-learn COPYING，保留完整许可。Starlette同仓库ID迁移记录为Kludex/starlette。
- 初始用户文件与模型回答分开记录；后续真实仓库状态作为显式外部版本更新。Global增加显式授权用户文件选项，默认索引仍只读assistant。每条方法/重复独立SQLite，库创建、文件更新、冷索引、前向、缓存恢复、解码和写回计时；未来输出仅用于评分。
- 最相关基线锁定Apache-2.0 ArcticInference@aca5d9a8a62474035c15d114d40a01abc8c94b51的官方线性后缀候选组件，接入相同严格验证循环，非完整vLLM/树系统。源码七文件SHA与许可已保存。服务器`.venv-dflash`安装nanobind2.7.0；默认g++9缺少C++20 span，改用已有clang++15及官方CMake后成功构建，直接接续查询通过，未改系统环境或上游算法。
- 阶段启动协议 `CONFIG_REVISIONS_PROTOCOL.md` 原拟开发90次、确认540次；后续确认资源收敛和保真门槛见顶部与协议修订，原计划不是已运行结果。
- 首轮本机检查暴露默认GBK读取和系统临时目录不可写；统一UTF8及项目内忽略目录，隔离Git外层仓库发现/自动换行，未改冻结数据。修复后本机完整75 passed、3 skipped，服务器DFlash环境29 passed，包含官方C++候选器+真实tiny Target验证。原服务器源码清单 `c6c269ffce98cbfd2036dca7ccd612892cef52287e12a2a637da2bf91bd0215b` 全文件一致；冻结实现/协议提交6b1451c已推送。
- GPU开发运行：`OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv-dflash/bin/python -u scripts/run_config_revisions.py --split development`，结果 `results/config-revisions-development/`。进程1223876；仅GPU0，不把部分配对当完整结论。
- 开发发现真实Target会遗漏未修改部分，任务质量与严格token一致性分别报告，不剔除失败。另发现diff评分v1未补最后LF而误判可应用性；新增评分v2及缺失LF回归，本机10 passed。原始输出/计时/评分不改，报告先按原版本审计再离线记录quality-v2.json更正；原GPU进程仍运行c6源码。该修正不改变Target或接受条件。
- 任何确认输出之前收敛确认计时：开发仍有正式AR；确认保留独立AR参考、五条加速路径及diff边界，共468次而非原拟540次，避免额外重复AR基准成本。主比较仍为DFlash/Global，参数和Target输入不变；确认不报告相对AR加速比。方法集合/缺失行审计已补测试，本机11 passed。协议修订在 `CONFIG_REVISIONS_PROTOCOL.md` 顶部。

本阶段实际准备/验证命令：

```powershell
.venv\Scripts\python.exe scripts/prepare_config_revisions.py --offline
.venv\Scripts\python.exe scripts/prepare_suffix_assets.py
.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-config-full
.venv\Scripts\python.exe -m pytest -q tests/test_config_revisions.py --basetemp .test-tmp-config-confirm-plan
.venv\Scripts\python.exe scripts/bundle_source.py
```

服务器官方后缀组件构建及GPU前检查（默认g++失败日志另保留）：

```bash
.venv-dflash/bin/python -m pip install --no-deps --timeout 15 --retries 1 nanobind==2.7.0
cmake -S models/arctic-suffix-audit/csrc/suffix_decoding -B models/arctic-suffix-audit/build-clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++-15 -DPython_EXECUTABLE=$PWD/.venv-dflash/bin/python -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$PWD/models/arctic-suffix-audit/arctic_inference/suffix_decoding -DCMAKE_BUILD_TYPE=Release
cmake --build models/arctic-suffix-audit/build-clang -j4
OMP_NUM_THREADS=4 PYTHONPATH=. .venv-dflash/bin/python -m pytest -q tests/test_config_revisions.py tests/test_dflash_reuse.py tests/test_reuse_report.py --basetemp .test-tmp-config-dflash
```

## 2026-10-18 — R/DFlash 接入、场景开发与独立小规模确认（完成）

- 用户授权实现并运行。新增 `backends/dflash_reuse.py`，薄适配已审计 MIT DFlash 循环，复用模型前向/严格最长前缀验证/KV裁剪；支持连续R后累计有效Target特征并在回切时补齐，设待消费缓冲门槛。
- 后缀索引保留旧 `match`，增加有界位置查询及显式当前prompt候选；`retrieval/reuse.py` 增加本请求游标和局部重匹配。当前prompt不伪装成过去事件；历史仍由原MemoryStore时间快照读取。
- 本机首轮完整59 passed、2 skipped（PEFT和审计DFlash环境）；服务器4.57.3中12项新测试全部通过，包括真实小型DFlash类与官方循环对齐。补充报告/发布检查另6项通过。
- 服务器DFlash环境新增固定 `pydivsufsort==0.0.19`，不修改原环境。初次GPU A阶段暴露零命中 `sa_search` 返回left=None，已补提前返回及回归；该失败批次保留为 `results/reuse-pilot-lookup-failure/`，未进入开发计时。
- 开发实际源码SHA `57a013cadafc9859dc002bdefce72d47acaae4774bd13263788a9140ff8e947e`；A八请求×三方法24次全部对齐AR。B按 `REUSE_PILOT_PROTOCOL.md` 完成12合成开发项目/18当前请求/五方法单重复90次，全部对齐。开发Global总18.314s、DFlash23.345s；配置修订Global12.053s、DFlash16.284s（1.351×），Cursor14.191s，Context16.362s。完整文档Global1.363×，边界0.998×；不是独立性能结论。
- 实际命令：`OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv-dflash/bin/python -u scripts/run_reuse_pilot.py`。源码/模型SHA已校验，A6000仅GPU0。结果 `results/reuse-pilot/`，作业日志 `results/reuse-pilot-job.log`。
- 离线报告 `REUSE_PILOT_RESULTS.md` 已从原始及公开gzip复算：重建每轮过去快照，核对固定参考、配对、场景标签和KV前沿。开发及首次失败证据13份已加入发布白名单/哈希清单，合计112份。过去资产0/12、当前参考0/18达到预算。
- 预先声明小规模确认 `REUSE_CONFIRMATION_PROTOCOL.md`：选择简单Global，在4个全新项目/4种JSON schema/两轮修订上做3重复，另加适配关闭R和同策略Context对照。数据与参数在确认生成前固定，计划96次计时；不增加概率模型或修改R参数。服务器相关18项测试通过，确认实际源码SHA `2518e677cde7d1ee4d7c4201b560d14f7a44bd75500c95e2497c0c072e2b46c2`。
- 确认命令为上述runner加 `--confirmation`；结果 `results/reuse-confirmation/`，独立复算入口 `python -m scripts.write_reuse_confirmation_report`。96次计时全部严格一致：DFlash41.991595s、关闭R适配42.000497s、Global33.862142s、同策略Context44.427281s。DFlash/Global=1.240075，[1.198,1.286]；延迟降低19.36%。Adapted/Global=1.240338，[1.199,1.286]，Context/Global=1.311998，[1.290,1.333]。4个项目都比DFlash更快，但仍是小型受控JSON工作负载。
- 确认Global有246轮R、123轮Draft，R接受5418/7329；关闭R为498轮Draft。R全部来自合法过去回答。同策略Context有21轮R但接受0/672，说明当前事实描述不能替代过去完整表达。8/8当前参考JSON可解析，且按冻结请求逐字段检查全部正确；当前0/8、过去0/4达到768预算。检查不筛选任何计时样本。
- 本机完整65 passed、2 skipped；服务器旧后端/PEFT环境完整66 passed、1 skipped、1个已有toy保存警告；DFlash专用环境相关18项通过。新增语义审计另6项通过。服务器测试原始日志及其源码来源随确认结果发布。
- 原始/压缩发布清单共123份（以前99份+本轮24份），模型/数据库不发布。开发、首次失败和确认来源分别保存，禁止以最终部署SHA替换实际运行SHA。报告 `REUSE_PILOT_RESULTS.md` / `REUSE_CONFIRMATION_RESULTS.md` 均离线重建，主结论限定在配置连续小改。
- 本阶段完成后不继续对确认项目调参；保留游标负结果，采用简单Global作为此场景的后续底座。更广的自然配置/代码/文档、并发、其他精度和采样尚未验证。

## 2026-10-17 — 用户修订：场景优先，允许复用成熟架构（方案已调整）

- 用户明确无需击败所有基线，优先找到设计适合的场景；允许借鉴已有架构并形成自己的有效差异。更新 `MEMORY_REUSE_RESEARCH_PLAN.md` 和审阅入口，作为此前通用门槛的替代。
- 第一优先场景为同项目多轮局部修订，应用需要较长完整产物且改动集中；模板任务为次级探索，短输出/diff/新话题作为边界。开发阶段选择场景，在新的测试项目上确认，不能按测试加速比挑样例或人为改变输出模式。
- 核心对照保留 AR、DFlash、一个最相关复用基线与本方案；消融围绕实际改动。成功条件限定到选定场景，允许速度/内存/维护成本的明确取舍；取消“所有场景、所有基线均优”和全局低重复退化硬门槛。
- 先实现简单可复用组合，连续接受长度模型按观测到的问题再决定；新算法复杂度不再作为前置条件。本轮只有文档修订，未执行新 GPU 实验。检查：`git diff --check`。

## 2026-10-17 — 新方向文献、接口核查与研究方案（规划完成，实验未开始）

- 用户要求详细规划新方向及 idea 运作机制。新增 `docs/MEMORY_REUSE_RESEARCH_PLAN.md`，把主线候选收窄为严格验证反馈下的历史接续长度估计及成本选择；保留 H 实现和已有负结果。
- 核查 REST、SAM-Decoding、SuffixDecoding、EfficientEdit、ReSpec、DReSD、DFlash、ReTrace。明确后缀检索、动态长度、混合回退、编辑后重匹配及位置反馈均有先例；尚未宣称新颖性或新方法有效。文献链接和比较限制在方案中。
- 读取实际 `SuffixArrayIndex`、`Router`、`MemoryStore` 和锁定 DFlash `spec_generate`。识别连续 R 后 Draft KV 落后、有效 Target 特征累积、anchor 与缓存位置等接入约束；没有修改模型/验证逻辑。
- 方案规定 A 正确性、B 开发机会诊断、C 独立项目评估及强基线消融；所有数据规模、长度和停止门槛为拟议参数，正式运行前冻结。未构建新数据、未训练或运行 GPU，暂无新结果目录。
- 本轮检查命令：`git diff --check`；仅文档变化不重复模型实验或代码测试。研究方案入口同步到 `docs/REVIEW_HANDOFF.md`。

## 2026-10-17 — H 蒸馏与 DFlash 基线（完成）

- 用户授权执行上一轮建议的两个交付；运行前协议 `docs/H_TRAINING_PROTOCOL.md`。先核验数据隔离/标签掩码，完成独立环境 DFlash 小检查及同权重 D/H 蒸馏，不修改 Target 或放宽验证规则。
- A6000 GPU0/GPU1 空闲，项目盘余 410GB；原环境 torch 2.5.1、Transformers 4.52.2、PEFT 0.11.1。DFlash 使用独立环境，不更新原环境。
- 已读取官方模型快照源码与上游 MIT 许可；锁定 HF revision、上游 commit 和权重 SHA，详见协议及 `configs/dflash-manifest.json`。模型源码/权重留在忽略目录。
- DFlash 独立基线已完成：12 请求（旧六合成+六类别公开各一条）×2方法×3重复=72 次，全部与同环境 AR 及旧 AR 严格一致。FP32/eager、Transformers4.57.3/torch2.5.1，AR=187.385s、DFlash=84.682s，2.213×，请求 bootstrap [1.713,3.263]；合成2.840×、公开1.894×，不是完整公开24请求复现。结果 `results/dflash-baseline/`；模型从 hf-mirror 传输，最终官方 SHA 核验通过。大文件直传留下的部分 tar 未用于加载。
- H 训练前增加独立 `.venv-train`（bitsandbytes0.45.5、pydivsufsort0.0.19/rank-bm25），避开全局旧 bitsandbytes CUDA124 导入问题，不覆盖原环境。服务器四项训练小测试通过；本机完整48 passed、1 skipped（本机没有所需 PEFT，服务器已实际验证）。
- 40 条过去回答先以源码 SHA `28a80ae186d2b116c40ff5e48361c739f35142717cc9c98b7c36c191b3223524` 冻结。审查发现重新加载原始 S 会恢复默认 assistant 阈值；在计时前停止资产任务，修正原始/训练后均为 constant/threshold0，并用 `--resume-assets` 只复用核验后的过去回答。正式训练/评测实际源码 SHA `89362a22632ac47397e53e189faf6175d8c16c82a3344816cec19ef8735a41b9`，不把初始未完成资产计算当正式性能结果。
- LoRA 四轮训练40.485s、1,146,880参数；开发响应loss最低为第2轮0.083185，已固定该权重。开发64条长度计时、107个stride4前缀、24条路由计时选择k4，门控任务类为空；未宣称关闭H即成功。
- 测试八项目417个全前缀：未训练H−D_format平均+0.505590，请求bootstrap [0.068376,1.021098]；训练后H−D_format为−0.009273，[-0.084270,0.056452]，新静态R弱匹配96位置为+0.125。训练D_format平均接受长度3.749，训练H为3.740；历史对弱Draft有效的信号，未转化为对齐后Draft上的稳定增量。
- 最终168次计时全部正确，加开发88次均与固定AR一致。测试AR64.075s，BASE_D62.933s、BASE_H48.837s、TRAIN_D46.522s、TRAIN_H46.099s、TRAIN_off36.149s、TRAIN_on36.482s。BASE_D/TRAIN_D=1.353，[1.079,1.537]；TRAIN_D/TRAIN_H=1.009，[0.972,1.055]；TRAIN_off/TRAIN_on=0.991，[0.953,1.052]。off/on验证450→366，模型起草93→132，节省验证仍未形成稳定净时间收益。
- 报告 `docs/H_TRAINING_RESULTS.md`，原始公开/合成记录 `results/h-training/`；完整训练前未读取测试标签。报告复算重新核验项目隔离、冻结历史快照、Target参考及summary。未根据测试反馈继续训练或调阈值。

实际环境和运行命令（服务器项目根目录）：

```bash
.venv/bin/python -m venv --system-site-packages .venv-dflash
.venv-dflash/bin/python -m pip install --no-deps transformers==4.57.3 tokenizers==0.22.1 huggingface-hub==0.36.0
.venv/bin/python -m venv --system-site-packages .venv-train
.venv-train/bin/python -m pip install --no-deps bitsandbytes==0.45.5 pydivsufsort==0.0.19 rank-bm25==0.2.2
OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv-dflash/bin/python -u scripts/run_dflash_baseline.py
OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv-train/bin/python -u scripts/run_h_training.py --resume-assets
```

最后一条为本次复用已冻结过去资产的实际命令；干净目录从头运行不带 `--resume-assets`。目录存在时脚本拒绝覆盖；adapter或测试标签已存在时拒绝资产续跑。本机下载使用 `python -m scripts.prepare_dflash_assets` 并核验官方SHA；服务器镜像仅传输同一固定字节。DFlash已以`cd9ac0b`推送，最终H报告与结果随本次完成提交发布。

- 本机最终完整回归48 passed、1 skipped in 12.76s；服务器独立训练环境49 passed、1 warning in 7.96s（LoRA小测试空模型名导致的保存元数据警告，测试通过）。命令 `.venv-train/bin/python -m pytest -q --basetemp .test-tmp-training-final`，日志 `results/h-training/tests-final.log`。GPU任务全部结束，服务器报告离线重建成功。
- 发布清单现共99份白名单JSON/无损gzip JSONL/LOG；DFlash5份、本轮H23份。模型权重、adapter、数据库与源码压缩包未发布。统一原始/压缩SHA核验及Git字节核验用于交付；精确配置、checkpoint SHA、前缀候选、逐轮成本均保留可复算。
- 补充直接比较：新配置未训练BASE_D/BASE_H=1.289，[1.048,1.463]，普通Draft上历史的确有正向效果；训练后H独立优势未通过。两项实验集合不同，不能直接用DFlash与训练后H两表加速比相除。没有按测试表现继续训练或改变门控。

## 2026-10-17 — Target 历史的对称 H 开关验证（完成）

- 用户授权验证更换历史后 H 是否产生净加速。协议 `docs/H_TEACHER_ROUTER_PROTOCOL.md`：复用上一轮冻结 Target 历史，双方共享同一完整 R 索引；固定旧参数，仅比较 AR/H-off/H-on/缓存 H-on，六请求、三重复，72 次计时和 8 次 smoke。
- 新入口 `scripts/run_h_teacher_router.py` 复用旧路由包装器，核验冻结原文 SHA、过去任务构造、历史版本替换和当前请求；与上一轮 AR token/prompt 逐条核对，不重新生成历史或调节阈值。结果目录 `results/h-teacher-router/`，拒绝覆盖。
- 运行前协议/实现提交 `b7e1e53`；服务器测试发现夹具只兼容发布 gzip，修复为兼容原始 JSONL/gzip 并提交 `0419b0b`，未启动失败的 GPU 运行。修复后本机/服务器针对性均通过，服务器 **8 passed in 4.81s**。
- 实际 GPU 命令：`OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python -u scripts/run_h_teacher_router.py`，只使用 GPU0。运行源码清单 SHA-256 为 `436c2daaa920df93a6c0b745da0ede3d6a2b39c3d273f3bef2df32b4a88c640a`；后补报告与测试不替换这个实际来源记录。
- 72 次正式计时 + 8 次 smoke 全部与旧 AR 的 token/Target prompt 一致，方法间快照相同，结束后历史未变。AR=81.109s、H-off=49.356s（1.643× AR）、H-on=53.934s（1.504×）、缓存 on=52.043s（1.558×）。本批 H-on 比 off 慢 **9.27%**，缓存后仍慢 **5.44%**；off/on=0.915，请求 bootstrap 95% [0.888,0.932]；off/缓存 on=0.948，[0.922,0.964]。
- H-off/on 验证轮数 **807/816**，小模型 H+D 起草 **77/96**；H 本身接受 96/120。六项目启用 H 均更慢；复制/代码修改验证轮数不变，文档修改增加 18 轮，新任务减少 9 轮但多 12 轮模型起草。缓存与未缓存路线/来源/接受计数序列 18/18 配对相同。
- 无 GPU 的追加同前缀审计将在线 H 的 60 轮起点与旧完整评分对齐：在线 H 接受 96/120，旧离线 D_native 同位置 90/120，人工指定一条 Target 历史的 H_teacher 为 93/120。此为已保存候选质量检查，不是替换路由后的速度；在线自动历史与旧单条 H prompt 不同，不能单独归因多历史干扰。
- 结果目录 `results/h-teacher-router/`，报告 `docs/H_TEACHER_ROUTER_RESULTS.md`。离线复算：`python scripts/write_h_teacher_router_report.py`；结果发布：`python scripts/publish_research_artifacts.py --source results/raw/research-stage2 --destination results`。
- 本机完整回归 **44 passed in 12.29s**，服务器最终完整回归 **44 passed in 6.27s**，日志 `results/h-teacher-router/tests-final.log`。命令分别为 `.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-teacher-final` 和 `OMP_NUM_THREADS=4 PYTHONPATH=. .venv/bin/python -m pytest -q --basetemp .test-tmp-teacher-final`。服务器也成功离线重建报告，两张 GPU 均已空闲。
- 本轮导出 9 份白名单结果，统一发布清单共 71 份；全部原始/压缩 SHA-256 核对一致。原有 62 份结果不变；模型、数据库、原始私有对话与源码压缩包均未发布。报告、协议、入口和测试提交 GitHub 供 Astra 审阅。

本轮补上“同一 Target 历史下 H 是否有净收益”的缺失对照：当前固定策略仍无收益。原人工历史批次不用于跨批来源加速因果估计；没有调节模板、块长、阈值或样例。六合成项目并非独立自然长期数据，成熟 B6/B7、新触发策略、训练、采样及并发仍待验证。

## 2026-10-17 — H 来源与对称路由追加实验（完成）

- 用户授权执行两项小实验，协议预先固定于 `docs/H_CAUSAL_PROTOCOL.md`。复用已有六个合成测试项目，明确属于追加归因诊断；固定 Qwen3-8B/0.6B、FP32/eager、非思考 greedy、k=2、当前输出上限 128，没有重新挑选样例或阈值。
- 新增 `memorydraft/evaluation/history_alignment.py`：仅从原始 order=1 记录构造过去 lower()/ten-minutes 任务，保存 order=0 请求和 order=1 Target AR 回答，再冻结派生历史后读取当前测试请求。版本替换键包含 user/record/order，避免未来同 ID 版本污染过去。人工/Target 历史选取同一合法记录，全前缀评分，另报等长度截断敏感性检查。
- `TransformersBackend.generate(... enable_history=False)` 仅允许 RHD + 独立快照 R 索引；跳过 H 语义检索，R/D/AR 的探测、EMA、驻留和冷却保持相同逻辑。关闭 H 的旧 RD 保留为辅助基线，不充当严格 H 开关对照。
- 新增测试覆盖过去任务构造/未来更正、H 关闭和路由一致性、相同历史/空历史质量控制。本机针对新增模块 **4 passed**；完整回归和 GPU 结果随后记录。
- 实际运行命令：`OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python -u scripts/run_h_causal.py`。结果目录 `results/h-causal/`；先来源诊断，再两请求 smoke，再六方法三重复的路由评测，避免 GPU 任务重叠。

### 结果与复核

- 实际运行源码清单 SHA-256：`d5072df7254a4867a73e4ed2548d502d30234e2e29acec056aad15dd4b15577f`。协议与实现先以 `8c230ec` 推送；原运行 provenance 保留其实际文件清单，后补报告/CPU 审计不冒充主运行源码。
- 六条模拟过去回答均未达到 256 token 上限，Target 生成累计 32.618s。冻结历史 SHA `55483ae5160588591aed932bca5fb62e188e301f39bd064c817e3abcab0c919a`，之后才加载当前请求。全前缀 **532** 个位置，新 AR 与第二轮保存的 token/Target prompt hash 完全一致。
- H_manual 平均接受长度 1.663，H_teacher 1.753；Teacher−Manual 为 **+0.090**，请求 bootstrap 95% **[+0.008,+0.238]**，主要来自文档修改项目。首位命中 86.09%→91.17%（位置等权）；等长度截断比较 +0.153，但截断会改变信息覆盖，不将其当作纯文风因果证明。
- 108 次路由配对计时 + 6 次 smoke 全部正确。AR=81.264s、D_native=69.960s、旧 RD=57.317s、对称 RHD_off=54.550s、RHD_on=60.086s、缓存 on=58.082s。相对 AR 分别为 1.000/1.162/1.418/1.490/1.352/1.399×。H-on 比 H-off 慢约 10.15%，缓存后仍慢约 6.47%；旧 RD 不能替代对称 H-off。
- H-off/on 验证回合 914/912，仅少两轮；模型 H+D 起草回合 76/120，多 44 轮。H-on 的 150/168 接受量不是全局增量收益，需扣除被它替代的 R/D 机会；不把其 89% 接受比例直接解释为系统成功。
- 两个主 GPU 实验结束后执行 CPU 后分析：`OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python scripts/audit_teacher_overlap.py`。来源改善的 51 个位置中，使用同一 Target 历史的静态 R 可覆盖其中 31 个。此为接续质量审计，不是在线速度；文档项目在新 R 弱匹配处仍有局部来源增益，但只有一个项目。
- 本机完整回归 **40 passed in 12.26s**；新增模块/报告/发布针对性检查 **6 passed**。服务器主运行前针对性 **4 passed**；两个 GPU 实验结束、最终源码部署后，完整回归 **40 passed in 6.00s**，日志见 `results/h-causal/tests-final.log`。
- 已导出只含合成/公开 JSON、压缩 JSONL、TXT/LOG 的结果，模型、数据库和源码压缩包不入库。`docs/H_CAUSAL_RESULTS.md` 由 `python scripts/write_h_causal_report.py` 离线重建；发布原始/压缩后 SHA 保存在统一 publication ledger。

本轮支持“历史来源对齐影响 H 候选质量”，但没有测出原人工历史上启用 H 的净收益；两个实验不合并冒充 Target-history RHD 的在线加速。成熟 B6/B7 及新的独立用户项目验证仍未完成。

## 2026-10-16 — 架构 v0.2

- 已完成：架构复核、六阶段计划、Codex 启动任务及工程约束。
- 未完成：底座选择、代码导入、模型运行、GPU 性能测试。
- 下一步：执行 CODEX_START.md 第一部分，选择独立 Draft 友好的现有底座，建立 AR/D 基线并尽早实现 H。
- 性能结论：无。本仓库初始内容为设计，不是已运行的推理系统。

## 2026-10-16 — 可运行原型与 GPU pilot

### 已完成的工程

- 选择并锁定 Transformers 4.52.2 / commit `237c7c356cce5f8a6befc0a5f71bc952bb5adc9a`，复用上游验证、KV 回滚、模型加载和独立 Draft 生成。SAM/RAPID 仅审查接口，未复制无明确许可代码；详见 `BASE_SELECTION.md`。
- H/D 共用 Qwen2.5-0.5B 权重，各自固定 prompt、开放 assistant 前缀、独立 KV、即时拒绝裁剪和懒追赶；Target 固定为 Qwen2.5-1.5B。实现 R 接续复制、语义 L1 + token L2、成本路由与 AR 退路。
- SQLite 不可变版本事件，按明确可用顺序恢复快照；用户隔离、未来更正/删除、持久化、最终回答写回已检查。候选库只含请求前的历史，没有把当前测试答案写入过去。
- CLI 提供 env/import/generate/replay/report；冻结历史跨方法回放，相关/空/等 token 无关历史消融，逐 token 一致性检查及配对总时间比。
- 19 项关键测试在本机通过，涵盖验证偏移、全接受/拒绝/EOS/长度、H/D/R/AR 切换和真实上游 KV 使用、检索计时公平性、异常恢复、时间与用户隔离、不可变事件、持久化、路由成本退路和配对统计。

### 环境与运行位置

- 本机 RTX3090 24GB；当前 Python 3.13.5 / torch 2.10.0+cpu，因此 GPU 运行转到服务器，不是本轮模型超出 3090 显存。
- 服务器 Linux，双 A6000 48GB，实际仅用 GPU0。项目路径 `/media/a6000/D/ZhaoBinHe/MemoryDraft`，独立 `.venv`，torch 2.5.1+cu124 / Transformers 4.52.2 / sentence-transformers 4.1.0。复用现有基础 Python，不修改其他项目环境。
- 0.5B 普通指令模型从固定 revision 下载；多语言 MiniLM 从本机官方源下载、SCP 传输，两端压缩包 SHA-256 一致：`f9eec229a244d12246e90fc4a58c1499456be3fe5110829919f5d7d3fedee3c8`。
- 本机 SSH 克隆 GitHub 超时，HTTPS 克隆成功；远端 Hugging Face 直连超时、镜像曾限流，均已绕过为官方源下载后传输，没有把网络失败写成成功实验。

### 实际命令与结果

本机检查：

```powershell
.venv\Scripts\python -m pytest -q --basetemp .test-tmp7
git diff --check
python scripts\bundle_source.py
```

GPU 回放及机制检查（服务器项目根目录）：

```bash
.venv/bin/python -m pytest -q --basetemp .test-tmp4
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONPATH=. .venv/bin/python scripts/diagnose_precision.py --config configs/a6000-pilot.json --output results/diagnostics/precision.json
.venv/bin/python -m memorydraft replay --config configs/a6000-validation.json --db pilot.db --requests data/synthetic/requests.jsonl --output results/validation/replay.jsonl --max-new-tokens 64 --block-size 8
.venv/bin/python scripts/run_a6000_suite.py
```

- `results/validation/`：初版路由，13 请求 × 6 方法 = 78 次，全部与 AR token IDs 一致。
- `results/ar-fallback/`：新成本退路，13 请求 × 2 方法 = 26 次，全部一致。
- `results/semantic/`：真正多语言 CPU embedding、最终路由，13 请求 × 6 方法 = 78 次，全部一致。
- `results/ablation-empty/`、`results/ablation-unrelated/`：各 5 个相同受控请求 × AR/D/H = 15 次，均一致；空历史 H/D 接受和提议计数相同。
- `results/demo/`：最终回答在 order=30 写入；另一个进程验证 order=30 尚不可见、order=31 可见、其他用户不可见，结果全部通过。同步写回单独计时并计入该 demo 的总延迟。
- 共 212 次配对方法运行通过，另有单请求写回检查；这些不是 212 个独立自然请求。初期失败的 BF16 运行不在通过次数内。
- 上述小型原始结果、summary、provenance 已同步到本机。最终套件包含运行源码 SHA-256 清单；初版词法运行只有模型/配置记录，不伪造缺失的 Git commit。

### 实测发现与修复

1. 延迟激活的 H 首轮 mask 未覆盖已确认 Y，会导致 RoPE 维度错误。已修复完整 `C_H+Y` mask，并通过路由切换回归。
2. BF16/SDPA 的 docs-3 在第 54 个 token 与 AR 分歧；诊断发现接近并列的量化 logits。相同请求 FP32/eager 一致，后续参考实验统一使用该配置，未放宽接受规则。
3. 普通小模型起草成本大，H 没有稳定提高接受量。基于真实成本添加 H/D/AR 探测及退路，仍保留 H 实际运行与独立消融。
4. 语义最终配置：AR 总耗时 **9.972s**、D **18.819s**、H **19.965s**、RHD **13.300s**。H 接受 320/852，D 接受 322/819。完整 RHD 实际走 R=2/H=21/D=21/AR=242 轮。当前组合没有净加速。

详见 `EXPERIMENTS.md`，表内数字来自对应结果文件。单次合成 pilot 不作统计显著性、通用性能或“越聊越快”声明。

### 尚未达到的研究完成条件

- B6/B7 成熟学习型 Draft 对照没有完成。已确认 EAGLE 官方 Qwen2-7B 配对，但它不匹配本轮 Qwen2.5-1.5B；不能拿普通 0.5B D 冒充。需要另选匹配 Target，并适配 Target 特征恢复。这个范围尚未实现，而非已经验证不兼容所有模型。
- 当前 pilot 是设置 A，使用明确标注的合成项目流；应用事实记忆设置 B、真实合法用户历史、更长输出及可靠重复仍未完成。
- 采样、并发、服务后端、树融合及训练未实现；CLI 明确拒绝非 greedy 参数。CPU 内存/索引增长成本与精简 trace 的正式测量仍待完成。
- 项目已具备可运行工程骨架与 R/H/D 机制闭环，不能据此宣布六阶段研究或性能目标全部完成。代码与结果保留于本机及授权服务器目录。

## 2026-10-16 — GitHub 审阅交付

- 用户已授权将代码与报告推送 GitHub，供 Astra 检查与分析失败原因。
- 本次整理核心代码、配置、测试、合成数据及小型实验结果；模型权重、虚拟环境、数据库和模型压缩包由 `.gitignore` 排除。
- 新增 `docs/REVIEW_HANDOFF.md`，区分实测事实、可能原因和缺失对照，特别标明 dtype/attention 同时变更的归因限制。
- 发布前检查远端 main、文件大小、敏感内容、结果一致性及 `git diff --check`。本次仅增补审阅文档，不重新运行已通过的 GPU 套件；既有测试与运行记录保留。

## 2026-10-16 — 按用户要求复核研究路线

- 阅读原始论文/官方项目：标准 SD、REST、SAM-Decoding、RAPID、DReSD、RASD、Transformers 官方 assisted/prompt lookup、EAGLE 模型表。新增 `docs/NEXT_EXPERIMENT.md`，架构顶部明确 R/H 独立检索和原生 D/同指令 D 的优先修订。
- 已完成离线日志审计：`python scripts/audit_pilot.py --input results/semantic/replay.jsonl --output results/research-audit/pilot_summary.json`。输入校验 token、历史快照及方法配对；纯 R 4/424 轮起草、接受 12/17 token、贡献 12/436 输出 token。没有重跑 GPU。
- 已执行 `.venv\Scripts\python.exe -m pytest -q tests/test_pilot_audit.py`，**3 passed**，覆盖接受率/覆盖率区分及错误 token/快照配对拒绝。
- GitHub API 检查上游 commit/许可；REST `154d0f0ae45de5d8a9cbd1c696f6ad5da10904bf` 根目录 Apache-2.0、检索子模块 MIT。源码发现 u16 token 持久化与无记录分隔问题，不能直接适配 Qwen。SAM/RAPID 未见 LICENSE，提交与前次审查一致；本轮未复制第三方代码。
- Hugging Face 公开元数据核验 Qwen3-8B/0.6B，固定 revision 和文件 SHA-256 见 `results/research-audit/model_preflight.json`。两个 tokenizer 文件分别逐字节一致、配置词表大小相同；未下载权重或运行推理，尚不能宣称新组合有加速。
- GPU 下一阶段按协议推进：官方 D/prompt lookup 基线 → 独立历史 R → 同前缀 H/D → H 混合检索 → 路由与 B6/B7。长历史数据按原始版本、许可和时间过滤导入；不得用当前测试答案构造历史。

## 2026-10-16 — 第二轮实验执行记录（进行中）

- 新增原生 `D_native`、原生上下文适配 `D_adapter`、上游 prompt lookup `PLD`；旧 `D` 保留为带空 reference 的 H 同指令对照。RD/RHD 的 D 改用原生上下文，避免弱化实用基线。
- R 使用独立快照的 pydivsufsort 索引，uint32 固定宽度编码、对齐检查及记录/特殊 token 边界；动态输入复用调用上游 PromptLookupCandidateGenerator。H 的 dense/BM25-RRF 与 R 分离。新增相同前缀的 H/D 质量诊断与请求成组 bootstrap 汇总。
- 固定 Spec-Bench 提交 `fd2c1cd7d2201ef71db4c5f4e455008f017967bf`，按六类任务的既定顺序选 12 个开发、24 个测试请求，只取首轮。合成设置 B 另有 6 个开发、6 个独立测试项目，每用户 4 条预先存在的历史；详情及许可在 `data/research/`。
- 模型权重使用 `configs/qwen3-download-manifest.json` 中的公开 Qwen 官方镜像，通过完整 HF SHA-256 校验；在授权服务器目录下载。服务器中途断网，用户恢复后断点续传；下载未混入性能有效统计。
- 旧模型对照命令：`OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_research.py --config configs/a6000-validation.json --output results/research-old-bounded --stage smoke --max-new-tokens 48`。8 请求 × 6 方法均正确；AR 9.192s，D_native 13.809s（0.666×），PLD 6.395s（1.437×），R 6.111s（1.504×）。当时后台仍有下载，此结果仅作开发诊断，不作正式稳定性统计。
- 发现并修复 PLD 末尾超长（54 对 48 token）；保留原失败现场。另发现 Qwen3 新版 generation_config 会回填 do_sample=True，导致原标为 greedy 的实验实际采样。现 Target/S 均显式关闭 use_model_defaults，Target 显式 do_sample=False；`test_checkpoint_sampling_defaults_cannot_override_greedy` 禁止 multinomial。错误采样运行移至 `results/qwen3-inherited-defaults`，不计有效结果。
- 无关历史控制改为与 H_dense 等 payload token 数直接注入；人工预选历史强制使用合法选定记录，不再由语义阈值过滤。保存实际前缀长度、来源及少量候选/参考 token，测试空历史 H 与 D_format 提议相同。
- 新套件命令：`OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python -u scripts/run_qwen3_pilot.py`。在 `results/qwen3-pilot/` 保存逐阶段 provenance、SOURCE_MANIFEST、配置、首个分歧及配对记录；所有阶段先核验部署文件哈希。主运行源码清单 SHA-256 为 `6be842aa6afc83a1ed59c8e9129b5eaea384e58cd8b7feec6180e80c1bd95ba5`。
- 修复默认采样后，BF16/eager 在 spec-121 的 index=26 仍有分歧；FP32/eager 的 8 请求 × 6 方法通过。开发校准 D_native 的 k=2/4/8 加速为 1.186/1.125/1.006，冻结 k=2、FP32/eager 后进入测试。该选择不读取测试结果。
- 本机项目 venv 已安装 torch 2.6.0+cu124、NumPy 2.1.3，并确认 RTX3090 可用；系统 Python 的 CPU PyTorch 未改动。主 GPU 实测仍在 A6000 上进行，不混合两种硬件/运行时。显式 greedy 修复后本机完整回归 30 项通过，服务器针对新对照的 8 项通过；报告统计测试另有 3 项通过。最终报告将在套件完成后从结果生成。

## 2026-10-17 — 第二轮主实验完成与成本消融

- A6000 主套件和后续诊断已完成，`results/qwen3-pilot/complete.txt`、`post-checks-complete.txt` 均存在。FP32/eager、k=2、输出上限 128，所有测试配置在开发阶段冻结；原主运行源码清单仍为 `6be842aa6afc83a1ed59c8e9129b5eaea384e58cd8b7feec6180e80c1bd95ba5`，未用新源码伪造旧 provenance。
- 合成设置 B：6 项目 × 8 方法 × 3 次 = 144 次计时运行全部正确。D_native 1.168×，R 1.487×，RD 1.424×，H 1.103×，RHD 1.356×。H 相对同指令 D 多用 3.610 秒，在线接受比例 77.59% 对 78.45%；RHD 相对 RD 多用 2.848 秒，检索累计 2.934 秒。不能把相减的推算当成缓存后实测。
- Spec-Bench 固定测试切分：24 请求 × 4 方法 × 3 次 = 288 次全部正确。D_native 1.000×，R 1.056×，PLD 1.117×。D 在数学任务 1.166×、QA 0.897×；本次不能宣称 D 已普遍快于 AR。
- 主同前缀诊断每 16 token 采样，共 35 位置，H_dense 相对 D_format 平均 +0.375、请求区间跨零。为解释与在线接受率方向的差异，追加同请求全前缀 532 位置检查，增量降为 +0.045、95% 请求 bootstrap [-0.058,+0.149]；静态 R 弱匹配处约 +0.005。H_selected 与 H_dense 相近。保留两份结果，后者标为事后探索；没有重新挑选测试项目。
- `scripts/run_post_checks.py` 在主计时完成后依次执行 `run_dense_prefix_check.py`、`diagnose_saved_mismatch.py`。BF16 的 spec-121 index=26 在相同已确认输入上复现：AR top2 同为 33.75、块验证为 34.00/33.75，最大 logits 差 0.25；query 长度 1/9，支持数值形状差异解释。该诊断不计速度。
- 新增逐文本片段、容量 4096 的 CPU embedding LRU，只给当前合法快照文本打分，查询仍在线编码。未缓存路径保留；H/RHD 使用独立空缓存做一次交错成本消融，不预先离线建库。部署清单 SHA-256：`8d0276f05a9867701466941fe2fa9f28c4bddb7d8aba98b4164e8528948f443d`。本机完整测试 **35 passed in 30.48s**，服务器 **35 passed in 5.40s**。

本机结果整理命令（纯报告，无 GPU）：

```powershell
.venv\Scripts\python.exe scripts/publish_research_artifacts.py --source results/raw/research-stage2 --destination results
.venv\Scripts\python.exe scripts/write_research_report.py
.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-final35
git diff --check
```

追加成本实验（服务器项目目录，主 GPU 运行完成后）：

```bash
OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python -u scripts/run_embedding_cache_check.py
```

结果路径：主报告 `docs/EXPERIMENTS_V2.md`，主配对记录 `results/qwen3-pilot/{memory-test,public-test}/replay-k2.jsonl.gz`，H 诊断 `prefix-test/` 与 `prefix-dense-check/`，成本消融 `embedding-cache-check/`。发布工具仅收集公开/合成实验的 JSON/JSONL/TXT/LOG，JSONL 无损 gzip；逐文件原始与发布 SHA 在 `results/research-v2-publication.json`。权重、数据库与原始下载档不上传。

成熟 B6/B7、自然长期用户数据、采样/并发仍未完成；下一机制重点是历史接续与固定 Target 对齐，而不是在这六个测试项目上继续换检索器挑最优结果。项目未被标记为六阶段完成。

### 缓存消融结果与发布收尾

- 追加 6 项目 × 8 条方法/缓存条件 × 3 次 = **144** 次计时运行全部正确；本轮主测试加成本消融共 **576** 次，不含开发校准/失败/离线前缀诊断，不是 576 个独立请求。
- 同批 AR=81.172s，D=69.858s，D_native=69.446s，RD=56.956s；H 未缓存/缓存为 73.331/71.414s，RHD 为 59.944/57.923s。缓存使 H 约快 1.027×、RHD 约快 1.035×；但缓存 H 仍比同指令 D 慢约 2.23%，缓存 RHD 仍比 RD 慢约 1.70%。候选增益不足的问题没有被缓存掩盖。
- H/RHD 检索累计分别从 2.980/3.000s 降到 1.008/1.015s；两个独立缓存均命中 63 次、新编码 9 个片段。逐轮路线、来源、提议数、接受数和推进量在各自 18/18 个缓存配对中相同。第一次编码也计入，没有把离线预处理当零成本。
- 完整报告由已完成结果重建，缓存为看过测试后的工程消融；对应 provenance/缓存访问记录/测试日志一并保留。已有主证据提交 `4bf7010` 已推送。新增 `.gitattributes` 固定发布文件原始字节，并用 `git show HEAD:<path>` 确认全部已提交文件与发布 SHA 相符，避免换行转换破坏核查。
