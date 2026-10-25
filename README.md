# MemoryDraft

**算法架构说明：**[树 + ASD：怎样用一次验证推进更多 token](docs/ASD_TREE_ARCHITECTURE.md)。用示意例子、执行流程和实测验证轮次解释成本树与 ASD 的组合及性能收益。

> [七任务独立扩样完成](docs/ASD_PAPER_EXTENSION_FINDINGS.md)：514新源任务/530匹配输入，四组9540主计时、640预算控制、320奖励评分及136局部性质检查完成。
> 严格成本树在530自然请求上保留DFlash输出，在GSM/MATH/HumanEval/MBPP/MT/Alpaca自然时延分别省10.25%/6.19%/2.64%/14.87%/13.92%/17.15%。
> 树+ASD较ASD在新GSM/MBPP/Alpaca省7.72%/10.21%/17.89%，MT自身历史双轮省12.93%；MATH慢5.50%、HumanEval正确53→52，MMLU格式合同损失保留。
> 独立奖励偏好MT50%、Alpaca46.09%，不宣称全面质量保持；[完整表](docs/ASD_PAPER_EXTENSION_RESULTS.md)、[匿名判定](results/asd-paper-extension/open-contract-audit.json)与[审计](results/asd-paper-extension/audit.json)公开。与单token AR的一题数值分歧及续跑修订单列，不称论文原配置精确复现。

> [七任务剩余指标补全](docs/ASD_PAPER_METRICS_FINDINGS.md)：747新固定计时、120独立奖励评分完成。
> GSM严格DFlash/AR直接测得3.845×，组合/AR跨会话锚点估计4.709×；实际Draft接受长度ASD5.351→组合6.792。
> 数学组合收益保持；开放奖励偏好MT为62.5%、Alpaca为25%，任务合同检查另列。完整[补充表](docs/ASD_PAPER_METRICS_RESULTS.md)与[文献基线](docs/ASD_LITERATURE_REFERENCE.md)区分实测、估计和原文报告值。

> [ASD论文七任务扩样本](docs/ASD_PAPER_FINDINGS.md)：241新源任务/249匹配输入、五组4980计时+249 AR完成。
> BudgetTree-ASD在GSM64题较ASD完整省9.86%、等128省11.64%，正确集合保持63/64；MATH32题省11.42%/4.70%、正确26→28。
> Joint未显示对第一组合的稳定增量。代码有实际失败/截断，MMLU有输出格式变化，开放裁判未通过换序控制；全部正负结果见[主表](docs/ASD_PAPER_RESULTS.md)及[质量复核](results/asd-paper/quality-audit.json)。

> [成本受控树 + ASD](docs/ASD_TREE_FINDINGS.md)：六类各8道已观察开发题，960计时+48 AR完成。
> GSM8K组合较线性ASD完整请求省7.89%、等128省10.95%，答案均8/8；同等128较严格树再省9.87%。
> MBPP完整省5.96%、质量保持；MATH/HumanEval有质量取舍，全部正负结果及复现见报告。既有工程成果继续保留。

> [第二阶段：剩余regret联合构树](docs/ASD_JOINT_TREE_FINDINGS.md)：六组1440计时+48 AR完成，旧四组960轨迹重现。
> Joint在GSM较第一阶段完整再省1.17%、等128再省2.48%，答案均8/8；同896 token验证126→122轮。
> MATH前缀省5.36%但完整慢4.97%；其他任务无新增收益。固定主干消融及质量取舍全部保留，Joint作为GSM研究候选。

> [ASD候选统一比较](docs/ASD_COMPARISON_FINDINGS.md)：原reserve96已用于24题14组初筛及72题确认，
> 完成2448计时+96 AR，全部执行一致性检查通过。位置适配在新GSM等128时延省1.42%，完整请求慢0.79%；
> Utility在新MATH完整请求慢32.04%，不能将初筛局部收益推广。全部正负结果及复现见[完整表](docs/ASD_COMPARISON_RESULTS.md)。
> 下文历史实验中的“保留题未生成”指各轮运行当时状态；目前这96题均已使用。

> OT-R 实验在 `codex/ot-r-probe` 分支进行。此前完整 infra 已独立冻结为
> `codex/infra-interview-v0.3` / `infra-v0.3-interview`，本机另存独立仓库、ZIP 和 Git bundle。
> 见 [面试版本位置与复现](docs/INFRA_INTERVIEW.md)。OT 运行配置为 `configs/runtime-ot.json`，
> 需安装额外依赖 `python -m pip install -e ".[dflash,transport]"`；实验规则见 [OT-R 协议](docs/OT_R_PROTOCOL.md)。
> [首轮结果](docs/OT_R_RESULTS.md)：80次计时严格一致，旧JSON上OT较后缀R降低约1.83%延迟，收益集中在一个旧项目；格式不匹配复制例仍退化。默认检索保持suffix。
> [24个新请求验证](docs/OT_R_VALIDATION_RESULTS.md)：192次计时严格一致，OT总体较suffix慢1.75%，两个预定适用场景均未达继续投入门槛；停止扩大同类OT实验。
> [实际触发位置诊断](docs/OT_R_DIAGNOSIS_RESULTS.md)：24次插桩回放均正确，定位到编辑区误触发、端点漏召回及排序问题；旧正例的OT消歧机制得到解释。插桩数据不作性能证据。
> [DFlash前缀引导验证](docs/OT_GUIDED_RESULTS.md)：新16请求/160计时全部严格一致，主场景引导OT较suffix降低2.38%延迟，引导Edit降低2.55%；未显示OT排序增量，默认仍suffix。显式实验配置`configs/runtime-ot-guided.json`，检索选项`ot-guided`/`edit-guided`。
> [内部锚点拼接开发检查](docs/OT_SPLICE_RESULTS.md)：SQL局部可减少验证轮，但首轮整体慢于Guided-Edit；定位到Draft前缀拒绝触发R冷却。[五请求最小修复检查](docs/OT_SPLICE_COOLDOWN_RESULTS.md)保留局部收益，尾部加权OT与Guided-Edit整体基本持平，仍有保留错误Draft前缀的问题。全部为已观察任务，尚无独立OT增量确认；显式配置`configs/runtime-ot-splice.json`，默认保持suffix。
> [OT完整矩阵三个小改动](docs/OT_MATRIX_RESULTS.md)：135次旧调用重现，14个同状态/39候选；64机制与128计时全部严格一致。局部顺序正则未改变选择，矩阵特征会误拒有用接续，组合仍慢于TailOT；该路线已停止继续优化，旧结果保留。
> [预计收益门控与动态长度](docs/OT_BENEFIT_RESULTS.md)：保留内部拼接，不使用完整矩阵特征。正式56机制/112计时全部严格一致；OT求解候选31→7，但验证65→67轮，比固定TailOT慢1.57%，比Suffix快2.46%，尚未显示新路由收益。14状态的跨项目接受预测误差较大，长度始终选32。`configs/runtime-ot-benefit.json`是基础模型配置；实验runner按项目排除后注入校准策略，默认仍suffix。
> [接受长度校准诊断](docs/ACCEPTANCE_RESULTS.md)：48次严格生成、56个去重Draft状态。用Draft置信度和条件生存估计，原14动作MAE从10.47降至8.42（−19.57%），Brier从0.235降至0.156；关键坏前缀与两处候选排序改善，但好长尾被低估，项目区间跨零，未通过预定门槛。预测器保留为诊断，不接入默认路由，不宣称加速。
> [前缀风险辅助拼接](docs/PREFIX_RISK_RESULTS.md)：已实现显式`edit-risk-splice`，仅在原边界存活概率<0.5且有安全替代边界时改选，固定原长度上限。旧56状态校准、独立6项目/12请求检查；新Target只有11/12完整输出合同正确，按冻结协议在机制/计时前停止。当前尚无风险策略性能结论，保留实现与失败参考。
> [风险拼接完整续测](docs/PREFIX_RISK_CONTINUATION_RESULTS.md)：按用户要求单列任务质量、保留全部12请求，60机制/120计时全部严格一致；本批没有合格拼接候选，风险未触发，无可归因的性能增益。[两个旧训练项目的覆盖检查](docs/PREFIX_RISK_CONTROLS_RESULTS.md)中，payment改选边界14→7、接受7→32、验证8→7轮，但无置信度消融也取得相同行为，仍慢于该例Guided-Edit；不作为泛化或置信度独有收益。

**固定 Target 的推理优化、服务与正确性评测工具链。**

ASD 近似解码实验独立位于 `codex/asd-splice`：[组合报告](docs/ASD_SPLICE_RESULTS.md)。
旧8请求/9方法已完成360次机制及计时运行；自然EOS全部与AR一致，但ASD零例外、零减轮，Splice-ASD较Splice慢0.28%，连续后缀变体慢0.60%。
关键拼接阻断的Target logit差距为26.8–53.9，未显示低差距例外释放长历史尾部的机会。默认CLI/服务仍严格验证；ASD实现允许改变输出，不作为无损加速。

[ASD论文任务小实验](docs/ASD_BENCH_RESULTS.md)已完成：GSM8K/MBPP各8题，官方B=8配置，共320生成。
GSM8K自然时延降低18.31%、固定工作量吞吐提高10.60%，两组最终答案均8/8；MBPP自然时延降低2.35%，公开测试均5/8。
内部拼接未触发，当前正结果归于ASD；本轮不是DSpark复现。MBPP固定128token的大吞吐增幅包含约72%自然EOS后的强制续写，不等同正常代码请求加速。

[ASD实际前缀收益预算验证](docs/ASD_BUDGET_RESULTS.md)已完成：新GSM8K 16题/MATH-500 8题，共288生成。
收益选择组相对原ASD自然时延下降4.37%/9.93%，但GSM8K正确数15/16→14/16；实际后缀组在MATH下降10.22%、GSM8K基本持平。
MATH完整结束5题的自然TPS几乎持平，总体TPS收益主要来自3道上限截断题；尚无稳定质量下的完整长解答每token增益结论。近似策略继续作为独立研究选项。

[三道截断MATH题补全](docs/ASD_COMPLETION_RESULTS.md)已完成36次生成：统一2048上限，四组均3/3正确且正常结束。
实际后缀/收益选择相对原ASD自然时延下降15.83%/14.16%，自然TPS提高8.61%/9.35%；全部保留旧512前缀。
实际后缀仍有一题时延退化6.26%；这是已观察题的补全诊断，尚未证明新题泛化，旧GSM8K质量退化记录继续保留。

[24道新题的收益归因](docs/ASD_ATTRIBUTION_RESULTS.md)已完成600次生成，新增固定保守ASD对照和遵守EOS的等128token前缀。
新MATH上Run/Utility相对原ASD自然时延下降13.40%/9.84%，但自然TPS下降2.10%/2.20%；Run输出缩短15.22%，每token验证轮反增1.73%。
等128token前缀Run比原ASD慢3.65%；新GSM8K上Run慢8.02%、Utility持平，五组均8/8正确。
MATH答案等价审计后严格DFlash/原ASD/Run/Utility/保守ASD为12/9/13/14/11（共16题，原保守主评分均少1题）。
局部支出选择改善仍存在，但未确认新增后缀门控的稳定吞吐优势；当前端到端正结果主要伴随输出路径变化，继续保留实验选项，不进入默认严格运行时。

[置信度动态定价小检查](docs/ASD_CONFIDENCE_RESULTS.md)完成4道已观察开发题、140生成及16同前缀干预分支。
Confidence-Utility在三道MATH上比Utility自然时延低3.76%，但输出缩短8.36%、自然TPS低4.78%、等128token前缀慢7.28%。
旧GSM8K错误题答对，但固定价格1/3也答对，价格3更快；16干预分支均答对，未获得质量风险标签差异。
概率集中不代表Target认可Draft，停止扩大该简单pair_mass价格规则；采集、干预工具与负结果保留，默认严格运行时不启用。

[Utility错误轨迹因果诊断](docs/ASD_UTILITY_CAUSAL_RESULTS.md)完成2道旧开发题、62次生成，751策略块通过独立标量审计。
gsm8k:21的9次单独拒绝有8次修复错答，但严格同前缀分支只有位置211出现质量差异：Draft直接收尾，Target继续计算后答对。
该位置已有高regret/Run信号，预算未耗尽；尚无置信度独立价值。6对等128后续token中，接受例外后验证轮更少/更多/相同为3/2/1，局部Run不等于未来计算收益。无新增在线加速结论。

[保留实际KV的计算收益检查](docs/ASD_COMPUTE_RESULTS.md)完成4旧题/16源块/100生成。
15个共同质量、等128token状态，保留当前Utility动作255未来验证轮，有限二选一oracle249轮（仅省2.35%，未达预定5%门槛）；固定特征留题预测器259轮，反而多1.57%。
32自然分支均最终答对，但这些分支在当前动作后都严格续写，原完整Utility的旧GSM错答仍保留。停止扩大这条未来计算预测路线，不增加复杂预测头；无新的线上加速结论。

[ASD剩余优化排查](docs/ASD_OPTIMIZATION_RESULTS.md)完成8道已观察题、420主生成及32单token分支，另有AR/profiler控制。
BF16/SDPA使严格DFlash的等128token时间较原FP32/eager下降38.30%/41.19%（开发/迁移）；同BF16下基础ASD额外下降11.73%/21.11%。
这些是执行组合和基础ASD的结果；全部6个长度变体在FP32和BF16均未通过继续门槛，未见ASD前缀导致Draft退化，不进入适配训练。
BF16主检查存在2/8题AR/D逐ID数值分歧，全部记录；自然质量/长度/截断单列，默认严格FP32/eager配置继续保留。

[更丰富数据的分阶段评估](docs/ASD_SUITE_FINDINGS.md)已完成第一阶段：六类新开发题各8道，1200次计时与48次机制回放，另48验证/96保留题尚未生成模型答案。
严格D在各类均快于AR；基础ASD在数学等128前缀有计算收益，Run/Utility尚无进一步稳定增益；短问答/JSON近似规则未触发，代码有质量提升和输出变长并存的例子。
完整记录见[跨任务结果](docs/ASD_SUITE_RESULTS.md)，[研究路线表](docs/ASD_RESEARCH_ROADMAP.md)逐项标明未做方向；不把旧4/8题诊断当作整条路线的最终否决。
[真实更长Draft预检](docs/ASD_BLOCK_RESULTS.md)另完成六类各1题/228生成：16块重现、24/32能执行且严格D与AR一致；Utility24在两道数学题有信号，严格D长块较慢，继续完整开发集检查，尚不作泛化结论。
[完整48题长块比较](docs/ASD_BLOCK_SUITE_FINDINGS.md)已完成3120计时/144机制：Utility24相对基础ASD16六类自然时延均更慢约2.9%–16.4%，基础ASD32还出现两个额外质量失败。保留16块，停止当前b16权重直接扩块扫参；验证48及保留96仍未生成模型答案。

[原生16块执行成本诊断](docs/ASD_EXECUTION_FINDINGS.md)已完成12题/144计时/48独立profile及77真实状态回放。模型前向占GPU活动99.68%–99.97%；resident接受策略的请求摊销proxy约0.16%–1.95%，不是可兑现提速或理论上限。优先做一次轻量验证器精简检查，主要执行优化转向Target/Draft编译融合；插桩延迟不计速度，全部228生成轨迹/预算重现。

[CPU摘要判定小检查](docs/ASD_CPU_FINDINGS.md)已完成420计时/430影子块/77状态回放：保持原策略轨迹与预算，ASD/Run/Utility时延降低0.30%/0.67%/0.64%；resident成本降低约80%–85%不能当作生成倍速。保留显式CPU实验选项，默认infra不改，主要优化进入模型编译/融合；未新增自然完整答案或验证池评测。

[MLP区域编译检查](docs/ASD_COMPILE_FINDINGS.md)完成120预检/288配对计时/36同输入回放：37个模块每卡共享2图、正式计时无重编译，四策略时延仅低0.17%–0.29%。本批token相同但13组账本浮点变化；生成代码只融合SiLU乘法，3个GEMM仍在，模块回放未更快。保留研究工具，停止MLP-only扫参。

[独立RMSNorm融合检查](docs/ASD_RMSNORM_FINDINGS.md)完成120预检/288计时/192真实状态：167个norm每卡共享4图、正式无重编译，DFlash/ASD/Run/Utility时延下降2.67%–2.75%，各12/12题更快。同输入norm回放快约5%–9%，生成代码确认归约融合。本批token全同但10组账本浮点变化，尚未核查新完整自然答案；默认infra不改，下一项扩48开发题自然EOS质量和时延。

[RMSNorm完整48开发题自然EOS评估](docs/ASD_RMSNORM_SUITE_FINDINGS.md)完成1152配对计时+96 AR质量对照：四方法时延下降2.78%–2.82%，各48/48题更快，240对完整输出全部相同、质量未新增变化。原有错答/待复核及一例Utility截断保留，54对账本浮点不同；正式无新编译。冻结当前执行候选，下一步预登记48题独立验证，默认infra保持原实现。

[RMSNorm独立48题验证](docs/ASD_RMSNORM_VALIDATION_FINDINGS.md)完成1152计时+96 AR对照：四方法自然时延下降2.68%–2.86%，240对完整输出全同，六类均改善，全部预登记工程门槛通过。DFlash一题计时退化和原有质量问题保留；reserve96未生成。继续显式profile的CLI/服务接入验收，默认eager不变。

[显式RMSNorm运行时](docs/RMSNORM_RUNTIME.md)已完成[真实GPU接入验收](docs/RMSNORM_RUNTIME_RESULTS.md)：`configs/runtime-dflash-rmsnorm.json`选择数值profile，12示例配对、取消恢复、三路SSE、HTTP队列和新进程CLI通过。默认eager/原面试版不变；冷编译计请求时间，R示例仅证明触发兼容，未显示额外提速，不以小样本代替独立性能结果。

[阶段3计算分配检查](docs/ASD_ALLOCATION_FINDINGS.md)完成六类各2题、480主生成及27个真实KV状态：请求oracle仅省0.93%，局部MBPP/短问答仍有7.13%/16.31%理想选择空间，但留题廉价选择器未通过任何合格场景，按预登记规则停止闭环扩测。实际缩短Draft8/只验证7的128/EOS请求总体更慢；保留原生16块、工具与全部结果，不把局部oracle当线上提速，reserve96未生成。

[阶段4有限树候选与后缀修复](docs/ASD_CANDIDATE_FINDINGS.md)完成六类各2题、216次完整自然计时，六方法严格输出一致。32节点树在GSM两题平均快3.45%，总体慢1.85%；48节点树慢18.87%，直接重验旧后缀4/8慢约17.7%。调用减少不足以抵消节点成本，后缀修复约70%首位再拒；按原门槛不自动扩48，保留窄场景线索及组件，不将有限负结果泛化为完整BASTION/ReTrace无效，默认及面试版保持。

[阶段5等预算Draft适配](docs/ASD_ADAPTATION_FINDINGS.md)完成32新GSM train/8内部校准、普通与ASD实际前缀两组rank8 LoRA各128更新，以及18开发题540次计时。ASD前缀适配+ASD在GSM8题自然时延低4.46%、共同达128的7题低3.34%，最终答案保持8/8；普通适配共同128反慢2.18%。全部strict输出与旧参考一致，未新增质量失败。单seed/已观察开发集及源进度差异限制仍在，额外增量区间跨1，不作独立确认；权重留服务器、默认不升级，reserve未生成。

[同位置控制与独立16题](docs/ASD_ADAPTATION_FOLLOWUP_FINDINGS.md)完成新增严格前缀适配及960计时/24 AR参考。**已接纳位置匹配严格前缀适配作为本轮数学ASD场景的有效优化方案**：新16题128时延降低3.23%、自然降低7.32%，适配未新增答案错误；自然值含错误长路径缩短，共同正确14题为3.70%/3.41%（事后诊断）。旧ASD前缀收益也已重现，局部成果保留；ASD内容独有优势按原门槛结果记录。按用户决定不补seed；新ASD各组14/16、strict15/16，范围和质量差异保留，默认/面试版保持，reserve未生成。

[数学ASD运行入口](docs/ASD_RUNTIME.md)现已接入CLI/HTTP/SSE，提供原Draft和已接纳位置适配Draft两个显式配置，每请求独立预算并标记近似输出。[工程验收](docs/ASD_RUNTIME_RESULTS.md)完成32直接结果、8 AR参考和2次独立CLI；真实流式、取消恢复及队列通过。冷CLI累计账本存在不到1e-5的末位差异，原零容差失败记录及修正说明保留；未改解码规则或新增性能百分比。

[长输入数学工作负载](docs/ASD_LONG_CONTEXT_FINDINGS.md)完成8个已观察任务×约512/2K/4K输入，216正式结果+24 AR与9独立profile；全部strict匹配AR、无截断/OOM/正式新图。位置适配在2K共同128五题省2.32%，三个长度自然请求均无净提速；ASD两组均7/8。4K首token约4.8秒，占平均请求约61%，据此推进公共示例前缀缓存，保留既有短输入收益。该轮模型请求时延不含HTTP和排队，未接连续批处理。

[公共前缀Target KV及DFlash特征缓存](docs/ASD_PREFIX_CACHE_FINDINGS.md)完成288配对生成及80真实HTTP负载请求。ASD命中自然时延在约512/2K/4K输入低13.69%/42.53%/59.10%，4K首token4.82→0.20秒；首次填充加一次命中的配对总成本低28.98%。同前缀HTTP请求流从空缓存起步命中19/20，总wall低62.45%；两种大前缀交替时受2GiB容量限制，0/20命中、慢0.10%。输出/验证总轮数和ASD汇总量保持，spent末位差异单列；默认关闭、仅缓存登记公开前缀、每请求独立活动KV/预算。配置`configs/runtime-dflash-asd-prefix-cache.json`用于常驻服务，单次CLI退出即清缓存；未实现分块去重或连续批处理。

[共享公共段修复](docs/ASD_SHARED_PREFIX_FINDINGS.md)完成96次4K配对生成及120次新HTTP负载请求：将两种公开长前缀统一登记为3934-token公共段，容量仍2GiB，交替流命中0/20→19/20，相对同卡新整前缀控制总wall降低53.94%；单一前缀流慢3.55%，保留原配置。输出/EOS/验证总轮数及ASD汇总量保持，质量未新增变化，spent末位差异单列。显式`configs/runtime-dflash-asd-shared-prefix.json`适合这类模板家族，在线缓存/引擎/接受器未改；尚未实现通用块或分支存储。

MemoryDraft v0.3 将原投机解码研究原型整理为可安装的单 GPU infra：统一 AR / DFlash / R+DFlash 运行入口，带有界队列和取消机制的 HTTP/SSE 服务，配对 benchmark、CPU/GPU profiler 与回归测试。复用成熟 Transformers、DFlash 和后缀索引，不重新实现 attention 或推理引擎。

当前支持 batch=1、greedy、FP32/eager，一个模型 worker 串行执行。HTTP 可并发排队；连续批处理、分布式推理和随机采样尚未支持。研究模块 H/D/RHD 与全部实验结果继续保留。

## 三分钟运行 CPU demo

建议 Python 3.11。以下 demo 只演示 CLI 与服务链路，不加载模型，所有报告标记 `synthetic=true`。

```bash
python -m venv .venv-infra
# Linux: source .venv-infra/bin/activate
# Windows: .venv-infra\Scripts\Activate.ps1
python -m pip install -e ".[serve,client]"
python -m memorydraft infer --config configs/runtime-demo.json --request examples/request.json --method AR --output artifacts/demo-answer.json
python -m memorydraft serve --config configs/runtime-demo.json --port 8000
```

另开终端，激活同一环境：

```bash
python -m memorydraft load-test --url http://127.0.0.1:8000 --request examples/request.json --method AR --count 20 --concurrency 2 --output artifacts/demo-load
```

也提供 CPU demo Dockerfile：`docker build -t memorydraft-demo .`，然后 `docker run --rm -p 127.0.0.1:8000:8000 memorydraft-demo`。容器不含模型、历史或 GPU 推理环境。

## 真实 GPU 推理

DFlash 与旧研究后端分别绑定 Transformers 4.57.3 / 4.52.2，使用独立环境。先安装适合硬件的 PyTorch，再安装对应 profile；运行时只读取已准备的本地模型，不自动下载。

```bash
python -m pip install -e ".[dflash,serve,client,test]"
python -m memorydraft infer --config configs/runtime-dflash.json --request examples/request.json --method R+DFlash --output artifacts/answer.json
python -m memorydraft benchmark --config configs/runtime-dflash.json --requests examples/requests.jsonl --methods AR DFlash R+DFlash --repeats 3 --output artifacts/benchmark-01
python -m memorydraft profile --config configs/runtime-dflash.json --request examples/request.json --method R+DFlash --output artifacts/profile-01
python -m memorydraft serve --config configs/runtime-dflash.json --port 8000 --queue-capacity 8
```

配置内路径相对于配置文件解析。默认 Qwen3-8B + DFlash FP32 profile 使用 A6000 48GB；RTX3090 可使用较小 Target 运行 AR，或运行测试。模型准备、API schema、SSE 示例、计时口径、故障排查与边界见 [完整使用指南](docs/INFRA_GUIDE.md)。

## 工程结构

| 模块 | 职责 |
|---|---|
| `memorydraft/runtime/` | 请求/结果协议、配置与模型加载、统一生成、benchmark 与 profiler |
| `memorydraft/serving/` | 单模型 worker、有界队列、背压、deadline/断连取消、HTTP 压测 |
| `memorydraft/backends/` | 成熟推理实现适配、严格验证、KV 回滚与补齐 |
| `memorydraft/memory/` | SQLite 不可变事件、按用户与可用时间选取快照 |
| `memorydraft/retrieval/` | token 后缀检索、记录边界与独立的历史辅助检索 |
| `memorydraft/drafters/`、`routing/` | 原 H/D/RHD 研究模块与路由 |
| `tests/`、`.github/workflows/` | CPU 正确性回归、两个依赖 profile 的 CI 定义 |

服务的 `GET /healthz`、`GET /metrics`、`POST /v1/generate` 可通过 `/docs` 查看。SSE 仅发送已由 Target 确认的 token IDs，最终返回完整文本。队列满返回 429，deadline 包含排队时间，取消在解码边界生效。默认仅绑定本机，可信调用者提供 user_id；身份认证和外部部署由接入环境负责。

## 验证与结果

```bash
# 旧研究 profile：完整 CPU/小模型回归
python -m pip install -e ".[research,serve,test]"
python -m pytest -q --basetemp .test-tmp

# 独立 DFlash profile：工程与解码检查
python -m pytest -q tests/test_runtime.py tests/test_serving.py tests/test_dflash_reuse.py

# A6000 上的真实模型验收，输出目录必须尚不存在
python -m scripts.validate_infra --config configs/runtime-dflash.json --output artifacts/acceptance-01
```

benchmark 对 token、Target 输入哈希与快照哈希逐项检查，遇到首个分歧即停止；报告包含全部在线成本。profiler 结果独立标记，不能当作未插桩性能。小样本不输出 P95/P99。生成物默认保存在 Git 忽略的 `artifacts/`。

本版[工程验收报告](docs/INFRA_VALIDATION.md)：本机146项通过/3项跳过、服务器相关54项通过；真实A6000的12次配对计时与三种方法流式输出通过严格检查，6个HTTP请求完成。该小样本中R+DFlash慢于DFlash，不能据此宣称检索通用加速。

后续[同负载迁移验证](docs/RUNTIME_MIGRATION_RESULTS.md)已完成：原8个配置修订请求、五组交错、120次计时全部正确；新版D42.061s、R+D33.882s，保留约19.4%延迟收益。候选对齐后与旧循环计数相同，未发现明显迁移退化；此前两个短样例的负结果属于不同工作负载。完整回归现为156项通过/3项跳过。

原研究的 19.4% 延迟降低仅来自特定受控 JSON 工作负载；后续归因表明当前请求事实展开即可覆盖几乎全部复用机会，不能作为长期记忆独特价值或通用生产加速的证明。见 [归因报告](docs/REUSE_ATTRIBUTION_RESULTS.md)。

## 项目记录

- [infra 使用与运维](docs/INFRA_GUIDE.md)
- [进度与实际运行记录](docs/PROGRESS.md)
- [保留的研究 README 与实验入口](docs/RESEARCH_README.md)
- [原研究架构](docs/ARCHITECTURE.md)、[审阅交接](docs/REVIEW_HANDOFF.md)
- [第三方来源与 DFlash MIT notice](docs/third_party/DFLASH_LICENSE.txt)

当前未授予整个项目统一开源许可。第三方代码保留各自许可与固定来源。权重、私有对话、凭据、数据库和大型 trace 不进入 Git。
