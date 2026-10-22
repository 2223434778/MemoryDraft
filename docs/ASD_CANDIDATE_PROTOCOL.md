# 阶段4：有限多候选与局部后缀修复（GPU生成前登记）

## 依据和边界

- DFlash https://arxiv.org/html/2602.06036v1 ：沿用已审计MIT adapter及固定模型，不新造推理引擎。并行Draft输出各位置分布，Top-1线性路径不是唯一可验证提议。
- BASTION https://arxiv.org/abs/2605.29727 与官方 https://github.com/kaist-ai-osi-lab/BASTION ：直接研究block-diffusion分布构造树与真实验证预算。本文只借鉴树验证与成本约束，不复现其置信度surrogate、best-first动态树或roofline校准；不能把简单树负结果当BASTION负结果。
- OPT-Tree https://aclanthology.org/2025.tacl-1.8/ ：共享前缀的树注意力隔离兄弟分支，节点位置取深度。该文主要AR Draft，不直接转抄到DFlash。
- ReTrace https://arxiv.org/abs/2608.29748v2 ：拒绝后缀可能仍有信息，但其表示融合与修正不同于本轮只重验token的免训练对照，不宣称已复现。其受错误前缀污染的旧Target logits/KV绝不能直接当新上下文的验证结果。

本轮新增代码为本仓库独立薄实现，没有复制或引入新上游源文件，故没有新的第三方代码许可/commit依赖；既有DFlash MIT与权重SHA继续核验。目标是当前工程栈的有限可行性，不主张树或后缀复用的原创性。

固定Qwen3-8B/DFlash b16权重及原Target prompt；FP32、highest、TF32关闭、eager attention、显式rmsnorm-inductor、greedy、空历史、严格最长匹配规则。native16为anchor+15 proposals。原服务默认、ASD预算/接受规则和面试引用不变；本轮用严格D先隔离候选结构，不将其结果冒充完整ASD算法性能。

## 固定六方法

1. native：现有线性16，增量EOS检查。
2. packed-linear：同一线性16经新mask/路径提交入口，但零分叉，用于隔离打包和选路开销。
3. tree32：保留native全部16节点；前四proposal位置各加一个该位置除native argmax外得分最高的替代token，接原位置Top-1后续，各分支含替代token共4节点，最多32节点。
4. tree48：同上，每分支最多8节点，总最多48节点。临近生成上限按原块截断；短块自然少于标称预算。不根据本次Target分布/答案选择分叉位置，不扫深度或置信度阈值。
5. repair4：普通Draft经过原matcher薄接口返回自身候选，严格验证后丢掉第一个拒绝token，以真实恢复token为下一轮anchor，直接取其后旧Draft的最多4token重验。至少2token才尝试，一次repair后无论是否成功都重新起草。
6. repair8：同上，上限8。原pending features和Draft追赶KV完整保留，1024上限仍强制追赶。拒绝Target状态不复用，只有候选token复用；每次Target均按真实新上下文重验。实际R接口的CPU token传输和匹配回调成本也不扣除。

树用一个Target调用验证全部节点，不逐分支forward冒充单次验证。每个节点只看已确认前缀及自己祖先；同层兄弟使用同一position_id。严格greedy从root开始沿匹配真实argmax的子节点行进，首个无匹配处生成恢复token；只把所选路径KV整理成连续后缀，所选hidden features提供下一轮Draft。保留原线性路径意味着理论树内推进不小于线性，但浮点shape变化仍可能分歧，全部记录。树入口排斥ASD近似verifier、R matcher、长度分配等混用，防止一轮同时改变多个主轴。

## 数据、计时、资源

首轮使用原selected_records六类开发题各前2共12，按固定次序偶/奇题分两卡。六方法每题3重复，seed=candidate:pilot:<id>:<repeat>交错，完整自然生成上限4096且遵守EOS，共216正式请求；每卡六方法各一次32token warmup，共12。保留旧完整DFlash/RMSNorm的token/text/EOS、prompt SHA、snapshot核对与完整质量。数学/代码/SQuAD/JSON沿用原评分器；计时排除离线评分，逐输出缓存评分不会给生成预热答案。新方法若输出不同全部保留，阻止相应候选扩测；不删难题/不强制续写。native也须与旧完整参考相同。

请求wall含prefill、全部Draft和Target、mask构造/Top-2提取/CPU设备同步/树选路/KV整理/特征提交/解码/停止与末同步。编译请求内成本保留；独立cache，记录warmup/正式新图，不以profiler时间替换wall。每卡图<=8、backend<=300秒、cache<=4GiB、peak allocated<=46GiB；CPU前置300秒、每worker3600秒，只结束本轮创建的进程会话。两卡运行前须空闲。

题级三次时延均值除同题同卡native均值，然后题等权，六类分别列；重复不作独立样本。报告Target/Draft调用、验证总节点、实际分支选择、同状态native接受长度与树增量、repair尝试/零接受/接受数。这些机制计数不能替代完整请求收益，也不能把两组不同轨迹的均值相减当孤立kernel成本。

## 继续/停止规则

候选必须各题token/text/EOS匹配旧完整native，prompt/snapshot一致、每臂3次token/轮轨迹稳定、原native也安全，资源全部满足。整体题均时延/native<=0.97，或任一家族至少2题均<=0.95且两题均更快；对应范围三个重复的题均比都<1。5%/3%是工程准入而非统计显著性。packed-linear只是控制，不作为新增算法准入。

任何候选在任一范围通过，则保持同参数，在全部48开发题做native+所有通过候选各3次完整自然输出。首12包含于48，不冒充新独立验证；不按通过家族挑选新题。扩测整体或原通过家族仍净快且所有安全才讨论ASD移植和独立新数据；不使用已观察validation48作盲测，reserve96不生成。若无候选通过，结束本固定树/直接token修复可行性，不继续扫预算或换题；训练后缀表示、完整BASTION自适应树等仍明确未评估。

## 工程、复现和证据

真实tiny Target测试树mask、分叉logits/所选KV对独立路径、下一步logits、stream/EOS/上限、跨轮feature追赶、默认线性回归、ties和非法组合；报告测试矩阵/重复/NaN/数值分歧和准入。先相关再完整本地回归，服务器相关检查后GPU。主协议/代码/统计生成前commit冻结、canonical Git blob部署、前后SHA；全部旧结果及infra main/tag/branch保护。

独立服务器目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-candidate-pilot`。入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_candidate_pipeline`。若扩测，新目录asd-candidate-expanded、同入口追加`--scope expanded --methods native <通过候选>`。原始`results/raw/asd-candidate-{pilot,expanded}`；发布`results/asd-candidate-{pilot,expanded}`；报告`python -m scripts.write_asd_candidate_report`（扩测追加相同scope/methods）。不上传权重、logits/hidden、大trace、cache或DB。
