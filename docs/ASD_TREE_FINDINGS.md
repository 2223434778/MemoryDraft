# 成本受控候选树 + ASD：GSM数学场景出现明确计算增量

第一步组合已实现，并完成六类各8道既有开发题、四方法的960次正式计时及48条同题AR参照。**在本批GSM8K上，组合较线性ASD完整请求时延低7.89%，共同128-token前缀低10.95%；两组最终答案均8/8正确。相对成本受控严格树，ASD使等128前缀再低9.87%。** 这支持继续研究树与ASD的组合，优先保留数学场景的实际计算收益。

MBPP也有小幅端到端收益；MATH和HumanEval存在质量或路径变化，单列呈现。数据均已观察，不称新独立盲测。每类8个请求是样本量，重复只估计计时稳定性；不要求所有任务或所有基线都获胜。完整四组结果见[生成表](ASD_TREE_RESULTS.md)，GPU前规则见[协议](ASD_TREE_PROTOCOL.md)。

## 实现了什么

复用现有DFlash、树attention及分支KV提交。独立实现[BASTION论文](https://arxiv.org/html/2605.29727v1)的路径概率/路径质量best-first思想，以每卡实测节点成本表替代论文roofline模型；没有导入上游代码，不称完整官方复现。上游可见commit `1577b5d9ee28783e15358f1dd791844e9f966172`，已核验MIT许可。

- 原生16块、最多15个proposal深度；每位置Top4，构造最多32节点，在8..32节点中按预计推进/cycle成本选树。EOS为叶子，预期推进不计EOS后代。
- 树上每个候选边使用其父节点的实际Target分数，保持原ASD `B=8, g=.25, m=2`。在可行连续根路径中优先最长，同长度优先低regret；局部门控分母取原生窗口剩余位置，与树宽无关。只提交选中路径的KV/特征/预算，预算0恢复严格greedy。
- 构树不使用Target未来分数、标准答案或新质量标签。尚未实现第二步“剩余regret预算驱动树形”，本轮是成本受控树与ASD路径选择的直接组合。
- 四组共享原Draft、Target、输入、FP32/highest/TF32关闭/eager及RMSNorm。全部在线TopK、归一化、传输、heap、mask、选择、验证、KV整理均计入请求wall。加载、预热、冷编译及一次性成本校准单列。

## 六类结果与适用范围

下表是组合相对线性ASD的逐请求时延比均值。负数更快；质量列沿用冻结最终答案/提供测试/EM/JSON合同评分。

|任务（各8题）|完整请求时延|等128时延（共同达到128的N）|正确数：ASD→组合|判断|
|---|---:|---:|---:|---|
|GSM8K|−7.89%|−10.95%，N7|8→8|主要正结果，保留|
|MATH-500|−10.02%|+5.69%，N8|7→6|存在速度与质量取舍，完整速度不能当纯计算提升|
|MBPP|−5.96%|无合格共同128子集|6→6|小幅完整请求正结果，保留|
|HumanEval|−15.97%|无合格共同128子集|7→6|未保留线性ASD的一个质量改善；严格树更合适|
|SQuAD|+0.47%|无共同128子集|6→6|近似零激活，当前无额外收益|
|合成JSON|−0.32%|−0.37%，N6|8→8|近似零激活，不归因于ASD新策略|

GSM自然点估计的95%请求bootstrap时延比区间为[.84984, 1.00070]；等128为[.81328, .96294]，相对严格树为[.85825, .93861]。保留小样本不确定性，同时接纳本批局部收益。MBPP自然区间[.85764, 1.02658]，不把8题外推为所有代码任务。

## 收益来自哪里

**树提供覆盖，ASD在树上提供额外推进，两项都有可观察贡献。** GSM严格树输出与AR/DFlash完全一致，验证291→235轮，完整时延较DFlash低11.66%，这是候选组织的收益。组合相对线性ASD的共同128集合中，7题均输出128token，共896token，验证157→126轮；同集合严格树142轮，加入ASD后再降到126轮。全部开销计入后，组合较线性ASD省10.95%、较严格树省9.87%，因此该计算增量不能只用最终输出变短解释。

GSM自然输出总量：线性ASD1648token/276轮，严格树1588/235，组合1757/234。组合比线性ASD逐题长度比均值增加7.60%，但验证更省，仍得到完整请求7.89%改善。组合相对严格树完整时延只省0.40%，与输出变长抵消每token增益一致；不能把9.87%的前缀收益直接当完整请求收益。

实际ASD轨迹的234棵GSM树中45棵接受过例外，同树严格前缀之外合计多推进109个位置。这个同状态诊断说明近似机制确实触发；它不等于另一完整策略的调用节省。四组交互比另存`interaction.json`，GSM等128点估计约1.033，区间跨1；目前支持有效组合，不作额外倍增效应主张。

MBPP严格树相对DFlash省5.71%，组合相对严格树再省1.85%，四组通过集合保持6/8。近似只在一轮触发，端到端小收益可以保留，但主要覆盖收益来自树。HumanEval严格树相对DFlash省9.67%，组合相对严格树反慢1.31%；本批该任务优先严格树。SQuAD/JSON的ASD例外均为0，千分级变化不提供近似策略贡献证据。

## MATH与质量变化

MATH组合相对严格树等128仍快8.71%，表明ASD能补充树内推进；但严格树本身比线性ASD等128慢16.14%，组合后仍慢5.69%。自然总量从线性ASD5657token/706轮降至组合4665/514，包含推理路径变化。完整时延快10.02%同时少答对一道，保留速度—质量取舍。

额外失败是`math500:448`：17模83的逆元。原ASD最终44，组合14；17×44模83=1，而17×14模83=72。这是真错，不是解析格式。`math500:444`各组的参数化概率表达式也均不等于固定1/2；`quality-review.json`保存具体反例，原评分及needs_review标记保留。正确最终答案不保证推理过程每一步正确。

HumanEval新增相对ASD失败是134：ASD通过提供测试，AR/DFlash/严格树和组合均失败。组合没有保留此例ASD的质量增益；严格树与AR一致本身仍成立。两类都不据此否认GSM/MBPP保留的局部成果。

## 对第二步的启示

可以继续“计算预算与近似预算联合分配”，主场景先选GSM，MATH作为压力测试。当前构树目标仍来自严格接受代理，没有利用剩余ASD预算：GSM组合的树最大深度平均8.81，MATH9.10，原窗口可达15。宽度扩展与较浅树会改变ASD可复用的连续后缀机会；这是值得验证的机制假设，尚无保留主干消融证明因果。

下一步优先比较保留较长主干、少量分支，以及剩余预算驱动的宽深分配。保留同树严格组和本轮直接组合组，才能区分联合规则的新增收益。约79% GSM组合块选择32节点上界，也提示节点范围是本实现的边界；本轮没有据结果扩大上界或扫参。

RMSNorm工程收益作为共同底座保留，公共前缀缓存另有长输入结果；不与本次百分比相乘。当前研究入口通过hook调用，尚未新增生产CLI/HTTP树方法，原严格默认与独立面试版保持。

## 执行、资源与审计

- 最终GPU源`05ec51424cc67ad3a38d86beed98ad3c3b47347e`，2058个canonical Git blobs，source SHA `6b89ac60d295dc44efc469717c61042c0bf0659932798bf7937edab0edf537d1`。本地完整ZIP25451573B/SHA `8ba3eec035d6e75471814356781e17b0847af4be91ffdcf74ee9c58b1194a52b`；实际传输已有engineering源+243文件delta2943869B/SHA `48ce8fcdd2af8888f4add14f16371251aba7b99958d5c73a1661e408b753b826`，前后全部2058字节SHA与Git一致，无换行偏离。
- 隔离root `/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-budget-tree-v2`，timeout PID1821773，2026-10-23 13:54:42 UTC启动，14:15:06 UTC完成末核查。两worker各480计时/24 AR，pilot24屏障通过后按原计划完成其余24，无按结果选题/改参。
- 960正式计时+48 AR=1008主生成。另每卡8条预热/零预算检查及1条校准生成，共18；合成上下文成本测量含额外Target前向，不混称正式生成。初版本的部分运行单独保留，不混入上述总数。
- 两卡加载20.45/19.82秒、成本校准4.92/4.92秒。峰allocated37804258816/37744344576B（最大约35.21GiB），每卡4图、正式新图0，backend3.416/3.417秒，编译cache各415135B。冷请求/预热原样保留，没有把一次性成本隐去后宣称单次CLI即获收益。末核查两GPU各19MiB/0%。
- 原pipeline ZIP850991B/SHA `aae86dc98ccad27c76b7acfcc7693f46e6e3f214621fa9f4ee3d84f126fdcfe1`保留；追加末核查/部署metadata的final ZIP851755B/SHA `63db7b3fa1a70f5b7cf52d526a3da3e7eb37e6e79d9276bdf7bace75481f32b0`，29成员逐SHA发布。JSONL无损gzip，不含模型/KV/logits/hidden。
- 首源ae2c6cb在代码复核发现EOS收益估计遗漏后停止，尚未比较速度/质量；原20成员ZIP264365B/SHA `cadde1be3680d09cd340a9a0127eed34b533615d651518701849d4ea1a30c349`保留在startup-eos。全包传输超时及系统Python3.8部署助手异常均发生在主评测之前，新目录准确重建后再执行。
- 最终本地完整979 passed/7 skipped/136.46秒，相关131 passed/2 skipped/13.29秒；服务器133 passed/9.33秒。六份本地XML保存于local-checks，区别初版与最终版。Windows3.13复建JSON/Markdown与服务器3.11原始载荷逐字节一致；源Git、29主载荷、账本、覆盖/重复、strict AR/input/预算、资源和infra三引用通过独立核查。

## 复现与主要文件

GPU命令（模型/.venv-dflash已准备，新目录有对应source manifest）：

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_tree_pipeline
```

本地纯统计与审计：

```powershell
.venv\Scripts\python.exe -m scripts.write_asd_tree_report
.venv\Scripts\python.exe -m results.asd-budget-tree.inspect_results
.venv\Scripts\python.exe -m results.asd-budget-tree.analyze_interaction
```

核心`memorydraft/backends/budget_tree.py`；接入`dflash_reuse.py`；统计`memorydraft/evaluation/asd_tree.py`；运行/归档`scripts/run_asd_tree{,_pipeline}.py`及`package_asd_tree_source.py`；报告`write_asd_tree_report.py`；新增`tests/test_budget_tree.py`与`test_asd_tree_report.py`。全部结果`results/asd-budget-tree/`。
