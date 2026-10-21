# RMSNorm独立验证：约2.7%–2.9%收益保持，允许显式运行时接入

48道此前未生成模型答案的validation题，六类各8，冻结候选通过全部预登记工程门槛。四方法整体置信区间上界均低于1，六类点估计均改善；240对完整token/文本/EOS全部相同，无新增质量或完成性变化。该结论支持当前配置的显式数值profile；不把它升级为默认实现、任意输入无损或统计质量非劣证明。reserve96未生成。

|方法|编译/eager自然时延比|题级分层95% bootstrap|时延下降|更快题|
|---|---:|---|---:|---:|
|DFlash|0.973216|[0.971616,0.975672]|2.678%|47/48|
|ASD|0.972030|[0.971376,0.972684]|2.797%|48/48|
|ASD-Run|0.972200|[0.971696,0.972704]|2.780%|48/48|
|ASD-Utility|0.971433|[0.969198,0.972971]|2.857%|48/48|

1152次相邻配对计时，首重复同时评分；另96次AR质量对照，共1248主要生成，另20短warmup。自然EOS/上限4096，每题三pair先log平均、题等权，5000次类别内bootstrap；重复不是独立新题。没有看验证输出调参数或筛样本，AR一次输出不作速度统计。完整表见ASD_RMSNORM_VALIDATION_RESULTS.md。

六类各方法点估计均低于1：数学约3.1%，代码约2.6%–2.9%，JSON约2.8%–2.9%，SQuAD约1.5%–2.7%。DFlash唯一退化为squad:5725f46289a1e219009ac0fc：三pair比0.9825/0.9788/1.1410，第三次编译组0.2984秒、对应eager0.2615秒。输出、轮数和图数未变；这一单次抖动保留在总体和区间中，不推断未经测量的系统原因，不为了全题变快重跑。

## 输出、质量及归因

240对完整输出逐token、文本、停止原因相同；每profile全部重复精确重现，两profile严格DFlash/AR各48/48。无主评分、MATH secondary、SQuAD F1、新未评估或截断退化。所有错答和待复核保留，下表每格分母8、两profile相同：

|任务|AR/DFlash|ASD|Run|Utility|
|---|---:|---:|---:|---:|
|GSM8K|8|7|8|7|
|MATH主/secondary|6/6|6/6|6/6|7/7|
|MBPP提供测试|6|5|6|6|
|HumanEval提供测试|7|7|7|7|
|SQuAD EM|6|5|6|6|
|JSON合同|8|8|8|8|

SQuAD F1：AR/DFlash/Run/Utility均0.9571429，ASD为0.8654762。math500:232的AR/DFlash在两profile均4096截断；其他策略无本批截断，不据此声称其质量更好。MATH各组仍有不同待复核答案，详见analysis.json；代码提供测试不是全隐藏测试保证。ASD原有质量风险继续存在，norm收益不能解释成接受策略改进。

58/240对账本末位不同，最大spent差5.14984130859375e-5；5318个非AR轮除regret外一致，全部240对去regret后的轮序列相同，长度和调用密度不变。支持本批收益来自执行成本；不代表浮点运算逐位相同。该优化同样帮助严格DFlash，不能作为ASD/Run/Utility独有算法增量，也不能把CPU、MLP及norm各轮百分比相加。

## 编译、资源与审计

原FP32 highest/TF32关闭、eager attention、原16块、相同CPU接受及模型revision不变。只编译原RMSNorm.forward，Target145/Draft22共167norm；每卡4图，正式1152计时和96 AR都无新编译。CPU各profile11319次fast、fallback0。运行后只读8个codegen文件，各一次融合norm调用、0外部GEMM；不重复GPU微回放，不虚构纯kernel占比。

outer编译3.734/3.684秒，backend3.346/3.298秒（嵌套不相加），controller约0.00356/0.00354秒。warmup wall13.09/13.06秒还含推理。缓存各416796字节，峰值allocated35.926GiB，两profile相同；绑定切换中位0.355ms计在线。当前同分布请求平均配对省71.97/63.99ms，条件摊销outer+controller约52/58请求，不能当生产ROI保证。

冻结2c8c51106cd01208756de5458abb9e46889f59be/1417 Git blobs，source SHA27188c05eb8a7f0b609f22b9109d43712a82627aba0bf4790c3b3458af2b0c5b，source ZIP SHAbe33ae68cf30d2da16419e7a24f17f885a418864f46d56099aa3cdcab12ee999。原候选11核心文件与e958532一致。服务器/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-rmsnorm-validation，PID4169165，11:53:36–12:30:37 UTC（含测试/加载/预热/评分/打包），两worker0退出，两卡结束19MiB/0%。源码前后1417 SHA一致。

结果ZIP911053字节，SHAdbf7c7f183fcecf35f15e760f0cefbbd1fb0580fad96e075c547031e306936a1。生成前763结果文件/15521载荷对validation/reserve命中0，只证明仓库已有结果隔离；不声称预训练无污染。后审计25冻结文件、946旧结果SHA、8批297已发布文件和infra三引用通过，只有两个既有CRLF例外。冻结执行/评分/统计代码未改。

本地相关33项、完整688 passed/5 skipped/107.24秒；服务器130 passed/48.40秒。报告在原服务器Python3.11.9重建，实际重评105唯一题/文本。Windows Python3.13.5的三条mbpp:124失败输出只在NameError尾部增加导入建议，正确性/代码SHA相同，完整字典审计因此拒绝本地重建。scoring-portability.json保留差异；未删error、放宽比较或修改主报告。纯统计在本地另作逐字段复核，10个浮点字段末位差约1.11e-16，其他字段相同，完整差异在statistics-rebuild-local.json；原服务器主统计保留不覆盖。

复现主报告须原评分环境：在冻结源码根目录将结果放results/asd-rmsnorm-validation，然后`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -m scripts.write_asd_rmsnorm_validation_report`。描述诊断`python -m results.asd-rmsnorm-validation.inspect_results`，后审计`python -m results.asd-rmsnorm-validation.audit_postrun`；后者需要原结果ZIP和Git历史。源码冻结之后的新runtime文件应在对应提交复现，不能拿当前运行时冒充旧候选。

决定：停止在validation继续调norm；发布独立结果，合入显式profile候选并另做真实CLI/服务接入验收。该示例验收只确认工程路径，不替代本次性能实验；默认eager及独立infra面试引用保持原样。
