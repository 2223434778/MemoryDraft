# RMSNorm完整48开发题：约2.8%的执行收益保持，完整输出一致

本轮支持冻结RMSNorm-only作为待独立验证的执行候选。六类各8道开发题，自然EOS/4096上限，四方法的编译/eager时延比约0.972，各48/48题更快。新增做编译评估的36题有相近收益，完整输出与质量未新增变化。默认infra保持原实现；这不是所有输入逐位无损或生产质量非劣证明。

## 实际范围与主结果

1152次四方法相邻配对计时，首重复同时评分；另96次AR完整输出对照，共1248主要生成，另20短warmup。不是再加480次质量生成。六类为GSM8K、MATH-500、MBPP、HumanEval、SQuAD和明确标为合成的JSON。验证48/保留96未生成模型答案，新增36仍是开发题。

|方法|自然时延比（编译/eager）|题级分层95% bootstrap|时延下降|更快题|
|---|---:|---|---:|---:|
|DFlash|0.972056|[0.971392,0.972712]|2.794%|48/48|
|ASD|0.972249|[0.971607,0.972898]|2.775%|48/48|
|ASD-Run|0.972017|[0.971337,0.972641]|2.798%|48/48|
|ASD-Utility|0.971827|[0.971221,0.972385]|2.817%|48/48|

六类均改善：GSM8K约3.12%–3.29%，MATH约3.10%–3.18%，MBPP约2.81%–2.96%，HumanEval约2.86%–2.95%，SQuAD约1.75%–1.90%，JSON约2.80%–2.93%。每类仅8题，区间针对当前开发样本。新增36题时延比D/ASD/Run/U分别0.972071/0.972108/0.971823/0.971673；旧12题自然完整结果也约0.972。没有因个别好题才保留收益，但尚非独立验证。

每题先合并三pair的log比，再题等权，5000次类别内bootstrap；重复不算新题。AR只有一次质量对照，不对其单次时间作速度统计。主结果、全部逐题分歧和资源记录见ASD_RMSNORM_SUITE_RESULTS.md与results/asd-rmsnorm-suite。

## 完整质量与数值

240对跨profile完整输出（48题×AR/四方法）全部token与文本相同。240个eager首轮精确重现旧完整开发集原16块轨迹/质量；所有正式重复重现自身首轮。两个profile内DFlash与AR均48/48逐token相同。本批没有新增主评分、有限标量secondary或SQuAD F1变化；评分按同题/同文本缓存，实际97个唯一文本重新评分。

下表两profile完全相同；每格分母8，保留原始质量问题：

|任务|AR/DFlash|ASD|Run|Utility|
|---|---:|---:|---:|---:|
|GSM8K|8|8|8|8|
|MATH主/secondary|7/7|7/7|7/7|7/7|
|MBPP提供测试|6|6|6|6|
|HumanEval提供测试|6|7|6|7|
|SQuAD EM|6|6|6|6|
|JSON合同|8|8|8|8|

SQuAD各组F1均0.8214286。MATH math500:444仍列待复核，Utility在该题仍生成到4096上限且没有boxed最终答案；这项旧截断在两profile都存在，不能写成全部自然结束或全部答对。代码没有不支持操作导致的未评估项；提供测试通过不保证全部隐藏边界。各策略之间的质量差异来自旧解码轨迹，本轮不把它们算作norm带来的提升。

54/240对账本浮点值不同，最大spent差9.5367431640625e-5。5058个非AR轮的所有字段除regret外一致，所有240对轮序列去掉regret后相同；192个非AR案例输出长度和调用密度不变。支持本批自然时延差来自执行成本，而非缩短输出或减少验证轮。FP32归约数值仍改变；前轮192同输入状态也已显示多数norm输出不是逐位相同，因此不能由这48题推广为全输入无损替换。

## 实现、成本与资源

只编译原Qwen3RMSNorm.forward，Target145/Draft22共167处；MLP仍eager，原FP32 highest/TF32关闭、eager attention、原16块、相同CPUASDVerifier与接受规则。策略构造和profile绑定/恢复全部计入主时延，后者中位约0.345ms；评分在一对两次生成之后、engine时钟外，加载/落盘也在外。

每卡warmup后4张图，完整自然运行后仍4图；1152正式计时及96 AR对照都没有新增编译。CPU接受两profile正式各11361次fast，fallback0。没有通过换用更宽松实现或忽略新编译换取速度。

outer编译3.829/3.839秒，backend3.434/3.452秒（嵌套不相加），controller约0.0036/0.0039秒；warmup wall13.217/13.224秒还含两profile推理，不能全当编译。空独立cache各414736字节/0.396MiB；两profile的自然峰值allocated都35.964GiB，满足8图/300backend秒/46GiB/4GiB预算。worker5400秒预算内完成，两卡结束后19MiB/0%。

当前每卡自然请求混合，平均每对节省71.73/60.88ms；在本批输出一致且无线上重编译的条件下，仅outer+controller约需54/64个同分布请求摊销。相比前轮128上限的123/127，主要是请求更长、单请求绝对省时更多，编译本身仍约4秒；不能当成通用部署ROI或忽略新形状/冷启动。

本轮不重复同输入微回放。运行完成后只读查看8个生成文件：每图一次融合mean/pow/rsqrt/mul的Triton norm调用、外部GEMM为0，与前轮机制一致；只上传名称/调用次数/SHA，不发布cache源码。相同完整轨迹、轮数和接受实现下成本下降，结合前轮norm回放快约5%–9%，支持这一执行优化。没有本轮逐kernel profiler，不虚构精确带宽占比或原版launch次数。

该收益同时存在于DFlash及ASD/Run/Utility，属于共同执行选项；Run/Utility相对基础ASD的算法价值仍需原完整基线判断。CPU、MLP、RMSNorm不同轮次的百分比不能相加，本轮也未测试MLP+RMSNorm组合。

## 决定与下一步

停止在开发集调整norm参数，冻结现有代码/模型/精度/策略及评价口径。下一步进入此前未生成模型答案的48题validation，预先登记质量、数值分歧和时延准则；全部错误和截断继续保留。独立验证通过后再考虑显式运行时选项、更多形状/长上下文或serving整合；本轮不直接升级默认infra，也不再用开发题证明泛化。

当前可说“RMSNorm融合在这组48开发请求、FP32 Qwen3-8B/DFlash和双A6000条件下，有约2.8%的完整自然请求时延收益且未观察到新增输出变化”。不能说“任意输入严格无损”“所有ASD优化已经完成”或“独立验证已通过”。

## 复现与审计

冻结源码ca51a6338b039b557b80cb919be4ec15c98a9dd3/1383 blobs，source SHA9502d8b40a9229d6fd105d97ea8b5ff049e0f2c489e14bf311103809779f4c86；source ZIP SHAfeeb0faa047a386668c605af7e809c02e4cd8fe81702f66dcbc9e11755cdcdd0。服务器/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-rmsnorm-suite，PID4069094，2026-10-21 09:10:55 UTC启动、完成日志最后写入09:41:59，约31分钟含服务器测试/加载/预热/评分/打包。两worker0退出、REGIONAL RMSNORM_SUITE COMPLETE；源码前后1383文件SHA相同。

结果ZIP863831字节/SHA6eeef14c8bc0ded8683acf1e4b2d2b62d6ccf17d153ce5c26b13b21ba5fb7bd2。本地37相关通过；完整671 passed/5 skipped/90.12s，服务器113 passed/30.19s。后审计21冻结文件、919旧结果SHA、此前7批254发布文件、7原运行时/接受文件及infra三引用通过，只有两个已知旧CRLF例外。旧RMSNorm实际analysis在内存逐字段复现；面试infra仍27e36199。

重建：`.venv\Scripts\python.exe -m scripts.write_asd_rmsnorm_suite_report`；后描述诊断`.venv\Scripts\python.exe -m results.asd-rmsnorm-suite.inspect_results`；后审计`.venv\Scripts\python.exe -m results.asd-rmsnorm-suite.audit_postrun`（需要本地原结果ZIP及Git历史）。codegen/interpretation是运行后解释，未修改冻结主统计。发布小型raw/测试/报告，排除权重、完整hidden/logits、编译及Python字节码cache和凭据。
