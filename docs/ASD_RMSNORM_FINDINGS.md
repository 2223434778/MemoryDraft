# RMSNorm-only：小幅、跨本批任务一致的执行收益

本轮值得保留并扩大质量核查。相同CPU接受器、FP32/原16块，RMSNorm编译相对同轮eager使端到端时延降低约2.67%–2.75%；四方法各12/12题更快，六类任务的点估计均改善。它是DFlash/ASD共同受益的执行选项，不能包装为Run/Utility独有的算法增量，也不能把不同轮次CPU、MLP与本轮的百分比直接相加。

## 结果及边界

六类各前两道已观察开发题，120预检（含24 AR）、288相邻配对计时、12独立诊断/192真实norm状态，另20短warmup。EOS前最多128，未生成validation/reserve及新的完整4096答案。主要估计沿用执行前冻结的题级log平均、5000次类别内bootstrap；重复和norm状态不是独立题。

|方法|编译/eager时延比|题级分层95% bootstrap|时延下降|
|---|---:|---|---:|
|DFlash|0.973247|[0.972584,0.973876]|2.675%|
|ASD|0.973270|[0.972207,0.974349]|2.673%|
|ASD-Run|0.972470|[0.971559,0.973363]|2.753%|
|ASD-Utility|0.972586|[0.971686,0.973514]|2.741%|

分任务完整表见ASD_RMSNORM_RESULTS.md。每类只有2题，SQuAD时延下降约1.4%–1.7%，其他类约2.5%–3.2%；不从这一开发子集宣称所有场景普遍获益。与上一轮MLP-only的0.17%–0.29%相比幅度明显更大，但比较依据仍是各轮自己的配对基线。

60对跨profile预检token全部相同；eager全部重现旧CPU/AR轨迹，两个profile各自DFlash/AR均12/12一致，288次正式重复全部精确重现对应预检。48非AR案例的输出长度、调用密度不变；581轮除regret浮点数以外字段相同。因此本批差别发生在执行成本，而不是接受更多token、缩短答案或少做验证轮。

10/60对账本不完全相同，最大spent差9.5367431640625e-5。192个同输入norm输出只有13个逐位相同，最大绝对差4.57763671875e-5、最大RMS6.5462057e-7，floor1e-6的最大相对差约4.57e-7。接受条件未放宽；编译改变FP32归约数值，仍须视作单独执行profile。128相同不证明完整长答案或所有输入无损。

## 收益来自哪里

实际发现Target145、Draft22，共167处Qwen3RMSNorm，包含每层隐藏状态norm、Q/K head norm、最终norm和Draft hidden_norm。只绑定这些模块；MLP、attention、KV和接受实现与eager组一致。profile绑定/恢复成本中位约0.346ms、策略构造均已计入主时延，正式计时0次新增backend编译。

独立诊断覆盖每owner第一处input_layernorm、post_attention_layernorm、q_norm、k_norm和最终norm；Target各取prefill/首验证，Draft首次并加hidden_norm。192个clone输入的stride全部与原始相同；两profile各5warmup+20次交错模块回放。按角色对12题模块中位比取几何平均，编译/eager约0.910–0.949，即约5.1%–9.0%更快；这与上一轮MLP回放没有变快不同。小模块回放包含Python调用和同步边界，不是纯GPU kernel时长，也不能据此精确切分端到端成本。

主实验完成后只读检查8个生成文件（每卡4张图）：每图仅一次融合norm Triton kernel调用，名称为triton_red_fused_add_mean_mul_pow_rsqrt_0或triton_per_fused_add_mean_mul_pow_rsqrt_0，无外部GEMM。只上传名称、调用次数和文件SHA，不上传缓存源码。代码生成、同输入回放及相同轨迹的在线对照共同支持“norm融合减少执行成本”；本轮没有逐kernel原版profiling，不能虚构精确launch次数下降、内存带宽利用率或将全部收益唯一归因某项硬件瓶颈。该检查是运行后的解释，不改变冻结统计。

## 编译与资源

每卡4张共享动态全图，准备及所有诊断后图数不变；backend3.463/3.607秒，outer3.839/3.962秒（嵌套、不相加），controller约0.0034/0.0035秒。准备wall107.51/120.18秒还含两profile预热及60预检生成，不能全部算纯编译。独立空缓存各412257字节，约0.393MiB；正式两profile峰值allocated都35.158GiB，均在冻结预算内。

按本轮各卡当前混合请求，平均每对节省约31.35/31.26ms；仅outer+controller一次性成本约需123/127个同分布请求摊销。这个条件估算不包括未知部署冷启动/新形状编译，不是生产ROI保证。小批调用是否值得启用，应直接考虑完整准备需求。

两worker正常结束，服务器97 passed/17.53s，本地完整655 passed/5 skipped/77.17s。源码前后1349文件SHA相同；后审计17冻结文件、892旧结果SHA、此前6批215发布文件、7原运行时/接受文件和infra三引用，只有两项已知旧CRLF例外。旧MLP实际analysis内存重算逐字段一致，旧结果未改。infra面试版仍27e36199，默认服务配置不变。

## 接下来

保留RMSNorm为显式研究选项，停止在这12题上扫norm参数。下一阶段优先扩到现有48开发题和自然EOS：固定两profile及原四策略/AR，核查长数学、代码完整答案、短问答与JSON合同，保持逐题质量和数值分歧；通过后再冻结方案进入未观察验证集。应先验证这项收益和质量能否保持，再考虑更大编译边界或组合MLP；组合效果需要新对照，不能假定可加。

运行源码39f99df5352796f7c2d02bc2b1d030cc19bd5657；source SHA bbb852bff12a23779e7a1cad658396c5ae0c62c14ea38417faaf0c116e688d90；source ZIP SHA de57293d2ba00b1005cc42d1b62c72946e3670d865e6162998270b31b552dd54。服务器/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-rmsnorm，PID4048957；入口见ASD_RMSNORM_PROTOCOL.md。结果ZIP346034字节/SHA d310ffe98b9268b913de5bb1a61d4b46ebf6c7666e013c89e9b197340a527719。小型raw、图事件和测试在results/asd-rmsnorm。

复现：`.venv\Scripts\python.exe -m scripts.write_asd_rmsnorm_report`；后描述诊断`.venv\Scripts\python.exe -m results.asd-rmsnorm.inspect_results`；完整后审计`.venv\Scripts\python.exe -m artifacts.audit_asd_rmsnorm_postrun`，发布目录保存其副本audit_postrun.py，可用`.venv\Scripts\python.exe -m results.asd-rmsnorm.audit_postrun`重跑（需要本地原结果ZIP）。完整hidden/logits、权重与编译cache不发布。
