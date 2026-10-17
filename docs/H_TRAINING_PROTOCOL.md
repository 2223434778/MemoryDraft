# 固定 Target 的小规模 H 蒸馏与成熟 Draft 验证

2026-10-17。用户授权执行，运行前固定。复用现有记忆快照、Transformers 验证/KV 与 PEFT；不从零开发推理引擎。

## H 训练与评估

- 新建公开合成项目，24 train、8 dev、8 test，项目/用户 ID 完全分离。四类任务为标签归一化、数值筛选、发布清单、JSON 配置；每类内部共享任务结构，不能声称自然用户或跨任务家族泛化。旧六项目不参与训练、选参或新测试。
- 先用固定 Qwen3-8B FP32/eager/non-thinking AR 生成全部模拟过去回答并冻结。过去请求 order=0，回答 order=1；当前请求 order=10。旧回答仅依赖旧任务。训练标签只来自 train 项目的当前 Target 回答，不进入检索历史；测试当前回答在权重及开发选择冻结后生成。
- 每个 train 项目同时构造 D_format 空历史、H 相关旧片段、H 无关片段三个样本；三者预测相同的固定 Target 输出，prompt token 的训练标签全部为 -100。H 用低成本词法 top-1 检索、128 token 历史预算，训练与推理一致；R 始终使用完整合法快照。
- 复用 PEFT 0.11.1 LoRA：rank=8、alpha=16、dropout=0、q_proj/v_proj，AdamW lr=1e-4，batch=1、累积4、四轮、seed=20261017。保存每轮开发响应 token loss，按开发 loss 选择 checkpoint；不按测试结果继续训练。是序列级硬标签蒸馏，不冒充 DistillSpec 的 on-policy/KL 完整复现。
- 部署前小测试发现全局旧 bitsandbytes 不支持当前 CUDA 12.4；因此训练使用独立 `.venv-train`，仅覆盖 bitsandbytes=0.45.5，保留 torch 2.5.1、Transformers 4.52.2、PEFT 0.11.1。训练不用量化；此依赖用于兼容 PEFT 的导入，不改原环境。
- 模型、精度、Target 模板及验证规则保持旧配置。训练只更新 Draft LoRA，合并后 D/H 共用同一模型；原始 Draft 作为配对对照。LoRA 权重不上传 Git。
- 开发集先比较 D_native、D_format、H 的同前缀连续接受长度，H/D 候选长度各在 {2,4} 中选择，并计入完整生成成本。主 H 质量对照是同权重 D_format，D_native 同时报告。检索及短历史选择均在生成当前评分答案前完成。
- 系统阶段比较 AR、原始/训练后 D、H、同一 RHD 路由 H-off/on；门控版仅当同权重 H 在对应任务开发子集同时改善质量与端到端耗时时启用。门控先判断任务再决定是否检索，H 首次 prefill/追赶全部计入。若开发无获益任务，明确记录门控关闭全部 H，不宣称 H 系统成功。
- 最终 test 八项目、三重复。只做一次预声明小试验，完整保留负结果。H 需在未见项目、同权重同前缀下改善候选质量，并在计入全部成本后胜过同权重 H-off 才算通过；不会只凭小模型训练后的整体改善宣布历史有效。

## DFlash 基线

- 模型 `z-lab/Qwen3-8B-DFlash-b16` revision `9b41424b7109f9c5413454f481b09a82b85333f4`，模型卡 MIT，权重 SHA `c702878094f38ad5843e6ac40b327720b3147ffbf669a5d9ca1974865ca6c080`。上游仓库 `z-lab/dflash` commit `07ebd93db9f472af339b644bb70221ad8428328a`，已读取 MIT 许可。官方模型快照内自定义代码单独核验 SHA，保留许可于忽略的模型目录，不复制到核心包。
- 独立 `.venv-dflash` 使用 Transformers 4.57.3；保留现有 torch 2.5.1，不覆盖原实验环境。先使用 FP32/eager 和现有 Target 权重做严格 greedy 一致性。环境兼容性未通过时保存实际错误，不冒充论文标准环境复现成功。
- 直接调用审计后的官方 `spec_generate`。先小规模正确性检查，再在旧六合成项目加六公开请求上对照同环境 AR，三重复、max_new_tokens=128、官方 block_size=16。模型加载/预热不计，在线 tokenization、生成、同步及解码计入；输出与同环境 AR 逐 token 核对，并另核对旧 AR。首次不一致停止并保存证据。
- 不把 DFlash 的底座提升算作 H 收益，不宣称已经完成 R/DFlash/H 融合。若后端数值行为变化，与旧 FP32 参考分开标注。

论文依据：[DistillSpec](https://arxiv.org/abs/2310.08461)、[OSD](https://arxiv.org/abs/2310.07177)、[DFlash](https://arxiv.org/abs/2602.06036)。论文提供设计依据，不替代本项目实测。
