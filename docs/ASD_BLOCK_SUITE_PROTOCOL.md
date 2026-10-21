# 完整48开发题：真实Draft块长单因素比较（执行前冻结）

承接ASD_BLOCK_PROTOCOL.md的六题预检。本轮不筛选Utility24看起来有利的题，保留原六类各8题、13组全部同轮对照。原48验证及96保留题不生成模型答案。

## 固定输入和方法

`data/research/asd-suite/manifest.json`原192池不改。只取全部48 development：GSM8K、MATH-500、MBPP、HumanEval、SQuAD、合成JSON各8。来源、许可、参考评分和主/辅助分数沿用原模块；不根据本轮长度、质量或重复耗时删题。与前阶段是同一开发集，不能冒充48道新独立验证题。

原Qwen3-8B revision b968826d9c46dd6066d109eabc6255188de91218 / DFlash revision9b41424b7109f9c5413454f481b09a82b85333f4；FP32/eager、greedy、batch1、非思考、空历史，Target prompt及停止/接受规则不变。没有新模型下载或训练。

四原策略DFlash/ASD/ASD-Run/ASD-Utility × 真实块16/24/32，加独立AR，共13组。近似组B8/g.25/M2、Utility价格2不变；严格D为B0原生路径。使用已验证的临时解码属性，训练config仍16，退出恢复；24/32是b16权重长度外推，不是重新训练的匹配模型。生产运行时/默认守卫不改。每长度真正前向起草，不用长Draft裁短冒充短Draft。

## 计时与完整性

- 每组每题自然EOS上限4096，重复3次；遵守EOS的前128重复2次。48*13*(3+2)=**3120次计时生成**，两卡各1560。
- 每类前2个开发题的12个非AR组合另做完整自然机制回放，12*12=**144次**；各卡72。合计3264次生成，短warmup13/worker单列。插桩耗时不进入时延比较。
- 两worker保持原第一阶段的每类题目位置奇偶分配，同题各组/重复同卡。固定种子`block-suite:<id>:<stage>:<repeat>`随机交错13组。每请求KV和预算重建，engine.generate全部在线成本计时，评分/落盘在计时外。
- 所有重复token、轮次、账本相同，否则保留现场、停止该运行，不合并成功/失败。自然质量仅在同轨迹下复用首次评分，不把重复当新质量样本。
- 全部16块与原第一阶段同题同方法的输入SHA/snapshot/token/轮次/账本/质量/EOS重现。严格D每长度与同轮AR自然/128共288对逐ID检查；浮点形状分歧保留首位置，不放宽验证或删除题。
- 自然/前缀仅允许最后上限裁短块改变ASD门控，边界按实际块长15/23/31识别，记录更早分歧，不以裁短解释掩盖它。

机制回放输出/账本与计时轨迹一致。新trace保留全部proposal IDs、对应Target top IDs、mismatch、regret、values、accepted、strict、feasible、spent，独立标量实现用真实候选EOS重算决策；不再枚举未观察EOS。没有记录完整词表logits或最终bonus logit，不宣称能重算全部原始浮点分数或完整Target前向。

## 预定比较与解释

分别对六类报告自然请求中位时延、全部正确数/待复核/受限未评估、EOS/截断、token、TPS、调用密度、regret和重复IQR/极差。保持MATH保守主评分及有限标量辅助；人工复核另存，不覆写原分数。SQuAD EM/F1，代码提供测试，JSON严格类型。参考答案不能进入Target输入或历史。

主要长度比较为每个策略24/16和32/16；保留32/24描述。每个长度另列ASD/D、Run/ASD、Utility/ASD、Run/D、Utility/D；严格D各长度对AR。另预定Utility24/ASD16、Utility32/ASD16，避免只相对Utility16较弱参照显示收益。这样既看长度效果，也看策略独立增量，不将同一regret预算误当相同质量风险。

每题先合并三/两次计时，以题为单位配对几何均值及固定种子5000 bootstrap；每类仍8题，不能作质量非劣保证。新增比较不挑最小p值，不用本批选择最终赢家，不把重复当更多样本。六题预检与本轮48题的重叠6题单列，另外42题仅作开发迁移诊断，不称正式独立验证。

自然所有错答/截断保留。共同EOS和共同正确EOS是事后诊断子集，不替换主结果。固定128比较要求两组均实际达到128，短答案不强制续写；不足5个等128样本只描述点值，不展示区间。

时延比=输出长度比×Target调用密度比×每调用摊销成本比；每调用成本包含prefill、Draft及全部在线开销，不当纯Target内核耗时或因果贡献。另列有效提交的proposal接受（裁掉EOS后未输出部分）、验证位置、真实完整Draft噪声块计算量；操作计数不能替代实际计时或FLOPs。

本轮不改价格/门槛/预算/题目，不消耗验证池，不因某类别均值不好否定整条方向。先解释跨任务差异，完成后再冻结后续机制或验证方案。

## 执行与发布

本地先`.venv\Scripts\python.exe -m pytest -q tests/test_asd_block_suite.py --basetemp .test-tmp-asd-block-suite --junitxml=artifacts/asd-block-suite-tests-first-local.xml`，随后相关真实KV与完整回归。部署前提交完整源码、导出Git原始blob/SHA，在独立`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-block-suite`创建目录，仅只读链接现有模型。

入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_block_suite_pipeline`。子进程`CUDA_VISIBLE_DEVICES=0/1 ... -m scripts.run_asd_block_suite --config configs/runtime-dflash.json --shard 0/1 --output results/raw/asd-block-suite/part-0或part-1`。执行前后源码/旧结果SHA核对；失败不覆盖，发布公开/合成压缩trace与测试、报告，不上传权重或私有记录。infra冻结三引用不动。
