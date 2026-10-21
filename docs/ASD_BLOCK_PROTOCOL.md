# 更长真实Draft：结构及小成本预检

第一阶段48新开发题基线已完成，见ASD_SUITE_FINDINGS.md。本轮是路线表阶段1的预检，不以每类1题判定方向有效/无效；通过后再建立48题跨任务单因素比较和独立验证。

## 冻结设计

每类固定第一个development请求，不根据上一轮输出长度或正确率选择：GSM885、MATH329、MBPP262、HumanEval82、SQuAD5734025d4776f419006616c6、合成JSON12。验证/保留题不生成答案。

- 原模型revision、FP32/eager、batch1、greedy、非思考、输入、EOS及四个原策略配置保持。
- 真实Draft块16/24/32（含anchor，分别15/23/31 proposal）。DFlash、ASD、Run、Utility每长度均实际前向，不能用32裁剪模拟16。b16权重在24/32属于训练长度外推，不称为匹配长度训练的模型。
- 独立研究context只临时覆盖已核验的draft.block_size解码属性；原config.block_size仍16，正常/异常退出均恢复。生产generate_reuse守卫和RuntimeConfig不改。原DFlash forward本身依传入位置/noise长度计算，仍须真实结构/KV检查。
- 每题12臂自然上限4096各1次、遵守EOS前128各2次；AR完整自然与前128各1次，共6*(12*3+2)=**228次**。warmup单列。自然一次不作稳定端到端性能结论，两个前缀重复不算更多题。
- 同题所有臂在同物理卡固定种子交错。两个worker按六类位置奇偶分配，各3题/114行；请求间KV/预算重建。
- 每个16块臂与第一阶段同题同方法同上限的token、输入SHA、snapshot、轮次、账本、EOS完全一致，否则保存失败并调查。24/32前缀重复一致；严格D各长度都与同轮AR逐ID比较，形状浮点差异记录、不删题、不放宽接受规则。

## 必须报告

完整矩阵/独立空snapshot、原始文本及最终质量、全部错答/截断、真实proposed/accepted/验证轮、regret/预算、恢复属性、各长度严格AR一致性、16块重现和重复计时。机制及插桩不计主速度，本轮不训练/定价扫描/挑赢家、不修改验证规则。

长块可能既改变Draft预测，也改变remaining或exact-run取值及近似支出。严格D同长度对照必须保留，不能将更宽近似可行域误称为预测变准。先检查执行可行、长度外推及明显质量问题，再推进同样六类48题；不由1题/类否定整个方法。

本地`.venv\Scripts\python.exe -m pytest -q tests/test_asd_block_probe.py --basetemp .test-tmp-asd-block --junitxml=artifacts/asd-block-tests-local.xml`。真实tiny DFlash结构测试在有固定本地资产及Transformers4.57.3的服务器执行；随机小权重结构测试不能作真实预测准确率或速度证据。

服务器独立目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-block-preflight`，冻结完整源码SHA，原模型只读。入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_block_pipeline`；两子进程分别CUDA_VISIBLE_DEVICES=0/1，`--config configs/runtime-dflash.json --shard 0/1 --output results/raw/asd-block-preflight/part-0或part-1`。运行前后核对源码和旧结果SHA，保留失败不覆盖。
