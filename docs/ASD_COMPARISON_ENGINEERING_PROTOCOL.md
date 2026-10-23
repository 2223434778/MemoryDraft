# 统一比较后的缓存组合检查（GPU前冻结）

本检查在统一比较confirm完成后执行，不改变原14组初筛、选择规则或72题确认。本文件先于读取任何confirm性能结果冻结。主要研究矩阵已经使用共同RMSNorm及CPU接受器；其通用执行收益继续引用已完成的独立配对实验，不相加成新速度。

GSM确认的两个候选按原质量损失/新增截断/自然时延/名称顺序选一个，与base DFlash、base ASD共3组。无收益也测试排序最前候选，不将其自动称为赢家。复用已观察的long-context8题及公开4K输入、共享3934-token登记段，容量2GiB/3 entries；不读题答案构建缓存，不训练或调参，不声称新独立题验证。

每题3组×off/cold/hit×2重复，共144完整自然4096/EOS计时，另8完整AR；两卡各4题，同题同卡。repeat0 off→cold→hit，repeat1 cold→hit→off，组次固定seed随机；每组冷前清缓存，活动KV/预算独立。旧适配器只读dense合并，基准切换在计时外；在线policy构造、缓存填充/复制/查找和推理均计入。冷编译在warmup内保留，正式不允许新增图，模型加载及评分单列，不含HTTP排队。

比较各组hit/off、cold/off及在相同缓存状态下candidate/ASD、candidate/DFlash。不拿候选命中对未缓存基线，不乘旧百分比。要求严格组完整匹配同题AR、输入hash不变、实际预算<=8；近似缓存配对若改变输出则完整保留并单列，不能据此声称固定输出计算加速。记录spent末位差异，不改接受器容差。

复用本轮FP32/highest/TF32关闭/eager attention/native16/RMSNorm。每worker1800秒、峰46GiB/编译8图/backend300秒/cache4GiB；canonical新源码单独冻结，新目录，不覆盖统一比较执行源。

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 ../.venv-dflash/bin/python -u -m scripts.run_asd_comparison_engineering --shard 0 --comparison ../asd-comparison-v2/results/raw/asd-comparison
```

另一张卡shard1。输出results/raw/asd-comparison-engineering；本轮仅公开结果和代码，不上传模型/KV/features。
