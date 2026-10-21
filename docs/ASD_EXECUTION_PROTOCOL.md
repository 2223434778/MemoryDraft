# ASD 原生16块：执行成本诊断（运行前冻结）

承接完整48开发题长块结果。当前阶段先量化模型执行、验证器和同步开销，不实现新接受算法，不扫描阈值/精度/块长，不使用 validation/reserve。来源仍是已观察开发集，诊断不能作为新独立验证。

## 固定条件

六类各取 development 固定顺序前两题，共12题；与前阶段机制子集一致。原 Qwen3-8B / DFlash b16 revision、FP32/eager、greedy、batch1、非思考、空历史、Target prompt、EOS及预算规则全部沿用。原生严格 DFlash、ASD、Run、Utility 四组，B8/g.25/M2、Utility价格2。每请求重建KV和预算。

每题遵守EOS生成最多128，正式时延每组3次，共144次。每类第一个/第二个题分别在GPU0/1，同题同卡随机交错，种子`execution:<id>:<repeat>`。输出不足128不强制续写；所有 token/输入SHA/snapshot/轮次/账本/停止原因须逐项重现完整长块阶段的16块128前缀。先完成该题正式计时，再运行该题全部诊断，诊断时延不进入性能统计。每worker四个32-token warmup独立记录。

## 三类测量及限制

1. **同步请求时延**：144次，沿用 engine.generate 全在线成本边界，不减去任何开销。按题合并重复、六类分别列4组，不用12题替换已有48题结论，不做质量非劣声明。短前缀不重新评分完整任务。
2. **CUPTI profiler**：48次，CPU+CUDA，关闭shape/stack/memory采集。在既有Target prefill、Draft前向（含lm_head/argmax）、Target验证前向外，增加accept_decision（含Target argmax及接受判定）、commit_kv（已确认token/KV/特征）、publish_stop（回调及停止扫描）范围。模型调用异步，CPU范围包含等待；scope host耗时与GPU时间不能相加。GPU user annotation不当kernel；仅累加每个CPU event直接关联的真实kernel一次，以最近范围归属，不累加父范围的inclusive device_time。原始设备事件另求时间区间并集及跨度，保留未归属时间。同步API的次数/host耗时按最近范围记录，等待不等于该范围自身计算。大trace不发布，仅发布小型聚合；若缺CUDA事件/链接，失败而非补零。插桩生成须与该题正式轨迹完全相同。
3. **输入已就绪的真实状态回放**：ASD/Run/Utility另36次捕获生成，只复制第0/4/12轮（不足则不补）的真实proposal、对应K个Target logits和请求已花预算，最多108个状态，逐题用完释放，不上传完整logits。捕获不能混入profiler。每状态原策略和严格前缀核心交错，5次warmup后各30次；构造策略/恢复预算/前置CUDA同步在计时外，调用及末尾同步计入。每次决策须重现捕获accepted/strict/relaxed/regret。空同步控制同样各30次，单列不相减。严格核心只对K个候选logits，省去在线K+1的bonus位置处理，不能把它当完整原生DFlash替换实测。回放没有前序模型队列，仅反映resident输入时接受算法成本，不代表删除在线等待后一定能获得相同加速。

最多228次生成（144计时+48profile+36捕获）以及至多108*3*35次小张量回放/空同步检查，8次warmup单列。全部诊断与计时逐ID/账本一致；旧结果SHA执行前后保持。报告accept_decision host等待与resident验证成本的区别，不用相加或相减制造提速，不在看到结果后更改样本。

## 执行

本地：`.venv\Scripts\python.exe -m pytest -q tests/test_asd_execution.py tests/test_asd.py tests/test_dflash_reuse.py --basetemp .test-tmp-asd-execution --junitxml=artifacts/asd-execution-tests-local.xml`。部署前提交源码/协议/测试，导出Git原始blob/SHA到独立`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-execution`，只读链接已有模型，前后核验全部文件，保留失败现场。

服务器：`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_execution_pipeline`。两worker分别`CUDA_VISIBLE_DEVICES=0/1 ... -m scripts.run_asd_execution_probe --config configs/runtime-dflash.json --shard 0/1 --output results/raw/asd-execution/part-0或part-1`。报告与原始小型结果发布GitHub供复核，infra冻结引用不动。

计时口径依据已核验的[PyTorch 2.5.1 profiler源码](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/autograd/profiler.py)及[FunctionEvent实现](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/autograd/profiler_util.py)：device_time_total递归包含子事件，self计时才排除子事件；本轮直接链接kernel归属避免重复累计。
