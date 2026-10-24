# ASD七任务独立扩样与预算匹配消融

2026-10-24，在本轮模型生成之前冻结。目标是验证成本受控树与ASD组合的跨任务收益，保留质量与停止行为的边界，不预设所有任务胜出。

参考[ASD v3第4节](https://arxiv.org/html/2608.03447v3#S4)：固定生成量比较计算吞吐，自然结束另审质量、长度、轨迹变化；同预算比较并做单参数控制。论文每条件四次计时，本轮自然两次、固定相邻严格对照一组，不冒称原协议精确复现。继续使用DFlash/Qwen3-8B/A6000 FP32/RMSNorm，而非论文DSpark/L20。

## 新数据与四组

从已固定版本/SHA/许可的公开源缓存选择514个此前未用源任务：GSM8K128、MATH64、HumanEval64、MBPP64、MMLU114（57学科各2）、MT-Bench16（8类别各2，双轮）、Alpaca64（有/无input均衡）。展开为530匹配请求。固定SEED字符串只用于ID的SHA排名，不新增训练或生成随机种子。

排除全部已跟踪data/research和results的七类ID，含旧验证池/保留池；保存1117个旧载荷SHA和476个保守排除ID。未按模型回答、长度、质量、速度选题。数据原始文件和选中manifest保持，默认生成提示沿用旧七任务工厂；MBPP只暴露所需函数签名，不暴露实现/测试。参考解法在生成前作预检，来源测试失败保留任务、不偷换。

四组DFlash、ASD、严格BudgetTree、BudgetTree-ASD全部进入七任务。共同Target/Draft、greedy、native16、停止规则、B8/g.25/M2。成本表沿用原物理卡已固定SHA；本轮补记只读表加载秒数。没有新训练、模型下载、路由或主参数选择。Joint旧消融保持。

每请求自然EOS/4096上限各2次、128且EOS各1次；方法顺序固定打乱、自然第二次反序。固定256/忽略EOS分别执行DFlash-before→ASD→DFlash-after、严格树-before→树ASD→严格树-after；两组三连顺序按ID预定。因此主计时9540，另530个自然AR参考、64个各方法真实自身历史MT第二轮；每卡8预热/零预算检查另列。

相邻strict前后必须逐token/text/账本相同，严格树与DFlash固定输出相同；固定严格前缀亦核对同题自然AR。所有主方法同请求Target输入SHA/快照相同；自然strict匹配AR全部token/text/停止。自然重复轨迹稳定、所有实际支出<=8。baseline漂移>3%全量保留并记录，不据此删除慢题或错题。固定strict统计取前后wall均值，原始两角色不覆盖。

MT匹配性能第二轮全部沿用同题严格AR首轮；独立真实双轮由各方法自身首轮续写，另列质量和端到端描述，不混作同输入配对比较。

## 预算匹配控制

各任务族在固定选中顺序的前4源任务共28源任务/32请求，生成前登记。ASD和树ASD都用B4、B12、g.125、g.5、M1五个单参数控制（其余保持B8/g.25/M2），各自然/固定一次，共640额外生成。B0/B8复用严格/主方法首重复，不重跑相同动作。逐条核对实际选中路径regret、cap和request ledger。

这些小切片用于解释速度—质量曲线，不根据其结果改主配置、选择新赢家，不声称完整复现论文所有gate/cap或无界控制。控制曲线开放输出保留，但未纳入主四组质量评分时不伪造准确率。

## 质量、成本与边界

沿用保守数学评分及有限标量补充、代码原标准测试（不入提示）、MMLU只答字母合同；MMLU可提取首选项补充单列，不改主评分。自然时延、总tokens/秒、输出长度/截断、lost/gained、共同128子集及其N、固定吞吐分别报告。实际接受Draft长度裁末轮并排除Target bonus，验证节点/输出token和调用量并列。按旧轨迹分层为后验解释，不训练路由。

开放80单元×4方法=320条，用既有固定Skywork Reward V2 Qwen3-8B权重和8控制评分；本机3090/BF16/SDPA，无A/B位置，超过16384标未评、不裁剪。使用原始标量reward，报告覆盖/胜平负及长度，仍是独立偏好训练代理，不是官方GPT-4 MT-Bench或AlpacaEval；本轮不调用付费外部推理API。明确属性/较大差异另复核，所有负结果保留。

请求wall包含全部在线构树/验证/选择/传输/KV开销；模型加载、成本表读取、预热/冷编译及离线评价分列。每卡峰allocated<=46GiB、cache<=4GiB、正式新图0。两个worker21600秒，outer22800秒；只操作owned进程。源码canonical Git blobs在前后逐文件校验，旧结果SHA和infra三个引用保持。公开小型文本/token/轮次账本和哈希，不发布权重/KV/logits/features或私有历史。

## 命令

```powershell
.venv\Scripts\python.exe -m scripts.prepare_asd_paper_extension
.venv\Scripts\python.exe -m pytest -q tests/test_asd_paper_extension.py tests/test_asd_paper.py tests/test_asd_paper_metrics.py tests/test_asd_tree_report.py --basetemp artifacts/pytest-paper-extension-tests
.venv\Scripts\python.exe -m scripts.package_asd_tree_source --output artifacts/asd-paper-extension-source.zip --manifest ASD_PAPER_EXTENSION_SOURCE_MANIFEST.json
```

新隔离root `/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-paper-extension`，`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_paper_extension_pipeline`。

本机奖励：`python -m scripts.run_asd_reward --study asd-paper-extension --input results/asd-paper-extension --model-directory <已校验的本地权重绝对路径>`。统计复算：`python -m scripts.write_asd_paper_extension_report --input results/asd-paper-extension`。实际执行源commit、SHA、命令、资源与结果在PROGRESS与最终报告追加，不以协议代替结果。
