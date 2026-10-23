# ASD论文七任务：五方法扩样本协议

2026-10-23，所有新模型生成之前冻结。主问题：成本受控树+ASD与Joint是否在更大数学样本保留收益，在其他论文任务上有何适用边界。不以全任务均胜出作为成功前提，不根据新输出删题/改参；旧开发正负结果完整保留。

## 来源、选择与范围

采用[ASD论文](https://arxiv.org/html/2608.03447v3)的七个任务名：GSM8K、MATH-500、HumanEval、MBPP、MMLU、MT-Bench、Alpaca。延用本项目Qwen3-8B/DFlash、FP32/highest/TF32 off/eager/RMSNorm，不称DSpark/L20/原论文完整prompt或切分复现，不拿跨硬件论文TPS直接相除。

冻结241源任务：GSM64、MATH32、HumanEval32、MBPP32、MMLU57（每学科1）、MT-Bench8（每类别1，双轮）、Alpaca16（有/无input均衡）。MT展开后249个匹配输入主请求。样本不是全量数据集，不将小类别数解释为整个分布。来源版本/原始SHA/许可在data/research/asd-paper/manifest.json，selected manifest SHA56ccef799adfffe2caf0c6f5bd807484bfbc831563a2b90f80e041536078eca4。

按固定种子MemoryDraft-asd-paper-v1:<ID>的SHA排名；MATH学科轮转，MT类别轮转。排除旧data/research与results中出现的全部四类源ID，加保守早期范围；1049文件原始SHA、235排除ID保留exclusions.json。所有旧data中已登记的validation/reserve也保守排除，不把未运行但已登记题回收。没有用新生成长度、答案或速度筛题；不改旧数据。MMLU/MT/Alpaca在本项目首次引入。训练集GSM ID与test空间分离。

GSM/HumanEval/MMLU保留MIT归属；MBPP/FastChat保留Apache-2.0来源许可；MATH保留原source MIT及HF card的既有归属说明。Alpaca数据为CC BY-NC 4.0，仅此研究数据使用，对应DATA_LICENSE/README另存，不将项目代码许可套给数据或面试版。只使用选定公开数据，不新增上游推理代码或权重。

## 五组与固定执行

DFlash、普通ASD、严格BudgetTree、BudgetTree-ASD、JointTree-ASD。共同Target/Draft权重、精度、模板、greedy、停止规则和B8/g.25/m2；Joint只读剩余账本参与主干长度，接受规则保持。树仍Top4、min8/max32、native16，成本表按物理卡沿用第一阶段已发布SHA。无重新校准/训练/新seed/阈值扫描，默认服务和infra引用不变。

各任务按族内选定顺序偶数GPU0、奇数GPU1。同一卡先数学96题，再余任务；第一阶段不基于性能筛选方法，全部五组进入扩展。自然EOS/4096上限各2次；128且EOS各1次；全部任务另256固定token/EOS忽略各1次。第二次自然方法顺序反转第一次的固定随机排列，降低顺序漂移。共4980正式计时+249同题自然AR；每卡9次预热/zero检查，另40次MT自身历史第二轮。重复不是额外题，也不要求换训练或生成随机种子。

自然严格两组逐token/text/停止与同题AR一致；固定256的两个严格组彼此逐token相同。fixed仅测计算吞吐，不用EOS后的内容评分。所有重复轨迹稳定，同请求/同测量条件Target输入SHA与快照相同；仅实际选中路径扣费，B<=8。遇到实现/数值/覆盖/输入/预算失败保留原始证据，停止有问题的owned运行，不放松门槛继续宣称无损。

模型加载、只读成本表、预热/冷编译、离线质量裁判另列。正式wall含请求构造/验证器/全部构树/TopK/传输/heap/mask/Target/Draft/KV整理。峰allocated<=46GiB、cache<=4GiB/卡、正式新图0，每worker14400秒、离线本地judge3600秒、outer19000秒。只管理runner自己的进程组。源码canonical Git blobs逐文件前后SHA；禁止权重/KV/logits/features/私有数据发布。

## 多轮输入与质量

MBPP接口补充在任何新模型生成之前完成：部分题干未给函数名，从原标准解法中只提取评测所调用的函数签名，不附函数体、helper、测试或预期输出。32题ID不变，数据SHA随明确接口更新；不让模型猜评测API。

MT主性能采用匹配历史：首轮同原问题，次轮所有组使用严格AR首轮回答作公共已实现历史。单列其输入SHA与history policy。另跑每组使用自身首轮回答的第二轮，形成完整真实双轮，作为对话质量/端到端描述；这些第二轮的Target输入可以不同，不混入匹配输入主速度表。ASD预算按每轮请求重置8，所有组相同；不是整个双轮只允许8。

数学保留既有最终数值/boxed主评分、有限标量等价与needs_review，未解析/符号式差异另复核。HumanEval官方测试、MBPP原3测试全部不入prompt；MBPP可用challenge tests另列（不是新声称EvalPlus或私有测试）。代码指标为单次greedy的标准测试通过率/pass@1，沿用现有静态限制与限时子进程，不声称完整OS安全sandbox。与旧MBPP“把公开测试给模型”的设置不同，不能跨批直接比较质量。217个可评分参考解法在模型前本地预检全部通过；服务器重做并记录，若参考测试失败保留任务/标明来源问题，不偷换题。

MMLU零样本生成A/B/C/D，对最终字母评分，不冒称完整57学科足样本估计。MT/Alpaca不伪造准确率。开放质量另做24个单元（8自有历史双轮+16指令）的本地盲评：同Qwen3-8B Target，方法名隐藏、ASD与其余4组各A/B互换，共192条判断。完全相同答案直接平局；其余固定rubric比较正确性/相关性/指令遵循/完整性，不奖励啰嗦。取首token的A/B/C限制概率argmax，不生成解释或额外采样；材料>4096输入token标未评，不悄悄裁剪。记录覆盖、顺序一致性、偏好（平局半分）、模型身份与成本。

这是有自评偏差的辅助审查，不是独立GPT-4评委、官方MT-Bench评分或AlpacaEval胜率；不调用外部付费API。官方MT-Bench/AlpacaEval质量复现仍需独立评委和对应原始评测包，此限制应在结论保留。固定参考解法/公开GPT-4参考只做评分复核，绝不进入生成提示或历史检索。

## 统计与判读

各族独立报告：自然题等权配对时延比、总token/总wall TPS、长度/截断/质量lost-gained；共同完整128子集及N；七任务fixed256；请求bootstrap区间。固定生成量与自然结束结果分表，不能用强制EOS后续写宣布完整请求提速。主比较普通ASD、第一阶段组合；严格树为独立贡献对照，Joint不能只对较弱基线报告。

机制统一为每Target调用推进token、验证轮数、每输出token验证节点、实际regret。树分支改变接受率分母，因此不能只凭树节点接受率排名。记录基线两次计时比和异常，不按漂移删有利/不利样本；这不是原论文四次strict-ASD-strict triplet协议。代码/质量评分不计入推理wall，本地开放裁判同样单列。后验按长度/预算分层只作机制观察，不训练新路由或用新标签重调算法。

## 命令与位置

准备数据：`.venv\Scripts\python.exe -m scripts.prepare_asd_paper`，需要已固定公开source cache。

相关检查：`.venv\Scripts\python.exe -m pytest -q tests/test_asd_paper.py tests/test_asd_joint_tree.py tests/test_asd_tree_report.py tests/test_joint_tree.py tests/test_budget_tree.py`。

冻结提交后：`.venv\Scripts\python.exe -m scripts.package_asd_tree_source --output artifacts/asd-paper-source.zip --manifest ASD_PAPER_SOURCE_MANIFEST.json`。

服务器新root `/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-paper`：`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_paper_pipeline`。

原始results/raw/asd-paper，公开results/asd-paper；复算`python -m scripts.write_asd_paper_report`。准确commit、目录、源/结果SHA、真实执行记录追加PROGRESS与报告，不只提交计划。
