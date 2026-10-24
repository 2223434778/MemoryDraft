# 七任务指标补全：固定AR、接受长度与独立奖励模型

2026-10-24，在补测生成与奖励评分前冻结。补充原asd-paper，不覆盖原4980计时/249 AR及失败裁判，不调ASD、树或任务选择。保留全部五组；不增加训练seed。

## 同工作量AR

原241源任务/249匹配请求不变，MT第二轮仍用原严格AR首轮历史。沿用Qwen3-8B/DFlash、原物理卡、FP32/highest/TF32 off/eager/RMSNorm。每请求一次DFlash-before→AR→DFlash-after，均256token、忽略EOS，共747正式生成，每卡2个预热。AR复用原engine.generate，只在该调用期间屏蔽EOS，finally恢复；不新建解码引擎。

逐token对照原fixed DFlash256，输入SHA/快照不变；不匹配则保存现场，停止该worker。所有请求均保留，3%前后/跨会话漂移只作诊断，不据结果删题。Target-only TPS和当前严格DFlash/AR比为本轮直接配对测量；原五组TPS/新AR是跨会话比值，另用逐请求新旧严格DFlash耗时比校准，明确标为估计，不冒充重新同窗测量五组。不得把文献AR与本机时延相除。

源码和旧结果前后SHA；每worker7200秒，outer7800秒；峰allocated46GiB、cache4GiB，每卡最多8图/正式新图0。模型加载、预热、编译和旧表读取成本分别记录；本轮严格控制不读取构树成本表。

接受长度直接复算旧轮次：tau_selected=日志accepted总数/轮数；tau_emitted按generated_before和实际输出长度裁剪末轮后再除轮数，均不包含Target bonus。另列输出token/Target调用、验证节点/输出token、提议量；树不能用线性候选接受率分母冒充相同计算成本。自然/128/fixed分开。旧AR自然耗时可作为同次原始参考，但与新fixed表分开。

## 开放任务补评

使用独立训练的[Skywork-Reward-V2-Qwen3-8B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-8B)，revision与逐文件LFS SHA256/Git blob固定在configs/asd-reward-model.json。Apache-2.0，原模型卡建议16,384token上限且不加system prompt。权重约15.15GB，只保存本地artifacts/models；不上传GitHub、不调用外部推理API。依据官方SequenceClassification接口读取原始标量reward，BF16/SDPA、batch1、不生成裁判解释或A/B标签。

本机RTX3090离线评分，不计生成wall；峰22GiB、预期运行上限1800秒，下载最多四次失败重试/文件、每次网络读取timeout60秒。先运行8对独立人工构造明显好坏控制（数学、代码、JSON、事实、翻译、计数等），至少7/8排序正确且反序重算绝对误差<=1e-5才进入任务评分；失败保留，不改控制/模型/阈值重试筛分。

全部24单元×5方法=120条评分；16Alpaca自然输出、8MT各自真实双轮。逐回答pointwise，无方法名、无A/B呈现顺序；固定随机顺序，完全相同对话按SHA复用同分，超过16384标未评、不裁剪。相对ASD：高分胜、低分负，完全相同或差<=1e-4计平局；报告配对覆盖/胜负平、逐任务差与输出长度，不把reward当准确率或官方MT-Bench/AlpacaEval。它与Target共享Qwen3基础模型，但使用独立偏好训练权重，仍可能有长度/风格/训练数据偏差。

旧192条第一token裁判失败证据保留。本轮是预登记的不同评估工具，不以新偏好来重调生成策略、挑题或撤回旧失败。变化最大及客观数学/代码题另做任务级核验。

## 复现

本地：`python -m pytest -q tests/test_asd_paper_metrics.py tests/test_asd_paper.py`。

下载：`python -m scripts.download_asd_reward --directory artifacts/models/skywork-reward-v2-qwen3-8b`；固定源本地副本：`python -m scripts.run_asd_reward --model-directory <已校验模型绝对目录>`。

服务器每卡：`CUDA_VISIBLE_DEVICES=<0或1> OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_paper_metrics --shard <0或1>`。新root asd-paper-metrics，原asd-paper只读；原始输出results/raw/asd-paper-metrics，最后发布results/asd-paper-metrics。补充报告另存，不重写原论文任务表。
