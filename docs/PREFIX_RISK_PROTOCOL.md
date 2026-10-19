# 前缀风险辅助拼接：一次有终点的独立项目检查

2026-10-19，执行前冻结。已有14状态中Edit在13状态达到同池局部最大接受长度，因此不继续优化OT排序。本轮只检查低成本Draft前缀风险能否帮助选择交接位置。

## 策略

精确suffix优先；失败后用原DFlash块与原Splice-Edit候选池，仍为4-token内部锚点、最多8候选、4次lookup、cheap≥0.65、最多32token，不增加Target前向、不动态缩短、不调用OT。候选是原Draft前缀加原历史连续尾部，严格Target验证不变。

先按原Edit分数/较短边界/较早位置得到baseline。仅当baseline的`P(Draft接受长度 >= 拼接边界) < 0.5`，且有其他候选该概率≥0.5时，从这些候选中按原Edit顺序选一个。没有安全候选则保留baseline；不预测历史尾部长度，不因低概率直接拒绝提议。阈值0.5固定，不按本轮输出调整。

模型使用旧`results/acceptance/collection/states.jsonl.gz`的56去重状态/9项目，完全沿用acceptance.py固定正则与项目等权logistic，拟合所有旧项目后仅导出6个prefix系数。训练完成于新Target输出之前。另拟合无置信度消融，同一模型/数据/规则，只将概率/熵/概率差特征置零。系数与训练SHA保存`data/research/prefix-risk/calibration.json`。

仅当准入候选中存在多个不同边界时读取本次已有Draft logits，计算top1概率、归一化熵和top1/top2差；否则跳过。不能使用旧轮logits，不能读取Target验证结果作当前决策。新分支只通过默认关闭的matcher接口接收logits，不改变旧方法。全部softmax、entropy、CPU/GPU同步、检索/策略、KV维护及解码成本计入请求时延。

参考[SVIP](https://arxiv.org/abs/2411.18462v2)的不确定性信号，但这里只校准交接前风险，不声称复现其动态长度，也不把DFlash逐位置概率视为Target接受概率。

## 新输入与对照

`cases.json`包含6个从未采集Target答案的作者构造项目，各2请求，共12请求。主场景4项目/8请求：SQL清理、TOML导入配置、恢复步骤、JSON配额，均为短修改后恢复长段复制；控制2项目/4请求：原文复制、全文替换。用户独立、历史order2、请求order10/11，各请求独立使用旧快照，不将前一测试答案写入历史。

项目与旧校准9项目无交集；prompt包含当前源文，不能据此证明长期历史必要性。这是独立项目的受控检查，非真实生产流量。新AR完成后所有项目都成为已观察任务，之后不得再称新holdout或调参确认。

五组：Suffix-R、Guided-Edit、Splice-Edit、Risk-NoConfidence、Risk-Splice。基础配置沿用`configs/runtime-ot-benefit.json`：Qwen3-8B/DFlash固定revision、FP32/eager、greedy、batch1、4线程、512输出预算、同一Target输入/停止规则/用户快照。默认suffix和独立infra版不变。

## 执行顺序与停止条件

1. 新目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/prefix-risk`部署冻结Git原始源码，SHA核验，`models -> ../models`。先运行相关回归。
2. 一次生成12个新AR，逐一核对prompt身份及快照。所有请求必须完整文本合同正确（仅忽略末尾LF数量）且EOS结束，否则停止本轮，不调整prompt、不挑剩余成功项目继续。
3. 12×5=60次机制检查，记录实际候选/边界/风险/决定及事后接受标签；每行strict匹配相应AR。标签只用于事后审计。若Risk-Splice没有任何候选变化，停止计时，报告当前场景未触发风险决策，不能当成性能成功或普遍无效。
4. 若有候选变化，进行12×5×2=120次未插桩交错计时，固定种子按请求/重复打乱方法；逐行strict检查，并核对机制/计时计数路径一致。模型加载、预热、构造新AR单列，不混入在线加速比。

继续投入参考门槛：主场景Risk-Splice相对Splice-Edit总延迟至少降低3%、相对Guided-Edit至少降低1%；4主项目至少3个比Splice-Edit快；Risk-Splice总轮数少于Splice-Edit，并至少两个请求减少轮数；Risk-Splice比无置信度消融快；控制总延迟退化不超过3%。同时报告相对Suffix、每项目数据及固定种子5000次项目bootstrap区间；小项目数/两重复不支持生产或广泛显著性声明。

任一关键strict错误立即停止。完整门槛未达就报告局部正负结果，结束本轮，不更换阈值、权重、候选预算或新一批输入凑成功。旧结果逐文件SHA保护，发布只含紧凑特征/原始结果/命令，不含权重、DB、完整logits。

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_prefix_risk_probe --stage mechanism --config configs/runtime-ot-benefit.json --output results/raw/prefix-risk/mechanism
# 仅当mechanism provenance中的timing_gate.passed为true：
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_prefix_risk_probe --stage timing --config configs/runtime-ot-benefit.json --mechanism results/raw/prefix-risk/mechanism --output results/raw/prefix-risk/timing
```
