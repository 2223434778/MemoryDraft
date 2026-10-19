# 两个已有案例的风险分支覆盖检查

执行前冻结。12请求续测已完成60机制/120计时，风险评估与改选均为0：没有合格候选进入风险模型。这不足以验证风险交接分支，故补充两个已有旧案例，固定只检查payment-runbook:1与shipment-contract:1，不增加新的随机场景。

两项目属于原56状态训练项目，答案/候选早已观察。选择它们明确是为了覆盖已知风险/安全边界，不能称新数据、holdout、泛化或确认性能；不能合入12请求的主结果。只报告已知机制能否激活、同状态接受差异及其本例在线成本。

本轮已在CPU用原固定系数核对：payment/15原边界14的存活概率约0.109、替代边界7约0.868；shipment/14两边界都不安全，应保持原候选，shipment/22原边界6安全，应保持原Edit。此检查不拟合或调整任何参数。

模型/Target/模板/严格验证、旧系数、0.5阈值、top8/4次查询及32上限全不变。复用两请求旧AR参考，用当前tokenizer复核prompt/快照。五组与12请求完全相同，各一次机制共10次、两次交错计时共20次。记录全部在线成本，计时与机制计数路径严格一致。任务质量单列不阻断；任何token/prompt/snapshot/来源或计数错误仍立即停止。

在新独立服务器`prefix-risk-controls`执行，原729文件执行场景不修改。原结果及12请求主结果逐文件SHA保护；输出`results/raw/prefix-risk-controls`，发布在`results/prefix-risk/continuation/controls`。不根据两个控制的输出追加参数或更换案例。

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_prefix_risk_controls --config configs/runtime-ot-benefit.json --output results/raw/prefix-risk-controls
```
