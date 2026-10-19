# 风险拼接：两个已有训练项目的机制覆盖

这两个案例为已知训练项目，特意选择以覆盖风险边界分支。答案/候选早已观察，不能作为独立泛化证据，不并入12请求续测主结果；没有调整模型、阈值、长度或候选池。

10/10机制、20/20计时严格匹配各自旧AR token/prompt/快照；20/20计数路径匹配机制。全部在线成本计入，未重新构造AR。

| 请求 | 方法 | 验证轮 | 两次总秒 | 改选 |
|---|---|---:|---:|---:|
| payment-runbook:1 | Suffix-R | 8 | 1.596651 | 0 |
| payment-runbook:1 | Guided-Edit | 7 | 1.484506 | 0 |
| payment-runbook:1 | Splice-Edit | 8 | 1.669519 | 0 |
| payment-runbook:1 | Risk-NoConfidence | 7 | 1.520009 | 1 |
| payment-runbook:1 | Risk-Splice | 7 | 1.519649 | 1 |
| shipment-contract:1 | Suffix-R | 9 | 1.782422 | 0 |
| shipment-contract:1 | Guided-Edit | 8 | 1.671890 | 0 |
| shipment-contract:1 | Splice-Edit | 8 | 1.712927 | 0 |
| shipment-contract:1 | Risk-NoConfidence | 8 | 1.711105 | 0 |
| shipment-contract:1 | Risk-Splice | 8 | 1.711838 | 0 |

| 实际改选 | 位置 | 原边界→新边界 | 原候选→新候选同状态实际接受 |
|---|---:|---|---|
| payment-runbook:1/Risk-Splice | 15 | 14→7 | 7→32 |
| payment-runbook:1/Risk-NoConfidence | 15 | 14→7 | 7→32 |

payment-runbook:1：Risk相对Splice-Edit延迟变化-8.98%，相对Guided-Edit +2.37%。

shipment-contract:1：Risk相对Splice-Edit延迟变化-0.06%，相对Guided-Edit +2.39%。

两控制中有/无置信度的实际逐轮候选提议完全相同：True。payment的局部改善不需要概率/熵信息也能由该消融取得，尚无置信度的独立增量证据。Risk仍未超过该例的Guided-Edit，不能把修复内部拼接退化说成超过所有合理基线。

解释边界：同状态接受增加不等于整条路径收益，必须结合表中总轮与全成本时延；无置信度消融若取得相同行为，也不能把收益独归于概率/熵信息。两个训练案例只回答机制是否能工作，不能说明新项目可泛化。

冻结执行`00bc845542cd6ab54778131a472a9e7945e8fd4e`，package SHA `098e4ec770108a3b4da457f16f46d86ac58f9dd0e6c46d146accb1ab4f2ff44c`；原结果及12请求主证据的已记录文件SHA未变。

```bash
python -m scripts.write_prefix_risk_controls_report
```
