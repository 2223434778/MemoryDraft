# 拼接前缀拒绝的冷却修复：事后小检查

2026-10-19。首轮所有原始结果已提交`bc6b39e`，保留不覆盖。此前16请求全部观察过，本轮再选其中5个故障/回归请求，属于开发检查，不是独立确认。

- 唯一推理行为变化：若Splice在保留Draft前缀内部拒绝，尚未到历史尾部，不触发历史检索冷却；首次历史token或之后拒绝仍保持原冷却。普通R、原Guided、Target验证和所有检索参数保持原规则。
- 固定5请求：payment-runbook:1、shipment-contract:1（两负例），inventory-query:0/1（两个SQL局部正例），confirm-schema-0:0（旧TailOT正例）。全部六方法都运行，先30次插桩检查，再同一package/runner下60次交错未插桩计时。
- 修复目标是两负例不再因前缀拒绝错过下一轮32token机会，SQL/旧JSON正例严格输出不回退。保留原机制门槛（至少两个SQL请求）作为计时入口；若不通过则停止。不是为新研究主张设置的统计门槛。
- 所有已发布旧证据包括首轮`results/ot-splice`递归SHA保护。独立服务器目录`ot-r-splice-cooldown`、独立结果目录`results/ot-splice-cooldown`，分别保留执行源码清单，禁止拿修复后源码冒充首轮执行版本。
- 沿用原FP32/eager、4线程、batch1、相同prompt/合法历史/AR参考、全部在线成本。首个严格差异停止。不得按这五个答案调其他参数；无新holdout、无新场景扩展。

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_splice_probe --cooldown-check --stage mechanism --config configs/runtime-dflash.json --output artifacts/mechanism-v1
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_splice_probe --cooldown-check --stage timing --config configs/runtime-dflash.json --mechanism artifacts/mechanism-v1 --output artifacts/timing-v1
```
