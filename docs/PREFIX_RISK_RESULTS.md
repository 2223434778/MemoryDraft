# 前缀风险辅助拼接：独立项目检查

2026-10-19。仅用旧56状态/9项目训练前缀生存，固定0.5边界风险规则；新6项目/12请求与训练项目隔离。四项目为短修改后恢复复制，另两个为复制/全文替换控制。全部为作者受控输入，不是生产流量；当前prompt含源文，不主张长期历史不可替代。

精确suffix优先。风险策略只在原Edit候选边界不安全、另有安全边界时改选；保持原长度上限32、候选池、预算与严格Target验证，不预测长尾、不求解OT。无置信度消融采用相同训练和规则。

12个新AR参考构造成本52.955987s，单列而非在线计时。

| 请求 | 文本合同 | 结束原因 |
|---|---|---|
| archive-pruner:0 | True | eos |
| archive-pruner:1 | True | eos |
| vector-ingest:0 | False | eos |
| vector-ingest:1 | True | eos |
| blob-recovery:0 | True | eos |
| blob-recovery:1 | True | eos |
| quota-ledger:0 | True | eos |
| quota-ledger:1 | True | eos |
| scheduler-copy:0 | True | eos |
| scheduler-copy:1 | True | eos |
| status-rewrite:0 | True | eos |
| status-rewrite:1 | True | eos |

Target未通过预先固定的完整输出合同，按协议停止；未运行60次机制或120次计时。这不能判断风险策略有无性能优势。本轮不调整提示、不挑成功项目继续，保留实现和失败参考。

失败项vector-ingest:0要求只将batch_size从160改为224并返回完整文件，Target只输出`batch_size = 224`，遗漏完整文件的其余配置内容。这是实质性完整性失败；其EOS结束，非输出预算耗尽。其余11项合同通过（忽略末尾LF数量），但不能事后仅选它们作为本轮完整验证。

固定Target与严格接受规则下，加速结果必须与AR相同，候选风险策略无法纠正这次遗漏。应保留原实验的停止结论；未来若重新验证，需要事先另立公共输出合同和参考基线，不能将本轮改提示后重新运行叫独立确认。

执行提交`ad231f712858868c418ce33494dd967d5de8230c`；package SHA `ba8782a55669b9bf4647d99b729c1f5023bd7ea78a87165d0688f5c20986a943`。原始旧证据递归SHA未变，独立infra版本和默认suffix保留。

交付时的真实tiny Target组合测试另发现均匀分布FP32熵可略超1，已增加[0,1]数值裁剪与回归；这是风险特征的边界修复，未改Target验证。本次冻结GPU检查停在AR阶段，未执行新风险策略的机制/性能路径；不能将组合单测当作其GPU性能验证。

来源核验使用冻结Git blob。仅两份早期research-audit JSON在本地Windows检出为CRLF、服务器Git原始导出为LF，报告只读兼容此已知差异，不修改原文件；其余证据字节SHA精确检查。

```bash
python -m scripts.write_prefix_risk_report
```
