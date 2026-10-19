# DFlash前缀引导的OT-R机制验证

2026-10-19。在新的Target输出出现前固定实现、任务和以下规则。默认运行仍为suffix；完整infra面试版不覆盖。

## 机制与来源

- 精确suffix达到6token时仍直接复用，跳过Draft。
- 精确匹配失败时先运行原有DFlash。取其前4个候选token，在既有合法suffix-array资源内查找一致的连续前缀，最多检查64个后缀项。端点需有至少8个历史上下文token；候选必须比原DFlash的候选块更长，最多32token。
- 该前缀确定接续端点，不再用旧窗口的固定±2偏移猜插入后的端点。对“已确认输出+4个未确认hint”与端点后的历史窗口评分，最多8候选，OT/Edit都保留原0.65阈值。最多4次引导查询，无候选不调用OT。新增倒排索引不构造。
- Guided-Edit和Guided-OT共用前缀、候选池、预算、块长度及Target验证，仅排序器不同。hint未被Target确认，可能错误；选出的候选仍逐token严格验证，错误hint不被当作已确认输出。
- 被替换的DFlash前向已经支付，真实draft_forward_rounds单列，不把R路线计数冒充跳过Draft的轮数。在线GPU→CPU hint同步、查找、评分、额外验证和KV操作全部计时。
- 借鉴LogitSpec“预测token引导检索”和SAM-Decoding“检索与模型起草组合”的机制；此处使用DFlash候选，不是复现其原算法，也不宣称新颖性。未复制新增上游代码。
  - LogitSpec（核对v3）：https://arxiv.org/abs/2507.01449v3
  - SAM-Decoding：https://aclanthology.org/2025.acl-long.595/

## 开发与新请求验证

1. 开发阶段仅原诊断八请求，Suffix/原OT/Guided-Edit/Guided-OT各一次，共32次，严格核对原AR；开发耗时不当作独立确认。检查旧长接续正例是否仍被新机制使用。
2. 新验证包含8个新编写受控项目×两独立修订请求，共16请求。edited-continuation为TOML/SQL/Python/JSON四项目；control为短文本、空历史、无关历史、SQL尾部编辑四项目。这些是作者编写任务，不是外部生产流量。
3. 两请求都编辑给定v1；合法历史在order2，当前order10/11；不写回任何当前答案。所有方法共享相同当前源文，不据此证明历史不可替代。
4. 开发通过后固定相同package和runner，在新请求上先生成16个AR，核对prompt、快照、结束原因及只修改指定span的契约。任何契约不符/长度截断则停止正式计时，保存失败，不换题或按答案改参数。
5. 正式DFlash、Suffix-R、原OT-R、Guided-Edit、Guided-OT五组，两重复，共160次；请求/重复内随机交错。FP32/eager、Qwen3-8B、原DFlash、4线程、batch1、最多512新token。模型加载、POT导入、预热和AR参考单列，所有在线增量成本计入。
6. 每次输出token、prompt及snapshot严格匹配AR，首次差异立即停止并保存。所有旧结果运行前后SHA不变。

## 预定判断

- 新edited-continuation组若Guided-OT比Suffix降低>=3%延迟、比Guided-Edit降低>=1%，且分别至少3/4项目更快、8/8契约正确/无截断，则认为有继续研究OT的局部信号。该门槛为工程决定，不是统计显著性。
- 另外单独报告Guided-Edit、Guided-OT相对原OT的变化和control退化。若只有Guided-Edit有收益，结论应归于引导/路由机制，不归于OT。
- 不在本轮后调参并重测同批宣称确认；未通过时保留负结果。四项目和两重复不足以证明泛化或稳定生产收益。
