# 真实发布说明影响初筛：小成本检验

2026-10-18。运行前协议：[RELEASE_TRIAGE_PROTOCOL](RELEASE_TRIAGE_PROTOCOL.md)。

6个公开项目、各3个指定版本，共18次Target AR推理。数据是真实发布说明，任务及输出合同由实验构造；没有自然用户需求/服务日志，不声称独立泛化确认。
标签在推理前由开发agent按来源逐例标注并冻结；没有独立领域专家复核，不使用Target自评。

| 质量检查 | 结果 | 门槛 |
|---|---:|---:|
| 严格输出合同及版本全部合格 | 7/18 | 18/18 |
| 五类判断与证据全部正确的请求 | 3/18 | 至少17/18 |
| 严格类别得分（整份合同失败时五类均计错） | 30/90 | 至少86/90 |
| 关键正类未通过严格评分（含整体合同失败） | 7 | 0 |
| 截断或未结束输出 | 0 | 0 |

| 项目 | 正确请求 | 有历史请求的独有覆盖 |
|---|---:|---:|
| psf/requests | 1/3 | 74.1% |
| encode/httpx | 1/3 | 70.7% |
| pallets/click | 0/3 | 34.6% |
| pallets/flask | 0/3 | 76.5% |
| Kludex/starlette | 1/3 | 75.7% |
| Textualize/rich | 0/3 | 79.5% |

独有覆盖是Global可复用位置并集减Context并集，仅统计12个有历史请求：
1633/2597 token（62.9%）；门槛10%。
达到项目级5%独有覆盖的项目为6/6，门槛至少4/6。
此为枚举来源位置、利用当前答案选择最长正确接续的离线oracle上限；忽略真实候选选择、路由和全部运行成本。不能称为实测接受率、可实现净收益或加速比。
所有18例均保留，包括错误答案；正确性是分类初筛合同，不是完整迁移计划、功能测试或安全结论。

失败详情：
- psf--requests / turn=0：错误类别 ['python_runtime', 'dependency', 'api_migration', 'security_fix', 'other_behavior']；schema=False，identity=False，error=Invalid area/status/evidence contract。
- psf--requests / turn=1：错误类别 ['security_fix']；schema=True，identity=True，error=None。
- encode--httpx / turn=0：错误类别 ['python_runtime', 'dependency', 'api_migration', 'security_fix', 'other_behavior']；schema=False，identity=False，error=Invalid area/status/evidence contract。
- encode--httpx / turn=2：错误类别 ['python_runtime', 'dependency', 'api_migration', 'security_fix', 'other_behavior']；schema=False，identity=False，error=Invalid area/status/evidence contract。
- pallets--click / turn=0：错误类别 ['python_runtime', 'dependency', 'api_migration', 'security_fix', 'other_behavior']；schema=False，identity=False，error=Invalid area/status/evidence contract。
- pallets--click / turn=1：错误类别 ['python_runtime', 'dependency', 'api_migration', 'security_fix', 'other_behavior']；schema=False，identity=False，error=Invalid area/status/evidence contract。
- pallets--click / turn=2：错误类别 ['python_runtime', 'dependency', 'api_migration', 'security_fix', 'other_behavior']；schema=False，identity=False，error=Invalid area/status/evidence contract。
- pallets--flask / turn=0：错误类别 ['security_fix', 'other_behavior']；schema=True，identity=True，error=None。
- pallets--flask / turn=1：错误类别 ['python_runtime', 'dependency', 'api_migration', 'security_fix', 'other_behavior']；schema=False，identity=False，error=Invalid area/status/evidence contract。
- pallets--flask / turn=2：错误类别 ['other_behavior']；schema=True，identity=True，error=None。
- Kludex--starlette / turn=1：错误类别 ['python_runtime', 'dependency', 'api_migration', 'security_fix', 'other_behavior']；schema=False，identity=False，error=Invalid area/status/evidence contract。
- Kludex--starlette / turn=2：错误类别 ['api_migration']；schema=True，identity=True，error=None。
- Textualize--rich / turn=0：错误类别 ['python_runtime', 'dependency', 'api_migration', 'security_fix', 'other_behavior']；schema=False，identity=False，error=Invalid area/status/evidence contract。
- Textualize--rich / turn=1：错误类别 ['python_runtime', 'dependency', 'api_migration', 'security_fix', 'other_behavior']；schema=False，identity=False，error=Invalid area/status/evidence contract。
- Textualize--rich / turn=2：错误类别 ['python_runtime', 'dependency', 'api_migration', 'security_fix', 'other_behavior']；schema=False，identity=False，error=Invalid area/status/evidence contract。

质量门槛：False；机会门槛：True。决定：`stop_candidate`。
未通过预定前置门槛，未启动最多48次加速方法计时；未扫提示、修改标签、放宽验证或剔除失败请求。这个候选检查按协议结束。

Target请求累计墙钟（含本次快照/写回等）：217.819s；模型加载13.761s，单列。
AR请求时间不是加速对照结论，也不包含模型安装、全部权重审计和离线评分。
FP32/eager、greedy、非思考、Qwen3-8B、Transformers4.57.3、A6000 GPU0、4线程。
同项目过去回答保留错误，并按完成顺序写回；每轮Target只见本次说明和共同定义，候选历史不补充缺失的当前事实。

推理源码SHA-256：`83f7e6f6b8f43965b9e916c8edd91e36cc484f85e933f25b9224763bef7cc302`。
冻结数据manifest SHA-256：`e5e0b6ac9ddd1a33da4b3dadb5d11e1f3211aab93b3b34006f566c9b9c3415e7`。
原始参考JSONL SHA-256：`f151438d2d31d2fd84aadfcc0773c93edbfaacb58dd85c54204c7ddb0b93d7ed`。

服务器首次评分：`PYTHONPATH=. HF_HUB_OFFLINE=1 .venv-dflash/bin/python -m scripts.write_release_triage_report`。
无模型离线复算：`python -m scripts.write_release_triage_report --report-only`。

事后诊断（不更改上述门槛/得分/停止决定）：
实际18/18均可解析JSON且版本/类别顺序正确；只看status为88/90类、16/18请求正确。
11请求超过1–2条证据上限，9请求引用不支持该类的行；两者可重叠。关键正类实际status漏判0，关键类误报1。
严格合同失败会令整份评分归零；不能把30/90称为纯分类准确率，或把严格评分的7项关键失败全部称为真实漏判。
另用两个不含来源事实/标签/当前答案的静态schema示例（共323 token）作探索性oracle对照：历史在prompt+静态示例之外独有覆盖为96/2597（3.7%）。
静态对照在看过输出后增加，格式选择有事后性，资源预算未逐请求严格配平，没有在线计时；它解释结构重复，不证明静态方案的实际速度或新门槛通过。详见 [诊断](RELEASE_TRIAGE_DIAGNOSIS.md)。
