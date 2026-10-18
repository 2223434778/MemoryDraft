# 发布说明影响初筛数据

18份真实changelog章节（6项目×3指定版本），并非自然用户对话或服务日志。版本选择在任何模型输出前固定，非全部连续版本；每份来源tag解析为commit，并核验tag路径与commit路径原始文件相同。保留完整当版章节，模型只见加行号的本版章节和共同分类定义。原始整本changelog仅放忽略缓存，发布文件保存源文件和片段的SHA-256及URL。

任务是升级影响初筛，不是完整迁移计划。五类定义、质量/机会门槛和退出规则见 [协议](../../../docs/RELEASE_TRIAGE_PROTOCOL.md)。数据与标签来自公开说明和开发agent的推理前来源审阅；没有独立专家复核。`labels.json` 不是模型自评，不进入Target prompt。允许的证据行指支持变更的整个bullet内的非空行；同类无需枚举所有变更，给1–2条充分支持即可。阴性表示本版未提及该定义下的变更，不等于安全升级。

许可：Requests Apache-2.0；HTTPX、Click、Flask、Starlette BSD-3-Clause；Rich MIT。已阅读固定最终所选commit的许可并保留全文及来源；不引入其推理代码。Starlette是之前核验迁移后的同一仓库Kludex/starlette。下游再分发应保留各项目版权和许可。

`manifest.json`绑定项目/章节、标签和六份许可；README不参与数据标签SHA。数据获取由显式只读GitHub连接完成，采集/推理脚本不自动访问网络。逻辑回放顺序不冒称发布日真实服务；只将本轮前已完成的同项目回答纳入历史，旧错误保留，当前材料和标签不被写成过去历史。

推理前复核收窄了三组过宽的证据行范围，剔除纯类型/内部重构行；最初准备manifest `6a4ab773...` 与未运行标签草稿保存在忽略缓存，没有GPU结果或据输出修标。最终唯一运行数据manifest为 `e5e0b6ac9ddd1a33da4b3dadb5d11e1f3211aab93b3b34006f566c9b9c3415e7`。

执行完成：18次Target正式生成、0截断，冻结严格质量门槛未过，加速方法计时未启动。事后诊断区分status判断与证据合同，并增加静态schema候选的离线oracle对照；原数据/标签/门槛未变。见 [结果](../../../docs/RELEASE_TRIAGE_RESULTS.md) 和 [诊断](../../../docs/RELEASE_TRIAGE_DIAGNOSIS.md)。
