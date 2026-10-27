# 树流聊天界面与真实解码过程

网页品牌为「树流」，采用原创树与水流标识。登录页、侧栏、输入框、选择菜单和详细面板加入半透明玻璃、背景模糊、柔和色彩和细高光边缘；不支持背景模糊时保留更不透明的可读底色。默认正文及手机正文均为 17px。模型／模式菜单打开时淡入并上移，支持方向键、Home／End、Escape 和真实能力禁用。菜单放在页面顶层，避免玻璃容器改变固定定位。账号、会话隔离、流式回答、停止生成及阿里云 SQLite 历史继续使用原链路。底层包名、数据库和冻结研究版本保留原名称，网页改名不要求重新注册或迁移聊天。

「展示推理过程」是输入框左下方26px圆形勾选框，只控制正文内的简短草稿与接受颜色；勾选时有短暂回弹、勾画和光圈。顶部「展示详细推理过程」只开关右侧轮次、完整候选和树图，两者互不勾选、互不展开／关闭，生成过程中也可分别切换。为支持生成中临时打开任意一个视图，网页对具备过程能力的投机请求持续收集本轮真实事件；两个视图关闭时仍有事件解码和传输成本。原始API的 `visualize:false` 继续关闭事件观察。详细面板不会因提交问题或勾选简览而自动展开。

## 四种模式

| 界面 | 实际请求 | 过程面板 |
| --- | --- | --- |
| 关闭 · 普通生成 | AR | 逐词生成，没有小模型草稿 |
| 投机解码 | DFlash | 实际小模型草稿及严格接受前缀 |
| ASD | ASD | 实际草稿、接受前缀、近似放行标记和预算 |
| ASD + 成本树 | BudgetTree-ASD | 实际候选树、若干叶路径和最终提交路径 |

黄色表示已经起草、等待验证；绿色表示实际采用；红色表示拒绝或丢弃；蓝色表示 Target 提供的纠正或补充。ASD 允许与严格 Target 贪心结果不同的 token，这些近似放行项以虚线标识。树上未采用的分支不一定逐 token 都错误，面板明确说明这一点。

一轮流程为：小模型起草 → 构造候选树（树模式）→ `draft` 事件 → Target 验证 → 原规则确定提交路径并更新 KV → `verified` 事件 → 发布正文。树图使用真实的 token ID、父节点和深度，最终路径来自验证器，没有为动画额外编造候选或改变 Target 输入。

首个 Target token 是已确认的锚点，不计入草稿接受长度。EOS、输出长度上限后的内容不会计入实际提交。单 token 无法独立解码的字节片段显示为 `#token_id`；正文仍解码完整前缀，避免中文碎片乱码。

### 为什么草稿可能重复

当前 DFlash 的 block=16 包含一个已确认锚点和最多15个候选位置。它在一次前向中并行预测这些位置，后面的位置没有看到前面位置刚刚选中的实际 token，因此相邻位置可能同时猜出同一个词。黄色草稿不是已经逐词写好的小模型回答；详细面板的 token 卡片间距也不代表原文包含空格。[DFlash 官方说明](https://z-lab.ai/projects/dflash/)介绍了这种并行起草；[D-Loop 论文](https://arxiv.org/abs/2610.06011)将相邻位置冗余复制称为 repetition trap，并分析其与接受长度的关系。

2026-10-27 的一次合成中文检查在原始候选 ID 中复现“帮助帮助帮助帮助”和“以下以下”：58轮中57轮有相邻相同 ID，44轮涉及文字 token；58轮正文候选都与原始 ID 的上下文解码一致。共361对相邻重复中，没有一对的两个位置都被接受；128个最终 token 与同输入 AR 完全一致。严格 DFlash 只提交与 Target 一致的最长前缀，首个不匹配位置及后缀被丢弃，由 Target 补充。该保证不应套用于允许近似接受的 ASD 模式。

这是一个输入、128 token 上限的定位检查，不是整体接受率或时延评测，也不能排除其他输入上的独立问题。它确认本次复现的重复来自真实模型候选，而非页面重复拼接；重复候选浪费位置，可能降低接受长度和加速收益。结果和检查脚本保存在 [results/chat-draft-repetition](../results/chat-draft-repetition/)。未给候选去重、添加重复惩罚或改动验证规则。

## 事件、动画与历史

`/v1/generate` 新增可选 `visualize: true`，只适用于流式请求。默认关闭；关闭时不执行事件所需的 GPU→CPU 候选拷贝。`/healthz.speculation_events` 表示服务是否提供事件。

```json
{"version":1,"phase":"draft","round":1,"proposal_ids":[12,34],"proposal":[{"id":12,"text":"例"},{"id":34,"text":"子"}],"tree":null}
```

随后同轮 `verified` 包含实际 `accepted`、`committed_ids`、`correction_ids`、`selected_path`、`relaxed` 和 ASD 预算。树模式的 draft 还包含 `tree.token_ids/parents/depths/tokens`；`selected_path` 以已确认根节点 0 开始。事件不传 logits、隐藏状态或模型权重。

事件和正文共用原有有界队列（默认 32），保持顺序、背压和协作取消。可视化有实际解码、序列化、传输和绘图成本，不应把它视为零开销，也不能用这里的在线耗时替代算法性能评测。前端收到事件即更新最新轮次，不保留定时回放队列。勾选展示时，简短黄色草稿直接接在正在生成的段落或代码末尾；验证后在真实正文对应位置标绿，Target 补充标蓝，未采用片段短暂标红并划线。树模式的正文只预览一条真实叶路径；完整树、其他候选和最终路径留在右侧面板。前端只保留最近 64 轮，落库保留最近 16 轮，每轮候选最多 128 个，单事件最多 64KiB。历史中可点击“查看推理过程”，用左右箭头切换保留的轮次。

动画只平滑显隐，不推迟验证状态或正文。验证阶段短于显示器一帧时，浏览器可能合并绘制；右侧仍可查看已保留轮次。右侧面板可以暂停跟随以浏览旧轮次，正文始终跟随最新事件。完成、停止或出错时移除正文临时预览与颜色，保存答案和代码复制均只使用模型已确认的原文。过程属于该条 assistant 消息的 metadata，和私有聊天一起保存在阿里云，不写回检索记忆，也不放入后续模型 prompt。停止时未验证的草稿仅在详细面板保留未完成状态。

正文定位由可选 `inline` 展示字段提供：服务端对实际生成 ID 前缀做上下文解码，发送有界候选片段及 UTF-16 起止位置，以适配浏览器字符串和 emoji。中文字符如果由 Target 字节补全，就标为 Target 补充，避免错误归功于尚不完整的 Draft 字节。Markdown 格式符与折叠空行也保留源位置计算。该层不调用模型、不重编码生成 token，也不修改接受规则；API关闭观察时不创建此展示解码器，网页的两个显示开关单独控制DOM。

2026-10-27 正文整合验收：相关 Python 60 项及 JS 状态 5 项通过；真实 DFlash 代码回答和成本树中文回答的保存文本均与 complete 事件原文一致，390px 手机无横向溢出；关闭展示、中途停止、菜单键盘及减少动画检查通过。四模式 GPU 开关配对最终严格通过。首轮曾出现计数／ASD metadata 严格比较失败，未保留到具体差异；原因尚未确定，不能把后续通过当作已定位该问题。见 [本轮验收](../results/chat-inline-ui/acceptance.json) 和 [GPU 配对结果](../results/chat-inline-ui/gpu.json)。

## 图标、动画与最近对话删除

界面使用本项目绘制的树与水流标识及线条 SVG，favicon与对话头像统一，不使用 Gemini 星形标识或第三方图标素材。欢迎语／建议分步淡入，侧栏平滑展开，按钮悬停与输入框聚焦过渡，正文新增片段淡入，生成时显示呼吸状态。系统启用 `prefers-reduced-motion` 时关闭动画，`prefers-reduced-transparency` 时移除模糊并提高背景不透明度。

最近对话每行提供删除按钮：电脑悬停或键盘聚焦时出现，手机直接显示。先确认再调用原有按用户鉴权的 DELETE 接口；取消不会删除。删除其他对话保留当前回答，删除当前对话返回欢迎页。生成中禁止删除，后端继续拒绝越权或正在生成的会话删除。

本轮回归命令增加 `node --test tests/chat_ui.test.cjs`，覆盖即时状态、突发事件不积压、保留轮次上限，以及历史浏览与实时视图独立。真实浏览器使用本地独立临时 SQLite 连接原 A6000，删除验收未使用生产聊天数据库。结果见 [实时界面验收](../results/chat-live-ui/acceptance.json)。

## 基础模型

模型由服务端目录配置，浏览器只能提交目录中的 ID，不能指定任意推理 URL。`GET /api/models` 返回模型名称、在线状态、可用方法和可视化能力；每次显式选择请求还会验证所选能力。未实现的模式禁用。

仅提供 AR 的模型在名称后标注“仅普通生成”，其禁用方法标注“尚未接入”。当前 Qwen3.8-27B 的 ASD 适配仍是未完成的独立工作；体验已有 DFlash／ASD／成本树应选择 Qwen3-8B。

原 Qwen3-8B + 配套 DFlash + 原成本表继续驻留 GPU0，四种模式齐全。较大模型选择 Qwen3.8-27B 的 Q8_0 量化：尽量保留精度，同时在另一张 48GB A6000 上独立运行，避免 SYS 跨卡互连及原服务争抢显存。这个选择是质量、显存和现有服务共存的折中，没有声称它在所有任务上都是最强模型。

Qwen3.8 使用不同的混合架构，现有 DynamicCache 树验证路径和 Qwen3-8B Draft 不能直接套用。因此较大模型的薄适配只提供 AR；不能以更换下拉框标签的方式宣称它已有 DFlash、ASD 或成本树。若以后适配，需独立验证 Draft、状态提交和原精度规则，并重新建立对应成本表。

来源：[Qwen 模型说明](https://huggingface.co/Qwen/Qwen3.8-27B)、[官方 GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF)、[llama.cpp 固定版本](https://github.com/ggml-org/llama.cpp/tree/b11907)、[原生服务接口](https://github.com/ggml-org/llama.cpp/blob/b11907/tools/server/README.md)。模型 Apache-2.0；llama.cpp MIT。精确权重 revision/SHA 和运行时 commit 保存在新模型准备清单。

原生 b11907 在 UTF-8 片段未凑成字符时暂缓发文本，同时可能遗漏对应流式 token ID。小补丁 `deploy/chat/llama-stream-token-ids.patch` 为这些步骤增加空文本的真实 token 事件，保持模型计算和采样原样；适配器以完整 ID 前缀解码。补丁 SHA 记录在模型清单，不能把重编码后的文本当成原始生成 ID。EOS 的显示过滤也使用服务器提供的特殊 token ID。

## 配置和验证

在原网关环境基础上增加可选目录（systemd EnvironmentFile 中 JSON 外层加单引号）：

```ini
MEMORYDRAFT_CHAT_MODEL_ID=qwen3-8b
MEMORYDRAFT_CHAT_MODEL_LABEL=Qwen3-8B
MEMORYDRAFT_CHAT_MODELS='[{"id":"qwen3.8-27b-q8","label":"Qwen3.8-27B · Q8","url":"http://YOUR_A6000_TAILSCALE_IP:8011"}]'
```

内部服务仍使用原 Bearer 鉴权；llama.cpp 本身只监听 loopback，外部通过适配器访问。不得把密钥或私有环境文件提交 Git。新模型适配器复用现有单执行线程、队列、取消、token 计数及 SSE 接口；普通生成的 token ID 来自原生 `/completion`，不以字符或重新编码内容伪造生成 token。

准备大模型：`python -m scripts.prepare_chat_model --root /path/to/chat-service/large-model --proxy http://127.0.0.1:YOUR_PROXY_PORT --patch deploy/chat/llama-stream-token-ids.patch`。脚本固定权重和上游版本，校验 28,595,763,648 字节文件及 SHA，使用显式 CUDA 12.1 工具链编译，避免机器上旧 CUDA 库混入链接。只在该独立目录保存下载和构建；不修改旧 DFlash venv。原生模型使用 GPU1、32,768 context、单 slot、flash attention、关闭 thinking；适配器从原生 `/props` 发现实际窗口。网页输出上限提高到16,384，输入历史上限仍为8,192，详见[长度设置](CHAT_DEPLOYMENT.md)。适配器用 `python -m uvicorn memorydraft.serving.llama:app --factory --host YOUR_A6000_TAILSCALE_IP --port 8011 --workers 1`，从环境读取内部鉴权。

这台服务器首次连续请求暴露了 CUDA graph 路径的 `stack smashing`。关闭上下文检查点／缓存未解决；GDB 回溯定位到 `ggml_cuda_graph_launch_chunk`，原生后端单独设置 `GGML_CUDA_DISABLE_GRAPHS=1`。这个运行设置只作用于新模型进程，原 DFlash 执行配置及成本表保持。构建时也关闭上游内置 Web UI，避免下载未固定的前端资源；聊天界面由本项目提供。

两个可选用户服务模板为 `deploy/chat/large-model-user.service.example` 和 `large-adapter-user.service.example`，替换路径／IP 后分别安装为 `memorydraft-chat-large-model.service` 和 `memorydraft-chat-large-adapter.service`。适配器复用内部鉴权环境文件；模型仅绑定 127.0.0.1。确认 `/healthz` 就绪及实际生成通过后，再把模型加入网关目录。

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_speculation.py tests/test_llama_serving.py tests/test_chat.py tests/test_serving.py
node --check memorydraft/chat/static/app.js
node --check memorydraft/chat/static/process.js
```

真实验收使用 `scripts/validate_chat_visualization.py`，以原私有环境文件提供鉴权。`--url` 执行 GPU 观察器开关配对；不传 `--url` 时执行网关四模式和过程保存检查，需要 `--private` 保存临时合成账号的浏览器会话。测试账号清理只能依据脚本产生的精确 ID，不能按用户名模糊删除。

2026-10-27：完整 Python 回归 1184 passed / 7 skipped。真实 GPU 四模式观察开关配对保持 token、计数和 ASD 预算一致；跨机网关四模式生成及过程落库通过。见 [GPU 结果](../results/chat-visualization/gpu.json)、[网关结果](../results/chat-visualization/gateway.json)。这些是可视化正确性验收，不是新的速度基准。

Qwen3.8-27B Q8 已在 GPU1 上线，服务实占约 26.7GiB。中文和包含 emoji／罕见字的两组真实请求中，适配器流式 token ID 与原生非流式完整 token ID 一致，完整前缀解码也一致；网页完成实际模型切换、模式禁用和回答保存检查。见 [模型固定版本与 SHA](../results/chat-visualization/model-prepared.json)、[适配器检查](../results/chat-visualization/large-model.json)。这是小规模功能检查，未进行新模型的长时间压力或跨模型质量评测。
