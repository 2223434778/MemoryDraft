# 开放任务的题干约束补充检查

2026-10-24。运行中的主GPU实验仍冻结于6d89480；本文件是另加的离线质量检查，不更改生成、主评分、样本或预算。规则在新数学生成开始后、任何新MT-Bench/Alpaca回答生成或查看前声明，不能称为整轮GPU实验开始前的原始预登记。

配置`configs/asd-extension-contracts.json`于12:36:25 UTC写入，SHA256 `7684f4063874ad1f86fbbe651365927277bc26d056dfde4f717a8b001e21993e`。12:38:06 UTC远端核对两worker仅完成GSM38/39题、当前仍为GSM，尚无MT rollout文件；快照保存artifacts/asd-extension-contracts-before-open.json。规则只绑定题干字段的SHA，不使用Alpaca原始参考output；所有回答生成后再按规则评价。

34项性质覆盖26个源任务、32个请求轮次，四方法共136个性质单元。其中9项可自动检查（36单元），25项由匿名的agent辅助内容检查（100单元）。这不是人工专家评审、独立训练的第三个裁判，也不替代完整任务准确率或官方GPT-4/AlpacaEval。

- 自动：三组字符串列表JSON、三整数1..5评级数组、JSON内的原始年份、JSON单行输出、消除指定性别代词、250/500词上限、给定字面标识符移除、Header/Body/Footer标签顺序。
- 内容：明确数学/概率、停车推理、命名实体与首字母、LCS边界错误、五步工作流、句数/字数、列表数量、单位/时间转换、翻译及三角形代码。允许合理等价形式，具体接受条件在JSON；未提供代码的LCS回答不因缺代码而失败。列表数量检查不等于医学效力或安全建议正确性认证。
- JSON检查允许解释或代码围栏中的完整JSON载荷；要求单行的题单独计算全响应的行数。嵌套数组不伪装为顶层评级数组；不从损坏外层JSON中挑内层片段通过检查。
- 词数固定为正则`\b\w+(?:['’-]\w+)*\b`的匹配数，保留空白token计数辅助判断；不是跨语言通用分词准确率。字面脱敏不等于完整隐私保证。格式性质通过不证明语义完整正确。
- 内容检查先导出隐藏方法名的回答包，作答后揭示映射。每项保留原回答SHA、判断依据和选择出的短句（需要时）；不把检查器看到的方法文本当作新的训练数据或检索历史。

原题MT111：两边从原点出发向量点积0、叉积绝对值6，面积3；对边平方20，外接圆半径平方5，面积5π。MT105的约束唯一给出Cheryl/Bert/Enid/Alice/David及red/blue/green/yellow/purple，秘书Alice。MT113的并集81%、neither19%，给定喜欢green时neither为0；对歧义作清楚区分的回答可接受。这些期望来自题目约束推导，不读取模型回答或原始参考解答。

在看回答前另标记6个来源解释限制：仅有图片链接、未提供文章/书籍正文、流行度缺少时间/口径、标量函数与数组语义不明确、第二轮家谱角色可能同人。全部保留在主计时和奖励评分中，不按这些标签筛除负结果；本检查不为它们伪造客观正确率。

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_asd_extension_contracts.py --basetemp artifacts/pytest-extension-contracts --junitxml=artifacts/paper-extension-contract-tests.xml
```

首次准备脚本以文件路径运行，无法解析本项目模块，配置未创建，导致附加测试10通过/1缺文件失败；改用`python -m artifacts.prepare_extension_contract_plan`创建配置并校正规则数量，专项12 passed/4.10s。两个XML均保留，不改变主GPU代码。最终报告区分自动/内容检查、未评项与来源限制，主质量分数保持原样。
