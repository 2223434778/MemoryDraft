# ASD候选统一比较（GPU前冻结）

用户授权一次完成多阶段，以六类任务的适用场景和真实收益判断研究价值，不要求所有场景胜出或补训练seed。本轮不改既有Target/默认runtime/面试版，不下载权重、不训练或调参。

## 数据与候选

固定asd-suite reserve96，沿已冻结文件顺序每类前4为screen24，其余12为confirm72。阶段间不重叠；先扫描已有结果token载荷确认未生成。生成前核数据manifest、源码和参考实现评分。全部请求共享相同原输入和空历史，不注入当前测试答案。

原Draft、ASD-prefix LoRA、ordinary-position LoRA分别配DFlash/ASD/Run/Utility，共12组，另原Draft Confidence-Utility(alpha .5)及严格tree32(TreeConfig(4,4))，共14组。旧适配器SHA固定在evaluation/asd_comparison.py；不得按新题换checkpoint。ASD B8/g.25/cap2，Utility价格2，Run exact-run，其他接受规则不变。

相同Qwen3-8B/DFlash、FP32/highest/TF32关闭/eager attention、RMSNorm编译、native16、greedy非思考；ASD/Run/Utility均使用已核验CPU摘要实现，Confidence使用原张量实现，成本计入。tree32为严格DFlash树，不称ASD树。

## 分阶段执行

screen24题×14组×(自然4096/EOS两重复+128/EOS一次)=1008计时，另24完整AR质量/严格一致参考。每题同卡，六类各半分到两卡；按固定seed交错组次，每请求fresh KV/预算。

screen完整后，每类选择两个候选：原Draft的Run/Utility/Confidence/tree32中一个；两适配Draft的八组合中一个。各类分别按相对基础ASD正确→错误数、额外截断数、题均自然时延比、arm名称依次升序，形成确定选择；MATH使用现有有限标量secondary_correct，保守主分同时报告。该规则不保证选择在确认集有效，也不删除质量退化候选数据。无合格正收益时仍测排序最前候选，结论可为无新增收益。

选择JSON和SHA在confirm启动前落盘；confirm72每题两个候选+DFlash/ASD四组×(自然3重复+128/EOS2重复)=1440计时，另72完整AR。共2448计时+96 AR，不把重复当题数；未入选组保留screen证据，不宣称已做72题确认。

每组重复token/轮次/账本/停止稳定，所有严格臂完整token/text/EOS与同题AR匹配，ASD预算实际<=8，输入SHA/独立snapshot相同；正确性或执行完整性失败停止并留现场。错答、长度截断、零激活、速度退化不因此中止。短EOS不强制续写，128只按双方都达到128的配对集合报告并标N。主表不删除错误请求。

## 统计与归因

同题重复时延均值，再取逐题比值的算术平均；分别相对同轮同卡基础ASD和严格DFlash。各任务独立报告自然时延、请求bootstrap2000区间、长度、调用密度、质量主/secondary、正确变化和截断，前缀计时独立列。自然更短也算端到端收益，但不全部归为每token效率。比较适配严格D与适配ASD，及适配×Run/Utility，结合长度和调用密度解释互补；小样本及多候选选择不作普遍质量非劣或首创性证明。

模型准备、编译、离线dense合并与基准切换另计；全部在线prefill/起草/判定/同步/KV/文本解码计时，不把基准换适配器当生产免费热切换。自然主成本不含HTTP、模型加载、文件写入和评分；不宣称服务SLA。

## 工程优化与资源

主矩阵固定同执行profile，不把RMSNorm或缓存通用收益归给接受策略。完成后优先用既有匹配工程证据结合新确认结果判断；若需验证新组合，以单独标记的工程子实验执行，不消耗确认答案调参数，不把异轮百分比相加。公共缓存仅登记公开模板，不能拿候选独享缓存与未缓存基线比较。

每worker最多14400秒、allocated<=46GiB、Inductor<=8图/backend300秒/cache4GiB；两卡空闲后执行，无新下载/训练。只终止本轮owned进程。新root /media/a6000/D/ZhaoBinHe/MemoryDraft/asd-comparison；旧适配器只读、前后核SHA；canonical Git blobs清单运行前后核验，不覆盖旧目录。

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_comparison_pipeline
```

输出results/raw/asd-comparison，公开结果results/asd-comparison；重建`python -m scripts.write_asd_comparison_report`。仅上传代码、协议、小型公开结果/输出和报告，不上传权重/KV/hidden/大trace。
