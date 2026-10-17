# 结构化产物连续修订：小规模独立确认

2026-10-18。开发场景表明，重复字段的配置修订中简单Global常优于游标；本阶段选择简单全局复用策略，不修改R超参数或增加学习模型。启动前必须先完成整个开发批次。本协议在生成本阶段任何过去/当前Target答案前保存。

- 这是独立项目的小规模确认，不是自然用户数据或全面泛化测试。四个新项目，workers/routes/steps/checks四种JSON schema，分别5/6/7/8项，初始参数和修改位置预先由 `confirmation_projects()` 固定；与开发项目ID完全分离。每项目两轮顺序修改，8个当前请求，每方法3次交错重复。
- 当前请求给出完整项目事实和格式约束，但不粘贴过去完整回答；过去回答由同一Target按先前要求生成、冻结。每轮所有方法和重复完成后，才提交同一当前AR回答成为下一轮合法过去版本。
- 四方法：官方DFlash；Adapted（同一适配关闭R）；Global（完整合法历史+当前prompt的有界全局匹配）；Context（相同Global策略、只从当前prompt取候选）。因此Global/Context只差过去历史，Global/Adapted只差复用机制，DFlash/Adapted测接入开销。
- 固定沿用开发阶段参数：R上限32、准入匹配6、索引3..32、扫描64、拒绝后冷却一轮、特征待消费1024。无游标、无局部重匹配、无训练。固定同一Qwen3-8B、DFlash、FP32/eager、4.57.3、A6000 GPU0。过去/当前输出预算均768，报告截断。显式greedy、同一Target输入、EOS和严格验证。
- 4项目×2请求×3重复×4方法=96次正式计时；另外8个AR参考和4个过去资产构造单列。当前参考和过去构造不得算作额外历史预知。在线含冷索引、快照、全部前向与缓存维护。
- 主比较为场景内 DFlash/Global 和 Adapted/Global；Context/Global检验过去历史增量。按项目配对bootstrap，重复不是独立样本；四项目区间仅支持有限场景结论。不会用结果重新选择长度、schema或项目。
- 首次token不一致停止并保存。优先确认实测净收益和来源归因，不要求全部基线全面获胜。若收益不存在，也保存并报告，不在本批测试继续调参数。
- 原先8项接入smoke已有成功证据且核心循环不变，此阶段不重复完整smoke。runner确认原批次已完成，保留来源SHA；新数据与多重复回放的CPU审计需先通过。

命令：`OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv-dflash/bin/python -u scripts/run_reuse_pilot.py --confirmation`。
结果 `results/reuse-confirmation/`；源码SHA在该目录provenance单独记录，不能以新源码SHA覆盖开发实际运行来源。
