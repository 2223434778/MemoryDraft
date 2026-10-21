# RMSNorm冻结候选独立验证（生成前登记）

候选来自e958532bb4dc7544692011115895f90df2bd3f43完整开发集结果。RegionalRMSNorm、接受器、模型/配置、输入及评分器11个核心文件SHA固定在run_asd_rmsnorm_validation_probe.py；本轮只适配验证集读取与审计，不改候选。预检查sum按Git原始LF计算，允许Windows物理CRLF。结果出现后不调整norm、策略或本协议判据。

## 题集与隔离

使用原数据池全部validation48，六类各8，不按题目/答案/生成结果挑选。reserve96不生成；保留验证题全部错答/截断/数值差异。生成前本地扫描仓库结果JSON/JSONL模型token记录：763文件、15521载荷，validation/reserve命中0；服务器每worker在生成前再查。扫描标准request_id/id/task_id（可继承父对象）及生成token列表，排除的只有本次实时raw目录，不排除其他旧raw。扫描范围是仓库已有结果，不能由此证明外部预训练无污染。

验证ID集合SHA5b565c98b8998f962b6ef321091dd92de5a572d8fefbb62ec11fb9d92b3b3c67，reserve ID集合SHAd2eb4fd0e42674faf02ace144b35788d0ab48f0460ec309d41bbb72e55eefbfc。原manifest和六数据文件SHA核验，各split两两不交。只读ID/划分用于排期，canonical评分预检在协议/代码冻结之后、GPU生成之前进行；参考答案不进Target输入。

## 执行与成本

原Qwen3-8B/DFlash b16 revisions，FP32 highest/TF32关闭、eager attention、greedy非思考空历史、batch1；只norm编译，MLP仍eager。DFlash/ASD/Run/Utility，近似两profile均CPUASDVerifier，B8/g.25/M2/Utility价格2不变。

每题两个profile各AR一次（96），四方法各profile3次相邻配对计时（1152），首重复同时评分，共1248主要生成，另20短warmup。自然EOS统一上限4096，不强制续写。每profile后两次须精确重现首轮token/文本/轮次/账本/停止原因；违反则保留失败，不合并不稳定计时。验证集没有旧输出，不能声称重现过去答案；输入与本轮匹配eager及AR对照一致。

保留既有随机顺序random.Random("rmsnorm-suite:<id>:<repeat>")，AR为repeat=-1；六类各位置偶/奇分GPU0/1，各24题。评分在单对两次生成后、engine时钟外；同题同文本只评分一次，重复复用。在线时延=engine同步wall（编码/生成/解码）+策略构造+profile绑定/恢复。首次在线新编译全部计入，不删后称steady；AR只有质量/输出对照，单次时延不作速度统计。

图8/backend累计300秒/allocated46GiB/独立空cache4GiB保持原预算，编译线程2，关闭max-autotune/CUDA Graph。worker固定7200秒：开发完整运行约31分钟，但未知验证长尾可能更长，预留上限，不看验证输出后增预算。两卡空闲才启动，不安装包/下载模型。超时仅终止本轮创建会话，保留失败。outer包含backend，不相加；warmup wall含推理。

## 统计与质量

每题三pair先取log平均，再题等权。整体48题5000次类别内重采样（各8），种子rmsnorm-validation:overall:<method>；取排序索引125/4874。各类8题用既有paired，种子rmsnorm-validation:family:<benchmark>:<method>。不再区分开发old12/new36，不以重复当独立请求。

所有48题为主结果，相同token子集仅描述。逐题首次token分歧、文本/EOS、长度/轮数、账本和质量保留。时延比=长度比×调用密度比×摊销成本比，后者含全部在线而非纯Target kernel。数学规范主评分、MATH有限标量secondary/待复核、代码提供测试/未评估、SQuAD EM/F1和JSON严格合同保持不变。

## 显式执行profile的预登记工程判定

完成全量运行和审计后，以下条件必须全部满足才继续接入显式profile；它们不是统计质量非劣证明：

1. 240对跨profile完整token、文本和停止原因全部相同；分别比较AR/四方法，不能只保留一致题。
2. 每题每方法无主正确性、MATH secondary、SQuAD F1下降，无新增未评估或截断。原有错误/截断保留，不要求所有题答对。
3. 两profile各自严格DFlash与AR均48/48 token相同。若原eager也有数值分歧，明确区分基线问题，但该工程准入仍不通过。
4. 四方法整体编译/eager时延比的95% bootstrap上界均<1，且六类各方法点估计均<1。不要求任意预设5%收益，也不因个别题慢而删除。
5. 资源预算、冻结源码、初始隔离及每profile重复一致性审计通过；正式新编译若有，计入结果而非扣除。

未通过则保留研究工具与完整负结果，不能在validation调参、删题、放宽接受规则或重命名为开发集。通过也只支持当前条件下的显式数值执行选项；默认eager保持，reserve保持未生成。后续工程接入另做首次调用计时、配置拒绝、元数据、stream/cancel恢复及检索路径检查，不把研究runner结果直接当CLI/服务性能。

## 验证与部署

先`.venv\Scripts\python.exe -m pytest -q tests/test_asd_rmsnorm_validation.py tests/test_asd_rmsnorm_suite.py`，再完整回归。内存重算旧48开发分析，必须逐字段相同，旧文件不改。冻结源码/协议/统计并导出Git blobs，独立服务器/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-rmsnorm-validation。

入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_rmsnorm_validation_pipeline`；worker `-m scripts.run_asd_rmsnorm_validation_probe --config configs/runtime-dflash.json --shard 0/1 --output results/raw/asd-rmsnorm-validation/part-0或part-1`。重建`.venv\Scripts\python.exe -m scripts.write_asd_rmsnorm_validation_report`。发布小型原始记录/测试/报告，核验旧结果和infra引用，排除模型、完整hidden/logits、缓存和凭据。
