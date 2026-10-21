# RMSNorm冻结候选独立验证

48 validation题，六类各8；1152配对计时+96 AR完整输出对照，另20短warmup。上限4096/自然EOS，reserve96未生成。首轮同时评分，不额外加算生成；未按验证结果调参。

|方法|编译/eager自然时延比|题级分层95% bootstrap|更快题|
|---|---:|---|---:|
|DFlash|0.973216|[0.971616,0.975672]|47/48|
|ASD|0.972030|[0.971376,0.972684]|48/48|
|ASD-Run|0.972200|[0.971696,0.972704]|48/48|
|ASD-Utility|0.971433|[0.969198,0.972971]|48/48|

## 预登记工程判定

|条件|通过|
|---|---|
|all_240_full_outputs_identical|True|
|no_observed_quality_or_completion_regression|True|
|strict_DFlash_matches_AR_in_both_profiles|True|
|all_four_overall_ci_upper_below_one|True|
|no_family_point_regression|True|
|resource_bounds|True|
|frozen_source_and_initial_isolation|True|

显式profile接入条件全部满足：True

## 分任务与质量

|任务|方法|时延比[95%]|token相同|主质量提升/下降/未评估|
|---|---|---|---:|---|
|gsm8k|DFlash|0.968684 [0.967833,0.969521]|8/8|0/0/0|
|gsm8k|ASD|0.968864 [0.967700,0.970045]|8/8|0/0/0|
|gsm8k|ASD-Run|0.968851 [0.968186,0.969459]|8/8|0/0/0|
|gsm8k|ASD-Utility|0.969158 [0.968438,0.969799]|8/8|0/0/0|
|math500|DFlash|0.969224 [0.967542,0.971506]|8/8|0/0/0|
|math500|ASD|0.968810 [0.967766,0.969809]|8/8|0/0/0|
|math500|ASD-Run|0.967957 [0.967090,0.968889]|8/8|0/0/0|
|math500|ASD-Utility|0.968898 [0.967570,0.970205]|8/8|0/0/0|
|mbpp|DFlash|0.972326 [0.971427,0.973114]|8/8|0/0/0|
|mbpp|ASD|0.972382 [0.970235,0.974504]|8/8|0/0/0|
|mbpp|ASD-Run|0.973582 [0.972319,0.974951]|8/8|0/0/0|
|mbpp|ASD-Utility|0.972662 [0.970849,0.974620]|8/8|0/0/0|
|humaneval|DFlash|0.973274 [0.971163,0.975529]|8/8|0/0/0|
|humaneval|ASD|0.971507 [0.970328,0.972705]|8/8|0/0/0|
|humaneval|ASD-Run|0.972453 [0.971174,0.973600]|8/8|0/0/0|
|humaneval|ASD-Utility|0.973847 [0.971509,0.976877]|8/8|0/0/0|
|squad|DFlash|0.984639 [0.976137,0.998592]|8/8|0/0/0|
|squad|ASD|0.979691 [0.977377,0.982197]|8/8|0/0/0|
|squad|ASD-Run|0.978879 [0.976958,0.980817]|8/8|0/0/0|
|squad|ASD-Utility|0.973407 [0.961016,0.980831]|8/8|0/0/0|
|json|DFlash|0.971237 [0.970227,0.972266]|8/8|0/0/0|
|json|ASD|0.970967 [0.969999,0.971873]|8/8|0/0/0|
|json|ASD-Run|0.971517 [0.970550,0.972447]|8/8|0/0/0|
|json|ASD-Utility|0.970634 [0.969507,0.971741]|8/8|0/0/0|

|任务|方法|eager/编译主正确|eager/编译secondary|eager/编译截断|未评估|SQuAD F1|
|---|---|---|---|---|---|---|
|gsm8k|AR|8/8（各8）|8/8|0/0|0/0|None/None|
|gsm8k|DFlash|8/8（各8）|8/8|0/0|0/0|None/None|
|gsm8k|ASD|7/7（各8）|7/7|0/0|0/0|None/None|
|gsm8k|ASD-Run|8/8（各8）|8/8|0/0|0/0|None/None|
|gsm8k|ASD-Utility|7/7（各8）|7/7|0/0|0/0|None/None|
|math500|AR|6/6（各8）|6/6|1/1|0/0|None/None|
|math500|DFlash|6/6（各8）|6/6|1/1|0/0|None/None|
|math500|ASD|6/6（各8）|6/6|0/0|0/0|None/None|
|math500|ASD-Run|6/6（各8）|6/6|0/0|0/0|None/None|
|math500|ASD-Utility|7/7（各8）|7/7|0/0|0/0|None/None|
|mbpp|AR|6/6（各8）|6/6|0/0|0/0|None/None|
|mbpp|DFlash|6/6（各8）|6/6|0/0|0/0|None/None|
|mbpp|ASD|5/5（各8）|5/5|0/0|0/0|None/None|
|mbpp|ASD-Run|6/6（各8）|6/6|0/0|0/0|None/None|
|mbpp|ASD-Utility|6/6（各8）|6/6|0/0|0/0|None/None|
|humaneval|AR|7/7（各8）|7/7|0/0|0/0|None/None|
|humaneval|DFlash|7/7（各8）|7/7|0/0|0/0|None/None|
|humaneval|ASD|7/7（各8）|7/7|0/0|0/0|None/None|
|humaneval|ASD-Run|7/7（各8）|7/7|0/0|0/0|None/None|
|humaneval|ASD-Utility|7/7（各8）|7/7|0/0|0/0|None/None|
|squad|AR|6/6（各8）|6/6|0/0|0/0|0.9571428571428572/0.9571428571428572|
|squad|DFlash|6/6（各8）|6/6|0/0|0/0|0.9571428571428572/0.9571428571428572|
|squad|ASD|5/5（各8）|5/5|0/0|0/0|0.8654761904761905/0.8654761904761905|
|squad|ASD-Run|6/6（各8）|6/6|0/0|0/0|0.9571428571428572/0.9571428571428572|
|squad|ASD-Utility|6/6（各8）|6/6|0/0|0/0|0.9571428571428572/0.9571428571428572|
|json|AR|8/8（各8）|8/8|0/0|0/0|None/None|
|json|DFlash|8/8（各8）|8/8|0/0|0/0|None/None|
|json|ASD|8/8（各8）|8/8|0/0|0/0|None/None|
|json|ASD-Run|8/8（各8）|8/8|0/0|0/0|None/None|
|json|ASD-Utility|8/8（各8）|8/8|0/0|0/0|None/None|

## 完整性与范围

{"questions": 48, "timed": 1152, "references": 96, "quality_rows": 480, "eager_old_trajectory_checks": 0, "unique_texts_rescored": 105, "compiled_token_matches": 240, "compiled_ledger_matches": 182, "strict_matches_AR": {"eager": 48, "rmsnorm-inductor": 48}, "compile_events_during_formal": 0, "compile_events_during_AR": 0}

验证题没有旧生成参考；两profile使用本轮匹配eager对照及相同Target输入。每profile三次重复精确重现；所有数值分歧/错答/截断保留。逐题首次分歧、长度/调用密度/摊销成本和质量详见analysis.json及原始gzip JSONL。

MATH规范答案主评分与有限标量等价分列，代码仅提供测试，SQuAD EM/F1，JSON严格合同；未评估与待复核不可冒充已确认失败。AR仅一次输出/质量对照，不统计AR速度。

判定是本轮显式执行选项的工程准入，不是任意输入无损或统计质量非劣证明。未通过不删除样本、不在validation调参；reserve保持未生成。

重建：`.venv\Scripts\python.exe -m scripts.write_asd_rmsnorm_validation_report`；协议ASD_RMSNORM_VALIDATION_PROTOCOL.md。
