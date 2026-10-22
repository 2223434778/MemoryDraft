# 阶段3：有限真实计算动作检查

开发题六类各2，原FP32/RMSNorm/16块。请求128上限、局部64新token均遵守EOS；各动作相同范围token一致时才可比较成本，不把截断前缀当完整自然答案。oracle是乐观有限动作参照，不是线上方法或全局上界。

|范围|最优固定动作|事后oracle/固定|跨重复oracle/固定|留题预测/固定|安全|
|---|---|---:|---:|---:|---|
|request/overall|native|0.990718|0.991221|1.041159|True|
|request/gsm8k|native|1.000000|1.000000|1.000000|True|
|request/humaneval|native|1.000000|1.000000|1.000000|True|
|request/json|native|1.000000|1.000000|1.246954|True|
|request/math500|native|1.000000|1.000000|1.000000|True|
|request/mbpp|native|0.993560|0.993560|1.000000|True|
|request/squad|half|0.958543|0.961586|1.008199|True|
|branch/overall|native|0.951273|0.951369|1.045080|True|
|branch/gsm8k|native|0.991436|0.991523|1.000000|True|
|branch/humaneval|native|0.997311|0.997311|1.011865|True|
|branch/json|half|0.990952|0.990969|1.000918|True|
|branch/math500|native|0.994562|0.994787|1.000000|True|
|branch/mbpp|half|0.928661|0.928661|1.003124|True|
|branch/squad|half|0.836896|0.836896|1.298426|True|

是否进入实际选择器评估：False

门槛及完整四动作代价、逐状态和留题模型见协议及analysis.json。家族小样本仅用于发现适用设置，不能冒充独立泛化。所有预填充/共同重放排除在局部clock之外，但当前起草/验证/KV提交/后续与同步包含；请求clock则含全部线上生成成本。
