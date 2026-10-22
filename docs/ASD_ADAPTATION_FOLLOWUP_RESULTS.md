# 阶段5后续：同位置严格前缀与独立GSM评估

仅新训ordinary-position，保持原ASD适配器不变、匹配其64训练状态的生成位置；相同32题、初始化、更新、内部校准和选模规则。旧8题单列，新16题为项目未使用的官方test题；不是上游未见数据或多seed确认。

|范围|阶段/模型/策略|时延/base|时延/普通|时延/同位置|TPS/base|正确或共同128题数|安全|
|---|---|---:|---:|---:|---:|---:|---|
|observed|natural/base/DFlash|1.000000|0.991756|1.006227|1.000000|8/8|True|
|observed|natural/base/ASD|1.000000|1.011147|1.023323|1.000000|8/8|True|
|observed|natural/ordinary/DFlash|1.008616|1.000000|1.014705|0.991756|8/8|True|
|observed|natural/ordinary/ASD|0.991738|1.000000|1.014051|1.015950|8/8|True|
|observed|natural/ordinary-position/DFlash|0.994288|0.985906|1.000000|1.006227|8/8|True|
|observed|natural/ordinary-position/ASD|0.980634|0.990701|1.000000|1.043126|8/8|True|
|observed|natural/asd-prefix/DFlash|0.987108|0.978963|0.992926|1.013570|8/8|True|
|observed|natural/asd-prefix/ASD|0.955884|0.966104|0.976189|1.045667|8/8|True|
|observed|prefix/base/DFlash|1.000000|0.986682|1.002459|1.000000|7|True|
|observed|prefix/base/ASD|1.000000|0.986135|0.997701|1.000000|7|True|
|observed|prefix/ordinary/DFlash|1.013828|1.000000|1.016090|0.986682|7|True|
|observed|prefix/ordinary/ASD|1.018455|1.000000|1.012105|0.986135|7|True|
|observed|prefix/ordinary-position/DFlash|0.998101|0.984586|1.000000|1.002459|7|True|
|observed|prefix/ordinary-position/ASD|1.010307|0.992411|1.000000|0.997701|7|True|
|observed|prefix/asd-prefix/DFlash|0.988682|0.975531|0.990765|1.012170|7|True|
|observed|prefix/asd-prefix/ASD|0.970204|0.956744|0.964816|1.034567|7|True|
|independent|natural/base/DFlash|1.000000|1.004889|1.001086|1.000000|15/16|True|
|independent|natural/base/ASD|1.000000|1.021827|1.156570|1.000000|14/16|True|
|independent|natural/ordinary/DFlash|0.995356|1.000000|0.996291|1.004889|15/16|True|
|independent|natural/ordinary/ASD|1.003074|1.000000|1.158455|1.017293|14/16|True|
|independent|natural/ordinary-position/DFlash|0.999334|1.004071|1.000000|1.001086|15/16|True|
|independent|natural/ordinary-position/ASD|0.926838|0.949510|1.000000|1.027394|14/16|True|
|independent|natural/asd-prefix/DFlash|0.998810|1.003530|0.999557|1.001719|15/16|True|
|independent|natural/asd-prefix/ASD|0.991731|0.998135|1.077662|1.013302|14/16|True|
|independent|prefix/base/DFlash|1.000000|1.012252|0.990520|1.000000|16|True|
|independent|prefix/base/ASD|1.000000|1.018432|1.038071|1.000000|16|True|
|independent|prefix/ordinary/DFlash|0.988952|1.000000|0.979032|1.012252|16|True|
|independent|prefix/ordinary/ASD|0.984321|1.000000|1.019719|1.018432|16|True|
|independent|prefix/ordinary-position/DFlash|1.011201|1.022957|1.000000|0.990520|16|True|
|independent|prefix/ordinary-position/ASD|0.967683|0.983519|1.000000|1.038071|16|True|
|independent|prefix/asd-prefix/DFlash|1.004884|1.016006|0.993676|0.997671|16|True|
|independent|prefix/asd-prefix/ASD|0.989509|1.008651|1.028508|1.020238|16|True|

共同128的实际比值及预先指定的请求级bootstrap、每题质量/截断/token/调用数见analysis.json。完整生成与共同128集合不同，不直接相减解释因果。新的严格源不必落在真实Draft轮边界，匹配的是Target确认前缀位置，不宣称这是线上可用的额外动作。
