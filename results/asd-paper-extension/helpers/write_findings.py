"""Build narrative and supplemental tables from the sealed extension artifacts."""
from pathlib import Path
import hashlib,json,subprocess

ROOT=Path.cwd();OUT=ROOT/'results/asd-paper-extension'


def main():
    a=json.loads((OUT/'analysis.json').read_bytes());d=json.loads((OUT/'descriptive.json').read_bytes())
    reward=json.loads((OUT/'reward-analysis.json').read_bytes());contracts=json.loads((OUT/'open-contract-audit.json').read_bytes())
    audit=json.loads((OUT/'audit.json').read_bytes());quality=json.loads((OUT/'quality-audit.json').read_bytes())
    review_commit=subprocess.check_output(['git','rev-parse','8c48406'],text=True).strip()
    pct=lambda r:f'{(r-1)*100:+.2f}%'
    family_order=('gsm8k','math500','humaneval','mbpp','mmlu','mtbench','alpaca')
    labels=dict(gsm8k='GSM8K',math500='MATH-500',humaneval='HumanEval',mbpp='MBPP',mmlu='MMLU',mtbench='MT-Bench',alpaca='Alpaca')
    main_table=['|任务/匹配请求|严格树自然时延 vs DFlash|树+ASD自然时延 vs ASD|共同128时延 vs ASD（N）|固定TPS增益 vs ASD|自然质量 ASD→树+ASD|',
        '|---|---:|---:|---:|---:|---:|']
    for b in family_order:
        f=a['families'][b];n=f['BudgetTree-ASD']['natural'];p=f['BudgetTree-ASD']['prefix']['ASD']
        q=f"{f['ASD']['natural']['correct']}→{n['correct']}/{n['scored']}" if n['scored'] else '独立RM/局部性质'
        main_table.append(f"|{labels[b]}/{n['requests']}|{pct(f['BudgetTree']['natural']['DFlash']['ratio'])}|{pct(n['ASD']['ratio'])}|"
            f"{pct(p['ratio']) if p['ratio'] is not None else '—'}（{p['n']}）|{pct(f['BudgetTree-ASD']['fixed']['tps']/f['ASD']['fixed']['tps'])}|{q}|")
    mechanism=['|任务|实际Draft接受长度 ASD→树+ASD|验证轮 ASD→树+ASD|输出token ASD→树+ASD|验证节点/输出token ASD→树+ASD|',
        '|---|---:|---:|---:|---:|']
    indexed={(r['benchmark'],r['stage'],r['arm']):r for r in a['acceptance']}
    for b in family_order:
        x=indexed[b,'natural','ASD'];y=indexed[b,'natural','BudgetTree-ASD']
        mechanism.append(f"|{labels[b]}|{x['tau_emitted']:.3f}→{y['tau_emitted']:.3f}|{x['rounds']}→{y['rounds']}|"
            f"{x['output_tokens']}→{y['output_tokens']}|{x['nodes_per_output_token']:.3f}→{y['nodes_per_output_token']:.3f}|")
    rm=['|任务|方法 vs ASD|胜/平/负|偏好率（平局半计）|平均reward差|', '|---|---|---:|---:|---:|']
    for r in reward['summary']:
        if r['arm']=='ASD':continue
        rm.append(f"|{labels[r['benchmark']]}|{r['arm']}|{r['wins']}/{r['ties']}/{r['losses']}|{r['preference_rate']*100:.2f}%|{r['mean_difference']:+.4f}|")
    props=['|任务|方法|性质通过/总数|失败|','|---|---|---:|---:|']
    for r in contracts['summary']:
        total=sum(r.get(k,0) for k in ('pass','fail','uncertain'))
        props.append(f"|{labels[r['benchmark']]}|{r['arm']}|{r.get('pass',0)}/{total}|{r.get('fail',0)}|")
    headings=['# 七任务独立扩样：成本树的跨领域收益与边界','',
        '本轮已完成514个此前未用源任务、530个匹配输入，四方法9540次主计时、640次单参数控制、530条自然AR参考、64次真实自身历史第二轮。'
        '结果支持把成本受控树作为独立模块保留，也支持树与ASD在若干场景组合；收益并不局限于数学。MATH本轮退化、代码功能错误和MMLU输出合同损失全部保留。','',
        '## 可以接纳的结果','',
        '严格BudgetTree相对DFlash，在GSM/MATH/HumanEval/MBPP/MT-Bench/Alpaca的自然时延分别下降10.25%/6.19%/2.64%/14.87%/13.92%/17.15%；MMLU基本持平。'
        '全部530个匹配请求的自然输出与DFlash逐token、文本、停止一致；三种阶段共1590组对照均保持。这里保留的是DFlash的实际输出，另有一题与单token AR的有限精度分歧，详见审计。','',
        '树+ASD相对普通ASD，在新GSM/MBPP/Alpaca分别省7.72%/10.21%/17.89%，最终正确数分别114→117/128、31→32/64；Alpaca用独立偏好和局部性质评价。'
        'MT-Bench匹配单轮均题时延省5.51%，按16个对话合并两轮后省10.12%，真实自身历史双轮省12.93%；后两项的对话bootstrap时延比区间分别为[.8516,.9439]和[.8235,.9200]。'
        '两种历史的输入定义不同，分别报告。','',
        'HumanEval自然省3.26%但正确53→52/64；MATH自然慢5.50%（时延比区间[.9771,1.1418]），正确44→45/64；MMLU自然慢1.30%，只答字母的主合同正确39→37/114。'
        '这些结果没有要求所有数据都胜出，也没有将速度收益直接等同于质量保持。','',
        '\n'.join(main_table),'',
        '负的时延变化表示更快，正的TPS变化表示更高吞吐。自然时延是逐请求配对比值的均值；固定TPS是总token/总wall；二者不是互为倒数的同一统计量。'
        '表中MATH质量包含冻结的有限标量等价补充；保守主评分为ASD42→组合43，严格组41。MT有16源对话/32匹配轮次，主表区间按轮次，独立对话区间另列。','',
        '## 收益从哪里来','',
        'GSM自然实际Draft接受长度5.440→6.648、验证5673→4750轮；MBPP为4.564→5.879、567→463轮；Alpaca为1.951→2.848、7489→5578轮。'
        '三者逐题长度比均值分别约+0.47%、+2.27%、−2.15%；共同128的GSM/Alpaca时间分别省9.63%/18.36%。这提供了超出单纯缩短回答的计算增量证据。'
        'MBPP共同128仅3题，不能用该小子集代表全部代码请求。','',
        '\n'.join(mechanism),'',
        '以上为自然首重复的实际发出Draft token/验证轮，排除Target首token和recovery/bonus；接受率不以所有树节点作Draft分母。'
        '验证节点/输出token上升说明分支并非零成本：更多并行候选用来交换更少验证轮，完整wall已包含构树、选择、拷贝、验证和KV提交。调用量不含初始prefill，wall包含。','',
        '四组因子对照表明主要增量来自候选树组织。固定工作量下，严格树/线性与近似树/线性的逐题时延比、交互比均已保存；多数任务的交互比大于1，Alpaca约1。'
        '支持有效组合，并不支持宣称普遍超乘法协同。不同近似方法的中间路径不相同，这不是同状态因果干预。','',
        '## 自然停止和质量的边界','',
        'MATH逐题长度比均值上升11.08%，固定TPS只提高4.37%，共同128基本持平；40个双方保守主评分正确且EOS的事后交集仍慢6.14%。'
        '因此退化不能只归因于一题截断，也不能靠接受长度提高推断完整请求变快。math500:323的ASD为1537 token且最终1/3，组合循环到4096且无最终boxed答案；213则从正确2变为错误1。'
        '正向例156/224/328分别修正最终36度→30度、3/256→9/256、遗漏x=0造成的范围计数2→3；主评分没有事后改写。','',
        'HumanEval/MBPP/MMLU固定256结果中，组合分别约66.71%/80.43%/89.42%的token位于本轮实际出现EOS ID的首次位置之后。'
        'HumanEval的固定TPS+48.42%因此不能称为正常代码请求提速48%；自然只有3.26%且新增一个净测试失败。','',
        '六个ASD与组合之间发生变化的代码题，已用12次小型语义见证核对。组合失去HumanEval91（把Is当独立I）、99（−15.5被错舍为−17），得到103（均值取整修复）。'
        'MBPP190把严格内部点4错计9，333修复原地排序无返回值的测试合同，74补回不同输入长度的检查。全部按原测试记分，未用新反例重写主准确率；具体代码SHA和输入/输出见quality-audit.json。','',
        'MMLU四组逐题可提取的首选项相同，均78/114正确、105/114可提取。组合相对ASD的主合同变化为5失3得，来自额外文字/格式，不能偷偷将首选项诊断替换只答字母的主评分。','',
        '## 开放质量：偏好与明确性质分开','',
        '独立训练的Skywork Reward V2 Qwen3-8B完成320行/223去重对话；8控制全过、重复误差0、无裁剪或未评分。'
        '它是pointwise偏好代理，既不是官方MT-Bench/GPT-4分数，也不是AlpacaEval，更不是校准的正确率。主raw质量字段沿用旧pairwise占位描述，实际本轮评价以这里的独立RM及属性审查为准。','',
        '\n'.join(rm),'',
        '组合相对ASD为MT7胜2平7负、Alpaca26胜7平31负；不能声称质量全面提高。相对严格组，Alpaca的质量—速度取舍也应保留。','',
        '另有34项题干定义性质/136方法单元：100个语义单元和10个自动失败逐项匿名审阅，26个自动通过保留。'
        f'判定SHA为d4ff8dd6cb717b08b21b829c8e40865013c0bb60f211a7028d1684dd4d8d069c，先以Git `{review_commit}` 冻结，再揭开标签、查看RM逐题分数。'
        '审阅为agent辅助，不冒充独立人类专家；性质不是完整回答准确率。','',
        '\n'.join(props),'',
        'ASD与组合在这34项的通过/失败结果相同：MT13/16、Alpaca15/18。严格组分别12/16、17/18。'
        '所有方法漏提取Europe、保留She/her、误认None边界LCS无bug，或漏删姓名的失败均保留；两个近似组在秘书题给出Alice而严格组陷入长循环，但在250词上限和文章组件去重上退化。'
        '颜色倒序单项通过的回答仍可错误声称Cheryl是秘书，说明局部性质通过不能代表整答正确。','',
        '## 控制曲线与使用范围','',
        '640次预定控制覆盖两近似方法的B4/B12/g.125/g.5/M1，B0/B8复用主轨迹；每族仅前4源任务，MT展开为8轮。'
        'MATH切片中M1把B8的2/4正确、1截断改为3/4、0截断；其他配置不形成跨任务统一最优。HumanEval g.5固定TPS提高但自然仍退化，MMLU g.5主质量下降。'
        '这些是预定小切片诊断，不据此改主参数、挑赢家或宣称新的独立验证；开放控制输出保留，本轮未另评其RM偏好。','',
        '建议保留两条明确路径：严格成本树用于希望保持DFlash输出的场景；树+ASD用于接受已披露近似风险的GSM、MBPP及自由生成场景。'
        'MMLU的精确输出合同、HumanEval的功能错误以及MATH的长路径风险不作通用升级。旧Joint证据仍保留，不因本轮结果重命名为已验证的新优势。'
        '输入长度、AR输出长度和双方正确子集只用于后验解释，未训练路由；AR输出长度是未来信息，不能当作部署时已知标签。','',
        '## 执行、数值和复现','',
        '本轮任务族与ASD v3第4节对齐，但模型/硬件/数据子集/重复次数和开放裁判均有明确差异，见[对应关系](ASD_PAPER_ALIGNMENT.md)。'
        '参考[ASD论文v3](https://arxiv.org/html/2608.03447v3#S4)，不把Stanford Alpaca子集称为AlpacaEval，不拼接文献AR速度与本机新方法时延。'
        '原249请求的固定AR对照仍单列于旧METRICS报告；本轮新固定比较直接对照本轮ASD/DFlash。','',
        '原6d89480波次在math500:323停止，完整172题原行字节复用，未完成题在1d39c56隔离root整题续跑。'
        '严格树/DFlash在该题与AR于index602分歧，实际前向最大logits差1.52587890625e-5，两个近并列词顺序互换；重复轨迹及实际argmax均通过。'
        '修订只把跨前向形状的身份差异单列，接受规则、Target输入和推理核心均未改；不宣称严格组全量逐字等同AR。原失败ZIP/部分输出与诊断全部保留，详见[续跑协议](ASD_EXTENSION_CONTINUATION_PROTOCOL.md)。','',
        f"相邻严格控制漂移>3%为{a['fixed_drift_over_3pct']}/{len(a['fixed_drift'])}，全部保留。两卡存在实际软件温度限频观察，未改风扇/功率/频率；不宣称与论文冷态硬件TPS直接可比。"
        '硬件XML首快照混入最高/默认频率的问题已留原记录并用显式current-clock查询纠正，不用2100/8001MHz冒称当时运行频率。','',
        '本机新专项69通过，最后元数据补丁17通过；服务器181通过。前一6d源完整1055通过/7跳过、附加规则12通过及此前临时目录/socket环境失败记录各自保留。'
        '独立核验2341新源/2330旧源、1117旧载荷、172完整复用、7000条实际发出位置轨迹、136属性绑定和320奖励消息绑定通过；统计JSON/Markdown跨平台重建逐字节相同。'
        '面试版三个引用保持27e36199f8a65acdb9a9c42ccb744ddfa9a9402b。','',
        '主要证据：[完整表](ASD_PAPER_EXTENSION_RESULTS.md)、[数据/源审计](../results/asd-paper-extension/audit.json)、[代码与质量见证](../results/asd-paper-extension/quality-audit.json)、'
        '[奖励对照](../results/asd-paper-extension/reward-analysis.json)、[匿名性质审查](../results/asd-paper-extension/open-contract-audit.json)。','',
        '```powershell',
        '.venv\\Scripts\\python.exe -m results.asd-paper-extension.inspect_results',
        '.venv\\Scripts\\python.exe -m results.asd-paper-extension.describe_results',
        '.venv\\Scripts\\python.exe -m results.asd-paper-extension.helpers.write_findings',
        '```','']
    (ROOT/'docs/ASD_PAPER_EXTENSION_FINDINGS.md').write_bytes('\n'.join(headings).encode())
    supplement=[(OUT/'report.md').read_text(encoding='utf-8'),'','## 独立开放质量','',*rm,'',*props,'',
        '以上RM为偏好代理、性质为选定局部检查，均不等于完整回答准确率。MT真实历史与匹配历史分列。','',
        '|方法|MT匹配历史双轮时延vs ASD|MT自身历史双轮时延vs ASD|自身历史95%对话区间|',
        '|---|---:|---:|---|']
    for r in d['mtbench_dialogues']:
        supplement.append(f"|{r['arm']}|{pct(r['matched_history_mean_ratio'])}|{pct(r['own_history_mean_ratio'])}|{r['own_history_ci95']}|")
    supplement+=['','## 实际推进与节点成本','',*mechanism,'','## 生成资源与评价成本','',
        '|GPU|续跑加载秒|只读成本表秒|峰allocated GiB（含复用前峰）|编译cache字节|预热后/结束图数|',
        '|---|---:|---:|---:|---:|---:|']
    for r in a['resources']:
        supplement.append(f"|{r['shard']}|{r['load_seconds']:.3f}|{r['cost_table_read_seconds']:.6f}|{r['peak_gpu_bytes']/2**30:.3f}|{r['cache_bytes']}|{r['compiler_after_warmup']['graph_count']}/{r['compiler']['graph_count']}|")
    p=json.loads((OUT/'reward/provenance.json').read_bytes());e=json.loads((OUT/'reward/execution.json').read_bytes())
    supplement+=['','两个会话共32次预热（另有失败尝试和纯诊断）。模型加载、旧成本表校准、编译/预热与离线评分不冒充在线零成本。正式wall包含prefill/构树/选择/KV；调用量字段仅为验证轮。',
        f"本机奖励权重校验{p['verification_seconds']:.3f}s、模型加载{p['load_seconds']:.3f}s、任务前向{p['timed_forward_seconds']:.3f}s；子进程监督wall{e['wall_seconds']:.3f}s，源码校验另计。峰allocated {p['peak_gpu_bytes']/2**30:.3f}GiB。",'',
        '来源：新GPU源1d39c56、复用/奖励评价源6d89480；具体命令、归档SHA、失败、资源和与论文区别见[结论](ASD_PAPER_EXTENSION_FINDINGS.md)、PROGRESS及原始provenance。','']
    (ROOT/'docs/ASD_PAPER_EXTENSION_RESULTS.md').write_bytes('\n'.join(supplement).encode())
    print(json.dumps(dict(findings_sha256=hashlib.sha256((ROOT/'docs/ASD_PAPER_EXTENSION_FINDINGS.md').read_bytes()).hexdigest(),
        results_sha256=hashlib.sha256((ROOT/'docs/ASD_PAPER_EXTENSION_RESULTS.md').read_bytes()).hexdigest(),review_commit=review_commit)))


if __name__=='__main__':main()
