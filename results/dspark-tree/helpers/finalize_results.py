"""Derived descriptive tables; never changes generation rows or frozen grades."""
from pathlib import Path
import json,hashlib,subprocess,statistics

from scripts.write_dspark_tree_report import read
from scripts.run_dspark_tree import ARMS
from memorydraft.evaluation.asd_paper import BENCHMARKS


def main():
    root=Path.cwd();out=root/'results/dspark-tree'
    a=json.loads((out/'analysis.json').read_bytes());rows=read(out,'rows');refs=read(out,'references')
    samples=a['samples'];natural=[x for x in samples if x['stage']=='natural']
    originals=json.loads((out/'local-checks/preservation.json').read_bytes())
    for ref,oid in originals['protected_refs'].items():
        assert subprocess.check_output(['git','rev-parse',ref],text=True).strip()==oid,ref
    objects={}
    for raw in subprocess.check_output(['git','ls-tree','-r','-z','HEAD']).split(b'\0'):
        if raw:
            metadata,name=raw.split(b'\t',1);objects[name.decode()]=metadata.split()[2].decode()
    for name,oid in originals['protected_git_objects'].items():assert objects[name]==oid,name
    refmap={r['request_id']:r for r in refs};strict=[]
    for row in rows:
        if row['stage']=='natural' and row['repeat']==0 and not row['arm'].endswith('-ASD'):
            strict.append(dict(request_id=row['request_id'],arm=row['arm'],equal_ar=row['token_ids']==refmap[row['request_id']]['token_ids']))
    quality_changes=[]
    repeat_quality_changes=[]
    repeated={}
    for row in rows:
        key=row['request_id'],row['stage'],row['arm']
        if key in repeated and repeated[key]!=row['quality']:repeat_quality_changes.append(list(key))
        repeated[key]=row['quality']
    fixed_after_eos=[]
    for benchmark in BENCHMARKS:
        for arm in ('DFlash-Tree-ASD','DSpark-Tree-ASD'):
            selected=[r for r in rows if r['benchmark']==benchmark and r['stage']=='fixed' and r['arm']==arm]
            total=sum(len(r['token_ids']) for r in selected)
            after=sum(len(r['token_ids'])-r['token_ids'].index(151645)-1 if 151645 in r['token_ids'] else 0 for r in selected)
            fixed_after_eos.append(dict(benchmark=benchmark,arm=arm,eos_id=151645,tokens=total,
                tokens_after_first_observed_eos_id=after,fraction=after/total))
    index={(x['request_id'],x['arm']):x for x in natural}
    for identity in sorted(refmap):
        x=index[identity,'DSpark-Tree-ASD'];y=index[identity,'DFlash-Tree-ASD']
        q=lambda r:r['quality'].get('secondary_correct',r['quality']['correct'])
        if x['quality']['correct'] is not None and q(x)!=q(y):
            quality_changes.append(dict(request_id=identity,benchmark=x['benchmark'],dflash_quality=y['quality'],
                dspark_quality=x['quality'],dflash_finish=y['finish_reason'],dspark_finish=x['finish_reason']))
    provenance=[json.loads((out/f'part-{i}/provenance.json').read_bytes()) for i in (0,1)]
    audit=dict(protected_old_files=len(originals['protected_git_objects']),protected_refs=originals['protected_refs'],
        object_map_sha256=originals['object_map_sha256'],strict_natural_comparisons=strict,
        strict_equal_ar=sum(r['equal_ar'] for r in strict),quality_changes=quality_changes,
        repeat_quality_changes=repeat_quality_changes,fixed_after_eos=fixed_after_eos,
        all_primary_rows=len(rows),references=len(refs),source_commit='62e801067dcb990ea9ae9a18ed1c09f02e0d04ab',
        new_graphs=[p['compiler_end']['graph_count']-p['compiler_after_warmup']['graph_count'] for p in provenance])
    (out/'derived-audit.json').write_bytes((json.dumps(audit,indent=2)+'\n').encode('utf-8'))
    lines=['# DSpark 成本树：完整八组结果','',
        '28个已观察源任务/32匹配输入；自然EOS4096上限两次、128/EOS一次、固定256忽略EOS一次。',
        '所有表格保留8种方法。自然耗时为每请求重复均值再按请求取均值；正确数按每请求一次计数。',
        'MATH使用原有事先规定的有限标量等价补充评分；开放任务无独立质量裁判，不把速度视为质量保持。',
        '接受长度为实际发出Draft token总数/验证轮数，排除Target锚点及恢复/额外token。','']
    for b in BENCHMARKS:
        lines += [f'## {b}','',
            '|方法|请求数|自然均时(s)|输出token合计|正确/评分数|截断|实际Draft接受长度|验证轮合计|节点/输出token|峰显存(GiB)|',
            '|---|---:|---:|---:|---|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            rs=[x for x in natural if x['benchmark']==b and x['arm']==arm]
            scored=[x for x in rs if x['quality']['correct'] is not None]
            quality=f"{sum(bool(x['quality'].get('secondary_correct',x['quality']['correct'])) for x in scored)}/{len(scored)}" if scored else '未独立评分'
            rounds=sum(x['acceptance']['rounds'] for x in rs if x['acceptance'])
            emitted=sum(x['acceptance']['emitted_draft'] for x in rs if x['acceptance'])
            nodes=sum(x['acceptance']['verification_nodes'] for x in rs if x['acceptance'])
            tokens=sum(x['tokens'] for x in rs)
            lines.append(f"|{arm}|{len(rs)}|{statistics.mean(x['seconds'] for x in rs):.4f}|{tokens}|{quality}|{sum(x['finish_reason']=='length' for x in rs)}|{emitted/rounds if rounds else 0:.3f}|{rounds}|{nodes/tokens:.3f}|{max(x['peak_bytes'] for x in rs)/1024**3:.3f}|")
        lines += ['','|方法|固定256总TPS|固定总时(s)|固定实际Draft接受长度|','|---|---:|---:|---:|']
        for arm in ARMS:
            rs=[x for x in samples if x['benchmark']==b and x['arm']==arm and x['stage']=='fixed']
            rounds=sum(x['acceptance']['rounds'] for x in rs);emitted=sum(x['acceptance']['emitted_draft'] for x in rs)
            seconds=sum(x['seconds'] for x in rs)
            lines.append(f"|{arm}|{sum(x['tokens'] for x in rs)/seconds:.3f}|{seconds:.4f}|{emitted/rounds:.3f}|")
        lines += ['']
    lines+=['## 完整性与保留检查','',
        f"旧版本{audit['protected_old_files']}份配置、结果和核心实现Git对象保持一致；原DFlash分支保持8e7a80b。",
        f"自然严格组与AR相同{audit['strict_equal_ar']}/{len(strict)}；正式新编译图{audit['new_graphs']}。",
        f"DSpark组合相对DFlash组合发生评分变化的请求数{len(quality_changes)}，具体ID/原评分保存在derived-audit.json。",
        'MT按统一AR首轮历史分别比较两轮；本次未测各方法自身历史，也未作对话级合并结论。',
        '原生块长、训练检查点与Markov头同时变化；此处比较可用底座，不将结果单独归因某一个结构。','']
    (root/'docs/DSPARK_TREE_RESULTS.md').write_bytes('\n'.join(lines).encode('utf-8'))
    print(json.dumps(dict(rows=len(rows),references=len(refs),protected_old_files=audit['protected_old_files'],
        strict_equal_ar=audit['strict_equal_ar'],strict_total=len(strict),quality_changes=len(quality_changes))))


if __name__=='__main__':main()
