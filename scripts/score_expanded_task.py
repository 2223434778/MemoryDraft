"""Single isolated quality worker; quality time is outside inference timing."""
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.deps'),str(ROOT/'third_party/expanded')]
os.environ['NLTK_DATA']=str(ROOT/'.nltk')
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def evaluate(record,text):
    benchmark=record['benchmark']
    if benchmark=='ifeval':
        import langdetect
        langdetect.DetectorFactory.seed=0
        from instruction_following_eval.evaluation_lib import InputExample,test_instruction_following_strict,test_instruction_following_loose
        inp=InputExample(**record['ifeval'])
        strict=test_instruction_following_strict(inp,{inp.prompt:text})
        loose=test_instruction_following_loose(inp,{inp.prompt:text})
        return dict(correct=strict.follow_all_instructions,instructions=strict.follow_instruction_list,
            loose_correct=loose.follow_all_instructions,loose_instructions=loose.follow_instruction_list,
            metric='official IFEval prompt/instruction strict and loose')
    if benchmark.startswith('longbench-'):
        metrics=module('longbench_metrics',ROOT/'third_party/expanded/longbench/metrics.py')
        fn=metrics.rouge_score if record['subset'] in ('multi_news','gov_report') else metrics.qa_f1_score
        value=max(fn(text,answer) for answer in record['answers'])
        return dict(correct=None,score=float(value),metric='official ROUGE-L' if fn==metrics.rouge_score else 'official answer F1')
    from memorydraft.evaluation.expanded_suite import validate_generated_code,extract_code
    code=extract_code(text)
    try:validate_generated_code(code)
    except (ValueError,SyntaxError) as error:
        return dict(correct=False,error=str(error),metric='test pass; source restrictions')
    if benchmark=='canitedit':
        import subprocess,tempfile
        limits='import resource\nresource.setrlimit(resource.RLIMIT_CPU,(8,8))\nresource.setrlimit(resource.RLIMIT_AS,(8589934592,8589934592))\n'
        with tempfile.TemporaryDirectory(prefix='canitedit-',dir=ROOT/'artifacts') as temp:
            # Some hidden tests inspect class source; -c has no source file.
            script=Path(temp)/'candidate.py'
            dependency_path='import sys\nsys.path.insert(0,'+repr(str(ROOT/'.deps'))+')\n'
            script.write_text(limits+dependency_path+code+'\n'+record['tests'],encoding='utf-8')
            p=subprocess.run([sys.executable,'-I',str(script)],cwd=temp,
                env={'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''},capture_output=True,text=True,timeout=12)
        return dict(correct=p.returncode==0,error=p.stderr[-400:] if p.returncode else None,
            metric='official hidden tests; isolated Python 3.11 execution')
    if benchmark=='livecodebench':
        import resource
        resource.setrlimit(resource.RLIMIT_CPU,(60,60))
        resource.setrlimit(resource.RLIMIT_AS,(4294967296,4294967296))
        tests=module('lcb_testing',ROOT/'third_party/expanded/lcb/testing_util.py')
        results,metadata=tests.run_test({'input_output':record['input_output']},test=code,timeout=2)
        return dict(correct=bool(results) and all(bool(v==True) for v in results),
            tests=len(results),passed=sum(bool(v==True) for v in results),metadata=metadata,
            metric='official LiveCodeBench public+private tests; 2s/test')
    raise ValueError('Unknown quality worker task')


if __name__=='__main__':
    try:
        value=json.load(sys.stdin)
        result=evaluate(value['record'],value['text'])
    except Exception as error:
        result=dict(correct=None,evaluation_error=type(error).__name__+': '+str(error))
    print(json.dumps(result))
