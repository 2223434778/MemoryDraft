from types import SimpleNamespace

import pytest
import torch

from memorydraft.evaluation.asd_paper import request_for
from memorydraft.memory.store import Snapshot
from scripts.run_external_trees import external_execute, result_tokens, load_upstreams, UPSTREAMS, collect_bastion


@pytest.mark.parametrize('arm,budget', [('DDTree-32',31),('DDTree-128',127),('BASTION',None)])
def test_official_adapter_passes_budget_prompt_stops_and_whole_output(monkeypatch, arm, budget):
    for name in ('reset_peak_memory_stats',):
        monkeypatch.setattr(torch.cuda,name,lambda *a:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *a:123)
    calls = []
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            calls.append(('prompt',messages,kwargs))
            return torch.tensor([[5,6]])
        def decode(self,tokens,**kwargs):
            assert tokens == [7,8,9]
            return 'answer'
    def generate(*args,**kwargs):
        calls.append(('generation',args,kwargs))
        if arm == 'BASTION':
            bastion.build_adaptive_best_tree_from_draft_logits()
        else:
            dd.build_ddtree_tree()
        return SimpleNamespace(output_ids=torch.tensor([[5,6,7,8,9]]),
            acceptance_lengths=[2],time_to_first_token=.01)
    dd = SimpleNamespace(ddtree_generate=generate,build_ddtree_tree=lambda:(torch.tensor([7,8]),))
    bastion = SimpleNamespace(bastion_generate=generate,
        build_adaptive_best_tree_from_draft_logits=lambda:SimpleNamespace(token_ids=torch.tensor([7,8,9])))
    engine = SimpleNamespace(target=SimpleNamespace(device='cpu',config=SimpleNamespace(max_position_embeddings=99)),
        tokenizer=Tokenizer(),stops=[9],synchronize=lambda:None,
        draft=SimpleNamespace(mask_token_id=0,block_size=16),
        config=SimpleNamespace(template_options={'enable_thinking':False},max_input_tokens=80))
    record = dict(id='test',benchmark='gsm8k',question='How many?',answer='#### 1')
    row = external_execute(engine,record,arm,16,dd,bastion,{'draft_latency_s':.01,'non_verify_latency_s':.001})
    assert row['token_ids']==[7,8,9] and row['finish_reason']=='eos'
    assert row['rounds']==[dict(advance=2,tree_nodes=3)] and row['peak_gpu_bytes']==123
    request=request_for(record,16)
    assert row['snapshot_id']==Snapshot(request.user_id,request.order,()).snapshot_id
    assert calls[0][1]==list(request.messages)
    assert calls[0][2]['enable_thinking'] is False
    kwargs=calls[1][2]
    if budget is not None:
        assert kwargs['tree_budget']==budget and kwargs['save_tree_traces'] is False
    else:
        assert kwargs['return_stats'] is True and kwargs['cost_gpu_type']=='a6000'
    assert row['latency_seconds']>0


@pytest.mark.parametrize('output,limit,message', [
    ([5,0,7],4,'prefix'),([5,6,7,8],1,'limit'),([5,6,9,7],4,'EOS')])
def test_rejects_upstream_contract_violations(output,limit,message):
    with pytest.raises(ValueError,match=message):
        result_tokens(torch.tensor([output]),[5,6],limit,[9])


def test_upstream_identity_checked_before_import(tmp_path):
    import json
    manifest={name:dict(commit=commit,url=url,license='MIT',files={})
              for name,(commit,url) in UPSTREAMS.items()}
    manifest['ddtree']['commit']='different'
    (tmp_path/'UPSTREAM_TREES.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='identity'):
        load_upstreams(tmp_path)


def test_upstream_calibration_context_prefill_cannot_build_gradients():
    observed=[]
    def collect(**kwargs):
        observed.append((torch.is_grad_enabled(),torch.is_inference_mode_enabled()))
        assert kwargs=={'target':'same-model'}
        return 'probe'
    assert collect_bastion(SimpleNamespace(collect_calibration_data=collect),target='same-model')=='probe'
    assert observed==[(False,True)] and torch.is_grad_enabled()
