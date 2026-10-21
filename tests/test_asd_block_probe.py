from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM

from memorydraft.backends.asd import ASDConfig, ASDVerifier
from memorydraft.backends.dflash_reuse import generate_reuse
from scripts.run_asd_block_probe import check_native, experimental_block, prefix_check, selected_records


def test_experimental_attribute_restored_on_exception_without_rewriting_config():
    draft=SimpleNamespace(block_size=16,config=SimpleNamespace(block_size=16))
    with pytest.raises(RuntimeError):
        with experimental_block(draft,32):
            assert draft.block_size==32 and draft.config.block_size==16
            raise RuntimeError("test cancellation")
    assert draft.block_size==16
    for value in (True,8,0,32.,64):
        with pytest.raises(ValueError):
            with experimental_block(draft,value):pass
    draft.block_size=24
    with pytest.raises(ValueError,match="checkpoint"):
        with experimental_block(draft,24):pass


def test_selection_is_first_per_family_and_native_check_rejects_changed_budget():
    records=selected_records()
    assert [r["id"] for r in records]==["gsm8k:885","math500:329","mbpp:262","humaneval:82",
                                      "squad:5734025d4776f419006616c6","json:12"]
    row=dict(token_ids=[1],prompt_sha256="p",snapshot_id="s",asd=dict(spent=0),rounds=[],finish_reason="eos")
    check_native(row,dict(row))
    with pytest.raises(ValueError):check_native(row,dict(row,asd=dict(spent=1)))


def test_larger_length_boundary_is_detected_before_31_proposals():
    full=dict(token_ids=list(range(60)))
    prefix=dict(token_ids=list(range(40)),rounds=[dict(generated_before=1,proposed=31),dict(generated_before=20,proposed=20)])
    prefix["token_ids"][35]=99
    assert prefix_check(full,prefix,32)==dict(required=20,common=35,agrees=True)
    prefix["token_ids"][5]=99
    assert not prefix_check(full,prefix,32)["agrees"]


@pytest.fixture(scope="module")
def tiny():
    torch.set_num_threads(1);torch.manual_seed(119)
    config=Qwen3Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                      num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=256,
                      eos_token_id=None,pad_token_id=0)
    config._attn_implementation="eager"
    target=Qwen3ForCausalLM(config).eval();ids=torch.tensor([[3,4,5]])
    with torch.inference_mode():
        reference=target.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=65,
                                  do_sample=False,use_model_defaults=False,eos_token_id=None)
    return SimpleNamespace(target=target,ids=ids,reference=reference)


@pytest.mark.parametrize("size",[16,24,32])
@pytest.mark.parametrize("policy",[None,ASDConfig(budget=0),ASDConfig(budget=8,value="exact-run")])
def test_larger_actual_blocks_keep_target_features_kv_stream_and_eos(tiny,size,policy):
    class Draft(torch.nn.Module):
        block_size,mask_token_id,target_layer_ids=16,31,[0,1]
        config=SimpleNamespace(block_size=16)
        calls=[]

        def forward(self,*,target_hidden,noise_embedding,position_ids,past_key_values,**kwargs):
            cached=past_key_values.get_seq_length();n=target_hidden.shape[1]
            # Approximate acceptance changes the prefix: compare with a fresh
            # prefill of the actually committed text, never the strict AR path.
            current=torch.cat([tiny.ids,tiny.ids.new_tensor([commits[:-1]])],dim=1)
            out=tiny.target(current,output_hidden_states=True,use_cache=False)
            expected=torch.cat([out.hidden_states[1],out.hidden_states[2]],dim=-1)
            torch.testing.assert_close(target_hidden,expected[:,cached:cached+n],atol=2e-5,rtol=2e-5)
            assert noise_embedding.shape[1]==size and position_ids.shape[1]==n+size
            self.calls.append(size)
            values=noise_embedding.new_zeros((1,1,n+size,8));past_key_values.update(values,values,0)
            return torch.zeros_like(noise_embedding)

    draft=Draft();commits=[]
    verifier=None if policy is None else ASDVerifier(policy)
    with experimental_block(draft,size):
        result=generate_reuse(tiny.target,draft,tiny.ids,max_new_tokens=65,on_tokens=commits.extend,
                              verifier=verifier)
    if policy is None or policy.budget==0:
        assert result.sequences.tolist()==tiny.reference.tolist()
    else:
        assert 0<verifier.relaxed and verifier.spent<=policy.budget
    assert result.sequences[0,3:].tolist()==commits and len(commits)==65 and draft.calls
    stop=commits[12]
    expected=commits[:commits.index(stop)+1]
    commits.clear()
    with experimental_block(draft,size):
        result=generate_reuse(tiny.target,draft,tiny.ids,max_new_tokens=65,stop_token_ids=[stop,1000],
                              on_tokens=commits.extend,verifier=None if policy is None else ASDVerifier(policy))
    assert result.sequences[0,3:].tolist()==expected and draft.block_size==16


def test_actual_audited_dflash_forward_accepts_extrapolated_positions_and_cropped_kv(tiny):
    from transformers import AutoConfig, AutoModel
    directory=Path(__file__).resolve().parents[1]/"models/dflash-qwen3-8b"
    if transformers.__version__!="4.57.3" or not directory.exists():
        pytest.skip("Audited server model assets/profile only")
    config=AutoConfig.from_pretrained(directory,local_files_only=True)
    config.hidden_size,config.intermediate_size=32,64
    config.num_hidden_layers,config.num_target_layers=1,2
    config.num_attention_heads,config.num_key_value_heads,config.head_dim=4,2,8
    config.layer_types,config.block_size,config.vocab_size=["full_attention"],16,32
    config.dflash_config=dict(target_layer_ids=[0,1],mask_token_id=31)
    for attention in ("eager","sdpa"):
        config._attn_implementation=attention
        draft=AutoModel.from_config(config,trust_remote_code=True).float().eval()
        for size in (16,24,32):
            with experimental_block(draft,size):
                result=generate_reuse(tiny.target,draft,tiny.ids,max_new_tokens=65)
            assert result.sequences.tolist()==tiny.reference.tolist() and draft.block_size==16
