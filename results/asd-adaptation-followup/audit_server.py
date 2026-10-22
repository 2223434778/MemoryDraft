"""CPU-only postrun tensor provenance; private tensors never leave this host."""
import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import load_file

root=Path('/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-adaptation-followup')
base=root/'results/raw/asd-adaptation-followup'
sha=lambda b:hashlib.sha256(b).hexdigest()
execution=json.loads((base/'execution-verification.json').read_text())
assert execution['status']=='complete'
features=[]
for i in (0,1):
    for line in (base/f'collect-{i}/examples.jsonl').read_text().splitlines():
        row=json.loads(line);path=root/row['feature_path']
        assert sha(path.read_bytes())==row['feature_sha256']
        sample=torch.load(path,map_location='cpu',weights_only=True)
        assert sample['context'].shape[1]==len(row['prefix_ids'])-1 and sample['context'].dtype==torch.float32
        assert torch.isfinite(sample['context']).all()
        assert sample['prefix_ids'][0].tolist()==row['prefix_ids'] and sample['labels'][0].tolist()==row['teacher_tokens']
        assert sample['anchor'].item()==row['prefix_ids'][-1]
        features.append(dict(prefix_sha256=row['prefix_sha256'],shape=list(sample['context'].shape),bytes=path.stat().st_size))
meta=json.loads((base/'train/provenance.json').read_text())
adapters={}
for name in ('selected','final'):
    path=root/f'artifacts/followup-adapter/{name}.safetensors'
    assert sha(path.read_bytes())==meta['adapter_sha256' if name=='selected' else 'final_sha256']
    tensors=load_file(str(path))
    assert {n:list(t.shape) for n,t in tensors.items()}==meta['tensor_shapes']
    assert all(t.dtype==torch.float32 and torch.isfinite(t).all() for t in tensors.values())
    adapters[name]=dict(sha256=sha(path.read_bytes()),bytes=path.stat().st_size,tensors=len(tensors),parameters=sum(t.numel() for t in tensors.values()))
result=dict(status='complete',features=len(features),feature_bytes=sum(r['bytes'] for r in features),context_shapes=sorted({tuple(r['shape']) for r in features}),adapters=adapters)
(root/'artifacts/server-artifact-audit.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(result))
