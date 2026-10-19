"""Standard-library preparation; reads previous experiments without importing them."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import os

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(4*1024*1024): h.update(b)
    return h.hexdigest()

source=(ROOT/'experiments/phase1b_memory_fix/run_phase1b.py').read_text(encoding='utf-8').split('\ndef weak_image_diagnostic')[0]
source=source.replace('HERE = Path(__file__).resolve().parent\nROOT = HERE.parents[1]',
    "HERE = Path(os.environ['PHASE1C_VARIANT_DIR']).resolve()\nROOT = Path(__file__).resolve().parents[2]\nassert HERE.is_relative_to(ROOT/'experiments/phase1c_normalization')")
source=source.replace('PHASE1B_FP32_SUBSET_V1','PHASE1C_NORMALIZATION_V1')
source=source.replace('Run: .venv/Scripts/python.exe -B -u experiments/phase1b_memory_fix/run_phase1b.py',
    'Utilities copied into Phase 1C; never imports the previous experiment.')
source=source.replace('    atomic_torch(path,payload)','    assert not path.exists(), str(path)\n    atomic_torch(path,payload)')
source=source.replace("torch.set_rng_state(saved['torch_rng'])","torch.set_rng_state(saved['torch_rng'].clone())")
source=source.replace("torch.cuda.set_rng_state_all(saved['cuda_rng'])","torch.cuda.set_rng_state_all([v.clone() for v in saved['cuda_rng']])")
ast.parse(source)
(HERE/'common.py').write_text(source,encoding='utf-8')
print('Disk:',shutil.disk_usage(ROOT),flush=True)
assert shutil.disk_usage(ROOT).free>20*2**30, 'Need >=20 GiB free for isolated periodic checkpoints'
protected=set(ROOT.glob('notebooks/*.ipynb'))|set(ROOT.glob('src/**/*.py'))|set(ROOT.glob('scripts/*.py'))
for directory in ['outputs/run','experiments/phase1_diagnostics','experiments/phase1b_memory_fix']:
    protected.update(p for p in (ROOT/directory).rglob('*') if p.is_file())
manifest=json.loads((ROOT/'experiments/phase1b_memory_fix/sample_manifest.json').read_text())
for r in manifest:
    protected.add(ROOT/'_qata_compatible/Train/GTs'/r['name'])
    protected.add(ROOT/'_qata_compatible/Train/Images'/r['name'].removeprefix('mask_'))
path=HERE/'preservation_before.json'
if not path.exists():
    inventory={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(protected)}
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    os.replace(tmp,path)
    print('Protected inventory:',len(inventory),flush=True)
else: print('Existing protected inventory preserved',flush=True)
