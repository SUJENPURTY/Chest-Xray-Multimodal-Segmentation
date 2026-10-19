"""CPU-only preservation inventory. Never imports torch or starts training."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(4*1024**2):
            result.update(block)
    return result.hexdigest()


def main():
    if (HERE/'training_microsteps.csv').exists() or (HERE/'checkpoints/latest.json').exists():
        raise SystemExit('Existing training state preserved; preparation cannot reset it')
    inventory_path = HERE/'protected_sources.json'
    if inventory_path.exists():
        print('Existing preservation inventory retained. No files overwritten.')
        return
    config = json.loads((HERE/'config.json').read_text())
    parent = ROOT/config['parent_checkpoint']
    assert parent.is_file()
    best = json.loads((ROOT/'experiments/phase2_pilot/checkpoints/best.json').read_text())
    assert best['epoch']==2 and best['sha256']==config['parent_checkpoint_sha256']
    calibration = json.loads((ROOT/'experiments/phase2_threshold_calibration/calibration_summary.json').read_text())
    assert calibration['status']=='completed' and calibration['baseline_verification']['passed']
    assert calibration['best_validation_threshold']==config['comparison_threshold']
    protected = set(ROOT.glob('notebooks/*.ipynb'))
    protected.update(ROOT.glob('src/**/*.py'))
    protected.update(ROOT.glob('scripts/*.py'))
    for directory in ['experiments/phase2_pilot','experiments/phase2_threshold_calibration']:
        protected.update(p for p in (ROOT/directory).rglob('*') if p.is_file() and 'cache' not in p.parts and '__pycache__' not in p.parts and 'tmp' not in p.parts)
    protected.add(ROOT/'outputs/run/checkpoints/best_dice.ckpt')
    protected.update(ROOT.glob('outputs/run/prepared_annotations/*'))
    records = {}
    for path in sorted(protected):
        if not path.is_file():
            continue
        stat = path.stat()
        record = {'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
        if stat.st_size <= 8*1024**2 or path.suffix in {'.py','.ipynb'}:
            record['sha256'] = digest(path)
        records[str(path.relative_to(ROOT))] = record
    with inventory_path.open('x',encoding='utf-8') as handle:
        json.dump({'files':records,'coverage':'All protected files: size/mtime; source code, notebooks and files <=8 MiB: fresh SHA256. Phase 2 parent checkpoint SHA256 is verified at execution before updates. Dataset pixels are not newly hashed.'},handle,indent=2)
    print('Preparation complete; protected files:',len(records))
    print('Free disk GiB:',round(shutil.disk_usage(HERE).free/2**30,2))
    print('No training, model loading, CUDA use, or test evaluation performed.')


if __name__=='__main__':
    main()
