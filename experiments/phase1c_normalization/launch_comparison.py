"""Standard-library controller: one ML subprocess at a time, accurate Python exit codes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
assert (HERE/'preservation_before.json').exists(),'Run prepare.py first'
log_path=HERE/f'controller_{time.strftime("%Y%m%d_%H%M%S")}.log'
for variant in ['A_current_bn','B_frozen_bn','C_groupnorm']:
    if (HERE/variant/'summary.json').exists():
        raise SystemExit('Existing completed results preserved; do not rerun the comparison')
    directory=HERE/variant
    directory.mkdir(exist_ok=True)
    console=directory/'console.log'
    if console.exists(): raise SystemExit('Existing log preserved: resume that variant explicitly')
    print('START',variant,time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
    with console.open('w',encoding='utf-8') as stream:
        command=[sys.executable,'-B','-u',str(HERE/'run_variant.py'),'--variant',variant]
        child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL,creationflags=0x08000000)
        (directory/'owned_process.json').write_text(json.dumps({'pid':child.pid,'command':command,
             'started':time.strftime('%Y-%m-%dT%H:%M:%S'),'controller_pid':os.getpid()},indent=2),encoding='utf-8')
        code=child.wait()
    with log_path.open('a',encoding='utf-8') as f:
        f.write(json.dumps({'variant':variant,'returncode':code,'completed':time.strftime('%Y-%m-%dT%H:%M:%S')})+'\n')
    print('EXIT',variant,code,flush=True)
    if code!=0:
        print(console.read_text(encoding='utf-8')[-5000:],flush=True)
        raise SystemExit(code)
    result=json.loads((directory/'summary.json').read_text())
    if not result['diagnostic_completed']:
        raise SystemExit('Safety stop before matched budget: stop comparison and investigate')
    print('RESULT',variant,result['final'],'validation32',result['validation_32'],flush=True)
print('ALL THREE VARIANTS FINISHED; no further training',flush=True)
