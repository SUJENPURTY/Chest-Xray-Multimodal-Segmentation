"""Single-runtime launcher; one deliberate verified process-restart test only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
if (HERE/'summary.json').exists(): raise SystemExit('Completed pilot preserved')
for attempt in range(2):
    resume=args.resume or attempt==1
    stamp=time.strftime('%Y%m%d_%H%M%S')
    command=[sys.executable,'-B','-u',str(HERE/'run_pilot.py'),'--resume' if resume else '--verify-restart']
    with (HERE/f'console_{stamp}.log').open('x',encoding='utf-8') as log:
        started=time.perf_counter()
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000)
        (HERE/'owned_process.json').write_text(json.dumps({'child_pid':child.pid,'command':command,'started':stamp},indent=2))
        print('STARTED',child.pid,'resume',resume,'console',log.name,flush=True)
        code=child.wait()
    with (HERE/'controller_log.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'returncode':code,'resume':resume,'started':stamp,'wall_seconds':time.perf_counter()-started,
            'console':str(log.name),'ended':time.strftime('%Y-%m-%dT%H:%M:%S')})+'\n')
    print('RUN EXIT',code,flush=True)
    if code==75 and not resume:
        print('Controlled checkpoint interruption succeeded; launching verified resume',flush=True)
        continue
    if code: raise SystemExit(code)
    assert (HERE/'summary.json').exists()
    print('Two-epoch pilot finished; no extension launched',flush=True)
    break
