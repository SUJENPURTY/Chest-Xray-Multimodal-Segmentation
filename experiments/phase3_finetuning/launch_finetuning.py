"""Explicit execution gate, isolated logging, and no automatic retries."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true', help='Use only after user approval')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print('Prepared only. No training launched. Approval is required before --execute.')
        return
    if (here/'summary.json').exists():
        raise SystemExit('Completed Phase 3 experiment preserved; refusing rerun')
    if not (here/'protected_sources.json').exists():
        raise SystemExit('Run prepare.py first')
    if args.resume and not (here/'checkpoints/latest.json').exists():
        raise SystemExit('No committed Phase 3 checkpoint available')
    if (here/'training_microsteps.csv').exists() and not args.resume:
        raise SystemExit('Existing history requires explicit --resume')
    if (here/'STOP_REQUESTED').exists():
        raise SystemExit('STOP_REQUESTED exists; remove it only when a resume is authorized')
    session = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    command = [sys.executable, '-B', '-u', str(here/'run_finetuning.py'), '--execute']
    if args.resume:
        command.append('--resume')
    with (here/f'console_{session}.log').open('x', encoding='utf-8') as log:
        started = time.perf_counter()
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        with (here/f'owned_process_{session}.json').open('x', encoding='utf-8') as handle:
            json.dump({'child_pid':child.pid,'command':command,'session':session},handle,indent=2)
        print('Started Phase 3 PID', child.pid, 'log', log.name, flush=True)
        code = child.wait()
    with (here/'controller_log.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps({'returncode':code,'resume':args.resume,'session':session,
                                 'wall_seconds':time.perf_counter()-started,'console':log.name})+'\n')
    if code:
        raise SystemExit(code)
    print('Two additional epochs completed. No extension launched.', flush=True)


if __name__ == '__main__':
    main()
