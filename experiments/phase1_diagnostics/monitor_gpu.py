"""Record whole-device usage (includes desktop/apps), independently of PyTorch."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

folder = Path(__file__).resolve().parent
rows = []
with (folder / 'gpu_monitor.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp_utc', 'device_used_mib', 'device_total_mib', 'gpu_util_percent', 'temperature_c', 'power_w'])
    while True:
        try:
            command = ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw',
                       '--format=csv,noheader,nounits']
            completed = subprocess.run(command, capture_output=True, text=True, timeout=15)
            completed.check_returncode()
            values = [float(x.strip()) for x in completed.stdout.strip().splitlines()[0].split(',')]
            row = [datetime.now(timezone.utc).isoformat(), *values]
            rows.append(row)
            writer.writerow(row)
            f.flush()
        except Exception as exc:
            print(type(exc).__name__, str(exc), flush=True)
        if (folder / 'preservation_verification.json').exists():
            break
        time.sleep(5)
result = {'samples': len(rows), 'scope': 'whole GPU including desktop and other applications',
          'sampling_seconds': 5, 'maximum_device_used_mib': max(r[1] for r in rows) if rows else None,
          'maximum_temperature_c': max(r[4] for r in rows) if rows else None,
          'maximum_power_w': max(r[5] for r in rows) if rows else None}
(folder / 'gpu_monitor_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
