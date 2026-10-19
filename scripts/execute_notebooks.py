"""Execute the numbered notebooks sequentially and persist each completed cell."""
import json
import os
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
if os.name == 'nt':
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF',
                          'max_split_size_mb:128,garbage_collection_threshold:0.8')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from src.config import from_environment

cfg = from_environment()
os.environ['DATA_ROOT'] = str(cfg.data_root.resolve())
os.environ['OUTPUT_DIR'] = str(cfg.output_dir.resolve())
report_dir = cfg.output_dir / 'execution'
report_dir.mkdir(parents=True, exist_ok=True)
paths = sorted((p for p in (ROOT / 'notebooks').rglob('*.ipynb')
                if '.ipynb_checkpoints' not in p.parts),
               key=lambda p: [int(s) if s.isdigit() else s.lower()
                              for s in re.split(r'(\d+)', str(p.relative_to(ROOT)))])
print('EXECUTION ORDER:', *[str(p.relative_to(ROOT)) for p in paths], sep='\n', flush=True)
statuses = {}
for number, path in enumerate(paths, 1):
    print(f'[{number}/{len(paths)}] Running {path.name}', flush=True)
    notebook = nbformat.read(path, as_version=4)
    # Stream long-running cell output into the log as it arrives.
    class StreamingClient(NotebookClient):
        def process_message(self, msg, cell, cell_index):
            result = super().process_message(msg, cell, cell_index)
            if msg['msg_type'] == 'stream':
                print(msg['content']['text'], end='', flush=True)
                nbformat.write(self.nb, path)
            return result

    manager = KernelManager(kernel_name='xray-project')
    client = StreamingClient(notebook, km=manager, timeout=None,
                            resources={'metadata': {'path': str(ROOT)}})
    try:
        with client.setup_kernel():
            for index, cell in enumerate(notebook.cells):
                print(f'CELL {index + 1}/{len(notebook.cells)} {cell.cell_type}', flush=True)
                client.execute_cell(cell, index)
                nbformat.write(notebook, path)
        statuses[path.name] = 'SUCCESS'
    except Exception:
        nbformat.write(notebook, path)
        statuses[path.name] = 'FAILED'
        (report_dir / 'status.json').write_text(json.dumps(statuses, indent=2))
        raise
    (report_dir / 'status.json').write_text(json.dumps(statuses, indent=2))
    print(f'[{number}/{len(paths)}] SUCCESS {path.name}', flush=True)
