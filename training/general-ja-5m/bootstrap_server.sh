#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
venv=${EXLLM_SERVER_VENV:-"$root/.venv-server"}

python3 -m venv "$venv"
"$venv/bin/python" -m pip install --upgrade pip
"$venv/bin/python" -m pip install -r "$root/training/general-ja-5m/requirements-server.txt"
mkdir -p "$root/training/general-ja-5m/work"
"$venv/bin/python" - "$root/training/general-ja-5m/work/server-environment.json" <<'PY'
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy
import safetensors
import torch

report = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "numpy": numpy.__version__,
    "safetensors": safetensors.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    "pip_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
}
encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
Path(sys.argv[1]).write_text(encoded, encoding="utf-8")
print(encoded, end="")
if not report["cuda_available"]:
    raise SystemExit("CUDA is not available in the isolated EXLLM environment")
PY
