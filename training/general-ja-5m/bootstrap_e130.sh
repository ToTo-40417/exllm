#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
venv=${EXLLM_TRAINING_VENV:-"$root/.venv-training"}

python3 -m venv --system-site-packages "$venv"
"$venv/bin/python" -m pip install --upgrade pip
"$venv/bin/python" -m pip install -r "$root/requirements.txt" --no-deps
"$venv/bin/python" "$root/tools/audit_training_data.py" \
  --manifest "$root/training/general-ja-5m/experiment.json" \
  --output "$root/training/general-ja-5m/work/data-audit.json"

"$venv/bin/python" - "$root/training/general-ja-5m/work/e130-environment.json" <<'PY'
import json
import platform
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
}
encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
Path(sys.argv[1]).write_text(encoded, encoding="utf-8")
print(encoded, end="")
PY
