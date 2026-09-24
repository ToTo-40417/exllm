import json
from pathlib import Path
import torch
from .model import EXLLM, EXLLMConfig
from .tokenizer import HybridTokenizer

def load_release_model(root=None, weights=None):
    root=Path(root or Path(__file__).resolve().parents[1])
    cfg=EXLLMConfig(**json.loads((root/'config.json').read_text(encoding='utf-8')))
    tok=HybridTokenizer.load(root/'tokenizer.json')
    model=EXLLM(cfg)
    path=Path(weights) if weights else root/'weights'/'EXLLM-v1.1-5m-release3.pt'
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'],strict=True)
    model.eval()
    return model,tok
