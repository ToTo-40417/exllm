import argparse,hashlib,json,struct,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.loader import load_release_model
from src.model import EXLLM,EXLLMConfig
MAGIC=b'EXLLM8\0\0'
def quant_row(t):
    a=t.detach().float().cpu().numpy(); mx=np.max(np.abs(a),axis=1); sc=np.where(mx>0,mx/127.0,1.0).astype(np.float32); q=np.clip(np.rint(a/sc[:,None]),-127,127).astype(np.int8); return q,sc
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='weights/EXLLM-v1.1-5m-int8-reexport.bin'); ap.add_argument('--checkpoint',default=''); a=ap.parse_args()
    if a.checkpoint:
        c=torch.load(ROOT/a.checkpoint,map_location='cpu',weights_only=False);m=EXLLM(EXLLMConfig(**c['config']));m.load_state_dict(c['model']);m.eval()
    else:m,tok=load_release_model(ROOT)
    recs=[]
    for name,t in m.state_dict().items():
        if name=='lm_head.weight':continue
        if name.endswith('.weight') and 'norm' in name:
            x=t.detach().cpu().numpy().astype(np.float16); recs.append((name,2,list(x.shape),np.array([],dtype=np.float32),x.tobytes()))
        elif t.ndim==2:
            q,sc=quant_row(t); recs.append((name,1,list(q.shape),sc,q.tobytes()))
        else:
            x=t.detach().cpu().numpy().astype(np.float16); recs.append((name,2,list(x.shape),np.array([],dtype=np.float32),x.tobytes()))
    out=ROOT/a.out; out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('wb') as f:
        f.write(MAGIC);f.write(struct.pack('<II',1,len(recs)))
        for name,qt,dims,sc,data in recs:
            nb=name.encode();f.write(struct.pack('<HBB',len(nb),len(dims),qt));f.write(nb)
            for d in dims:f.write(struct.pack('<I',d))
            f.write(struct.pack('<II',len(sc),len(data)));f.write(sc.astype('<f4').tobytes());f.write(data)
    sha=hashlib.sha256(out.read_bytes()).hexdigest()
    if a.checkpoint:
        man={'format':'EXLLM8','format_version':1,'model':'EXLLM','model_version':'1.1.0-5m','architecture':'decoder-only Transformer, RMSNorm, ReLU, tied token/lm-head embedding','config':m.cfg.__dict__,'unique_parameters':m.num_parameters(),'tokenizer':'hybrid-char-byte-fallback','tokenizer_vocab':m.cfg.vocab_size,'quantization':{'matrix_weights':'symmetric int8 per output row','matrix_scale':'float32 per row','rmsnorm_weights':'float16'},'aliases':{'lm_head.weight':'tok.weight'},'tensor_count':len(recs),'file':str(out.relative_to(ROOT)),'bytes':out.stat().st_size,'sha256':sha}
        out.with_suffix('.manifest.json').write_text(json.dumps(man,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'file':str(out.relative_to(ROOT)),'bytes':out.stat().st_size,'sha256':sha}))
if __name__=='__main__':main()
