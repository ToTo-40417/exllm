import argparse,json,struct,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.model import EXLLM,EXLLMConfig
MAGIC=b'EXLLM8\0\0'
def load_bin(path,manifest_path):
    man=json.loads(Path(manifest_path).read_text(encoding='utf-8'));m=EXLLM(EXLLMConfig(**man['config']));vals={}
    with open(path,'rb') as f:
        if f.read(8)!=MAGIC:raise ValueError('bad magic')
        ver,n=struct.unpack('<II',f.read(8));
        if ver!=1:raise ValueError('unsupported format')
        for _ in range(n):
            nl,nd,qt=struct.unpack('<HBB',f.read(4));name=f.read(nl).decode();dims=[struct.unpack('<I',f.read(4))[0] for _ in range(nd)];ns,nb=struct.unpack('<II',f.read(8));sc=np.frombuffer(f.read(ns*4),dtype='<f4').copy() if ns else np.array([],dtype=np.float32);raw=f.read(nb)
            if qt==1:a=np.frombuffer(raw,dtype=np.int8).reshape(dims).astype(np.float32)*sc[:,None]
            elif qt==2:a=np.frombuffer(raw,dtype='<f2').reshape(dims).astype(np.float32)
            else:raise ValueError(qt)
            vals[name]=torch.from_numpy(a.copy())
    sd=m.state_dict();aliases=man.get('aliases',{})
    for k in sd:
        src=aliases.get(k,k)
        if src in vals:sd[k]=vals[src]
    m.load_state_dict(sd);m.eval();return m,man
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bin',default='weights/EXLLM-v1.1-5m-int8.bin');ap.add_argument('--manifest',default='weights/EXLLM-v1.1-5m-int8.manifest.json');ap.add_argument('--out',default='weights/EXLLM-v1.1-5m-int8-dequant-test.pt');a=ap.parse_args();m,man=load_bin(ROOT/a.bin,ROOT/a.manifest);torch.save({'model':m.state_dict(),'config':m.cfg.__dict__,'meta':{'name':'EXLLM','developer':'ToTo','version':'1.1.0-5m','quantized_source':a.bin}},ROOT/a.out);print(a.out)
if __name__=='__main__':main()
