import argparse,json,random,time,math
from pathlib import Path
import torch, torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from .model import EXLLM,EXLLMConfig
from .tokenizer import HybridTokenizer
class J(Dataset):
    def __init__(self,path,tok,maxlen):
        self.rows=[]
        for line in open(path,encoding='utf-8'):
            r=json.loads(line); seq=tok.encode_example(r['prompt'],r['answer'],maxlen); k=seq.index(tok.ASSIST); x=seq[:-1]; y=seq[1:]; y=[-100 if i<k else z for i,z in enumerate(y)]; self.rows.append((x,y))
    def __len__(self): return len(self.rows)
    def __getitem__(self,i): return self.rows[i]
def collate(batch,pad):
    mx=max(len(x) for x,_ in batch); X=torch.full((len(batch),mx),pad,dtype=torch.long); Y=torch.full((len(batch),mx),-100,dtype=torch.long)
    for i,(x,y) in enumerate(batch): X[i,:len(x)]=torch.tensor(x); Y[i,:len(y)]=torch.tensor(y)
    return X,Y
@torch.no_grad()
def ev(m,l):
    m.eval(); s=n=0
    for x,y in l:
        z=m(x); s+=F.cross_entropy(z.reshape(-1,z.size(-1)),y.reshape(-1),ignore_index=-100,reduction='sum').item(); n+=(y!=-100).sum().item()
    return s/max(n,1)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=8); ap.add_argument('--batch',type=int,default=96); ap.add_argument('--lr',type=float,default=9e-4); ap.add_argument('--threads',type=int,default=8); ap.add_argument('--resume',default=''); ap.add_argument('--out',default='weights/EXLLM-v1.0-fp32.pt'); a=ap.parse_args()
    torch.manual_seed(20260922); random.seed(20260922); torch.set_num_threads(a.threads)
    tok=HybridTokenizer.build_from_jsonl(['data/train.jsonl','data/valid.jsonl'],768); tok.save('tokenizer.json')
    cfg=EXLLMConfig(vocab_size=tok.vocab_size); m=EXLLM(cfg)
    if a.resume: m.load_state_dict(torch.load(a.resume,map_location='cpu')['model'])
    tr=J('data/train.jsonl',tok,cfg.max_seq_len); va=J('data/valid.jsonl',tok,cfg.max_seq_len)
    tl=DataLoader(tr,batch_size=a.batch,shuffle=True,collate_fn=lambda b:collate(b,tok.PAD)); vl=DataLoader(va,batch_size=a.batch,shuffle=False,collate_fn=lambda b:collate(b,tok.PAD))
    opt=torch.optim.AdamW(m.parameters(),lr=a.lr,betas=(.9,.95),weight_decay=.04); total=a.epochs*len(tl); warm=max(10,int(total*.03)); step=0; best=1e9; hist=[]; t0=time.time()
    for ep in range(1,a.epochs+1):
        m.train(); ss=nn=0
        for x,y in tl:
            step+=1; q=0 if step<=warm else (step-warm)/max(1,total-warm); scale=step/warm if step<=warm else .12+.88*.5*(1+math.cos(math.pi*q))
            for g in opt.param_groups:g['lr']=a.lr*scale
            opt.zero_grad(set_to_none=True); z=m(x); loss=F.cross_entropy(z.reshape(-1,z.size(-1)),y.reshape(-1),ignore_index=-100); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); n=(y!=-100).sum().item(); ss+=loss.item()*n; nn+=n
            if step%50==0: print(json.dumps({'step':step,'loss':float(loss),'sec':time.time()-t0}),flush=True)
        v=ev(m,vl); rec={'epoch':ep,'train_loss':ss/nn,'valid_loss':v,'sec':time.time()-t0}; hist.append(rec); print(json.dumps(rec),flush=True)
        if v<best:
            best=v; Path(a.out).parent.mkdir(exist_ok=True); torch.save({'model':m.state_dict(),'config':cfg.__dict__,'meta':{'name':'EXLLM','developer':'ToTo','version':'1.0.0','params':m.num_parameters(),'tokenizer':'hybrid-char-byte-fallback','data':'project-generated-only'},'history':hist},a.out)
    Path('eval/train_history.json').write_text(json.dumps(hist,ensure_ascii=False,indent=2),encoding='utf-8'); print('best',best,'params',m.num_parameters(),'vocab',tok.vocab_size)
if __name__=='__main__':main()
