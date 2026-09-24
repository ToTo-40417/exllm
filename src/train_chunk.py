import argparse,json,random,math,time
from pathlib import Path
import torch, torch.nn.functional as F
from .model import EXLLM,EXLLMConfig
from .tokenizer import HybridTokenizer
from .train import J,collate

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=300); ap.add_argument('--batch',type=int,default=16); ap.add_argument('--threads',type=int,default=2); ap.add_argument('--target',type=int,default=3000); ap.add_argument('--lr',type=float,default=9e-4); ap.add_argument('--ckpt',default='weights/EXLLM-v1.0-train.pt'); ap.add_argument('--data',default='data/train.jsonl'); a=ap.parse_args()
    torch.set_num_threads(a.threads)
    tok_path=Path('tokenizer.json')
    if tok_path.exists(): tok=HybridTokenizer.load(tok_path)
    else:
        tok=HybridTokenizer.build_from_jsonl(['data/train.jsonl','data/valid.jsonl'],768); tok.save(tok_path)
    cfg=EXLLMConfig(vocab_size=tok.vocab_size); m=EXLLM(cfg); opt=torch.optim.AdamW(m.parameters(),lr=a.lr,betas=(.9,.95),weight_decay=.04); gs=0
    if Path(a.ckpt).exists():
        c=torch.load(a.ckpt,map_location='cpu'); m.load_state_dict(c['model']); opt.load_state_dict(c['optimizer']); gs=c.get('global_step',0)
    ds=J(a.data,tok,cfg.max_seq_len); rng=random.Random(20260922+gs)
    t0=time.time(); losses=[]
    m.train()
    for local in range(a.steps):
        idx=[rng.randrange(len(ds)) for _ in range(a.batch)]; x,y=collate([ds[i] for i in idx],tok.PAD)
        gs+=1
        warm=120
        if gs<=warm: scale=gs/warm
        else:
            q=min(1.0,(gs-warm)/max(1,a.target-warm)); scale=.08+.92*.5*(1+math.cos(math.pi*q))
        for g in opt.param_groups:g['lr']=a.lr*scale
        opt.zero_grad(set_to_none=True); z=m(x); loss=F.cross_entropy(z.reshape(-1,z.size(-1)),y.reshape(-1),ignore_index=-100); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); losses.append(float(loss.detach()))
    Path(a.ckpt).parent.mkdir(exist_ok=True); torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'global_step':gs,'config':cfg.__dict__,'meta':{'name':'EXLLM','developer':'ToTo','version':'1.0.0','params':m.num_parameters(),'tokenizer':'hybrid-char-byte-fallback','data':'project-generated-only'}},a.ckpt)
    print(json.dumps({'global_step':gs,'chunk_loss_first':sum(losses[:50])/min(50,len(losses)),'chunk_loss_last':sum(losses[-50:])/min(50,len(losses)),'sec':time.time()-t0}))
if __name__=='__main__':main()
