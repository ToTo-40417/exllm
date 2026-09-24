import argparse, json, math, random, time
from pathlib import Path
import torch, torch.nn.functional as F
from .model import EXLLM, EXLLMConfig
from .tokenizer import HybridTokenizer
from .train import J, collate

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--in-ckpt',default='weights/EXLLM-v1.0-train.pt')
    ap.add_argument('--out-ckpt',default='weights/EXLLM-v1.0-recovery.pt')
    ap.add_argument('--data',default='data/recovery.jsonl')
    ap.add_argument('--steps',type=int,default=400)
    ap.add_argument('--batch',type=int,default=32)
    ap.add_argument('--lr',type=float,default=1.5e-4)
    ap.add_argument('--threads',type=int,default=8)
    a=ap.parse_args()
    torch.manual_seed(20260924); random.seed(20260924); torch.set_num_threads(a.threads)
    tok=HybridTokenizer.load('tokenizer.json')
    c=torch.load(a.in_ckpt,map_location='cpu',weights_only=False)
    cfg=EXLLMConfig(**c['config']); m=EXLLM(cfg); m.load_state_dict(c['model']); m.train()
    ds=J(a.data,tok,cfg.max_seq_len); rng=random.Random(20260924+c.get('global_step',0))
    opt=torch.optim.AdamW(m.parameters(),lr=a.lr,betas=(.9,.95),weight_decay=.02)
    losses=[]; t0=time.time()
    for step in range(a.steps):
        ix=[rng.randrange(len(ds)) for _ in range(a.batch)]
        x,y=collate([ds[i] for i in ix],tok.PAD)
        frac=step/max(1,a.steps-1); lr=a.lr*(0.15+0.85*0.5*(1+math.cos(math.pi*frac)))
        for g in opt.param_groups: g['lr']=lr
        opt.zero_grad(set_to_none=True); z=m(x)
        loss=F.cross_entropy(z.reshape(-1,z.size(-1)),y.reshape(-1),ignore_index=-100)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); losses.append(float(loss))
    gs=c.get('global_step',0)+a.steps
    meta=dict(c.get('meta',{})); meta.update({'name':'EXLLM','developer':'ToTo','version':'1.0.0','params':m.num_parameters(),'tokenizer':'hybrid-char-byte-fallback','data':'project-generated-only'})
    Path(a.out_ckpt).parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':m.state_dict(),'global_step':gs,'config':cfg.__dict__,'meta':meta,'finetune':{'source':a.in_ckpt,'data':a.data,'steps':a.steps,'lr':a.lr,'loss_first':sum(losses[:50])/min(50,len(losses)),'loss_last':sum(losses[-50:])/min(50,len(losses))}},a.out_ckpt)
    print(json.dumps({'out':a.out_ckpt,'global_step':gs,'loss_first':sum(losses[:50])/min(50,len(losses)),'loss_last':sum(losses[-50:])/min(50,len(losses)),'sec':time.time()-t0}))
if __name__=='__main__': main()
