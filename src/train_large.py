import argparse,json,math,random,time
from pathlib import Path
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from .model import EXLLM,EXLLMConfig
from .tokenizer import HybridTokenizer
from .train import J,collate,ev
from .loader import load_release_model

def transplant(dst,src):
    od=src.cfg.d_model; nd=dst.cfg.d_model; of=src.cfg.d_ff
    with torch.no_grad():
        dst.tok.weight.zero_(); dst.pos.weight.zero_()
        dst.tok.weight[:,:od].copy_(src.tok.weight)
        dst.pos.weight[:,:od].copy_(src.pos.weight)
        for i,(db,sb) in enumerate(zip(dst.blocks,src.blocks)):
            db.norm1.weight.fill_(1);db.norm2.weight.fill_(1)
            db.norm1.weight[:od].copy_(sb.norm1.weight)
            db.norm2.weight[:od].copy_(sb.norm2.weight)
            db.qkv.weight.zero_();db.proj.weight.zero_();db.fc1.weight.zero_();db.fc2.weight.zero_()
            for q in range(3): db.qkv.weight[q*nd:q*nd+od,:od].copy_(sb.qkv.weight[q*od:(q+1)*od,:])
            db.proj.weight[:od,:od].copy_(sb.proj.weight)
            db.fc1.weight[:of,:od].copy_(sb.fc1.weight)
            db.fc2.weight[:od,:of].copy_(sb.fc2.weight)
        dst.norm.weight.fill_(1);dst.norm.weight[:od].copy_(src.norm.weight)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=10);ap.add_argument('--batch',type=int,default=64);ap.add_argument('--lr',type=float,default=5e-4);ap.add_argument('--threads',type=int,default=8);ap.add_argument('--data',default='data/mix.jsonl');ap.add_argument('--valid',default='data/valid.jsonl');ap.add_argument('--out',default='weights/EXLLM-v1.1-large.pt');ap.add_argument('--d-model',type=int,default=288);ap.add_argument('--layers',type=int,default=6);ap.add_argument('--heads',type=int,default=9);ap.add_argument('--d-ff',type=int,default=896);a=ap.parse_args()
    torch.manual_seed(20260923);random.seed(20260923);torch.set_num_threads(a.threads)
    tok=HybridTokenizer.load('tokenizer.json');cfg=EXLLMConfig(vocab_size=tok.vocab_size,d_model=a.d_model,n_layers=a.layers,n_heads=a.heads,d_ff=a.d_ff,max_seq_len=128)
    teacher,_=load_release_model('.');m=EXLLM(cfg);transplant(m,teacher);del teacher
    tr=J(a.data,tok,cfg.max_seq_len);va=J(a.valid,tok,cfg.max_seq_len)
    tl=DataLoader(tr,batch_size=a.batch,shuffle=True,collate_fn=lambda b:collate(b,tok.PAD));vl=DataLoader(va,batch_size=a.batch,shuffle=False,collate_fn=lambda b:collate(b,tok.PAD))
    opt=torch.optim.AdamW(m.parameters(),lr=a.lr,betas=(.9,.95),weight_decay=.03);total=a.epochs*len(tl);warm=max(20,int(total*.04));step=0;best=1e9;hist=[];t0=time.time()
    for ep in range(1,a.epochs+1):
        m.train();ss=nn=0
        for x,y in tl:
            step+=1;q=0 if step<=warm else (step-warm)/max(1,total-warm);scale=step/warm if step<=warm else .1+.9*.5*(1+math.cos(math.pi*q))
            for g in opt.param_groups:g['lr']=a.lr*scale
            opt.zero_grad(set_to_none=True);z=m(x);loss=F.cross_entropy(z.reshape(-1,z.size(-1)),y.reshape(-1),ignore_index=-100);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step();n=(y!=-100).sum().item();ss+=loss.item()*n;nn+=n
            if step%100==0:print(json.dumps({'step':step,'loss':float(loss),'sec':time.time()-t0}),flush=True)
        v=ev(m,vl);rec={'epoch':ep,'train_loss':ss/max(nn,1),'valid_loss':v,'sec':time.time()-t0};hist.append(rec);print(json.dumps(rec),flush=True)
        if v<best:
            best=v;Path(a.out).parent.mkdir(parents=True,exist_ok=True);torch.save({'model':m.state_dict(),'config':cfg.__dict__,'meta':{'name':'EXLLM','developer':'ToTo','version':'1.1.0-5m','params':m.num_parameters(),'tokenizer':'hybrid-char-byte-fallback','data':a.data,'initialization':'v1.0 transplant'},'history':hist},a.out)
    print(json.dumps({'best':best,'params':m.num_parameters(),'steps':step,'sec':time.time()-t0,'out':a.out}))
if __name__=='__main__':main()
