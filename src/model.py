from dataclasses import dataclass, asdict
import torch, torch.nn as nn, torch.nn.functional as F

@dataclass
class EXLLMConfig:
    vocab_size:int=1029
    max_seq_len:int=128
    d_model:int=160
    n_layers:int=4
    n_heads:int=5
    d_ff:int=512
    rms_eps:float=1e-5

class RMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-5): super().__init__(); self.weight=nn.Parameter(torch.ones(dim)); self.eps=eps
    def forward(self,x): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)*self.weight
class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__(); d=cfg.d_model; self.n_heads=cfg.n_heads; self.head_dim=d//cfg.n_heads
        self.norm1=RMSNorm(d,cfg.rms_eps); self.qkv=nn.Linear(d,3*d,bias=False); self.proj=nn.Linear(d,d,bias=False)
        self.norm2=RMSNorm(d,cfg.rms_eps); self.fc1=nn.Linear(d,cfg.d_ff,bias=False); self.fc2=nn.Linear(cfg.d_ff,d,bias=False)
    def forward(self,x):
        b,t,d=x.shape; h=self.norm1(x); qkv=self.qkv(h).view(b,t,3,self.n_heads,self.head_dim).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]; a=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        x=x+self.proj(a.transpose(1,2).contiguous().view(b,t,d)); h=self.norm2(x); return x+self.fc2(F.relu(self.fc1(h)))
class EXLLM(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg; self.tok=nn.Embedding(cfg.vocab_size,cfg.d_model); self.pos=nn.Embedding(cfg.max_seq_len,cfg.d_model)
        self.blocks=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)]); self.norm=RMSNorm(cfg.d_model,cfg.rms_eps); self.lm_head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False); self.lm_head.weight=self.tok.weight; self.apply(self._init)
    def _init(self,m):
        if isinstance(m,(nn.Linear,nn.Embedding)): nn.init.normal_(m.weight,0.0,0.02)
    def forward(self,idx):
        b,t=idx.shape
        if t>self.cfg.max_seq_len: raise ValueError('context too long')
        p=torch.arange(t,device=idx.device); x=self.tok(idx)+self.pos(p)[None,:,:]
        for block in self.blocks: x=block(x)
        return self.lm_head(self.norm(x))
    def num_parameters(self): return sum(p.numel() for p in self.parameters())
