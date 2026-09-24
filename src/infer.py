import json, math
from pathlib import Path
import torch
from .model import EXLLM,EXLLMConfig
from .tokenizer import HybridTokenizer,UTF8State,normalize_text

def load_model(ckpt='weights/EXLLM-v1.0-train.pt'):
    tok=HybridTokenizer.load('tokenizer.json'); c=torch.load(ckpt,map_location='cpu'); cfg=EXLLMConfig(**c['config']); m=EXLLM(cfg); m.load_state_dict(c['model']); m.eval(); return m,tok,c

def bad_text(s):
    if not s or '\ufffd' in s or '\x00' in s: return True
    if any((ord(c)<32 and c not in '\n\t') for c in s): return True
    # obvious repetition collapse
    if len(s)>=12:
        for n in range(1,7):
            unit=s[-n:]
            if len(unit)*4<=len(s) and s.endswith(unit*4): return True
    return False

@torch.inference_mode()
def generate(m,tok,prompt,max_new=64,temperature=0.0,top_k=12,confidence_fallback=True):
    ids=tok.encode_user(prompt)
    # Reserve output space. Left-trim user content but preserve role tokens.
    reserve=min(max_new, m.cfg.max_seq_len//2)
    if len(ids)>m.cfg.max_seq_len-reserve:
        keep=m.cfg.max_seq_len-reserve-3
        body=ids[2:-1][-max(1,keep):]; ids=[tok.BOS,tok.USER,*body,tok.ASSIST]
    out=[]; state=UTF8State(); lps=[]
    for _ in range(min(max_new,m.cfg.max_seq_len-len(ids))):
        x=torch.tensor([ids+out],dtype=torch.long); logits=m(x)[0,-1].clone()
        # Structural specials are never generated inside assistant text.
        logits[tok.PAD]=logits[tok.BOS]=logits[tok.USER]=logits[tok.ASSIST]=-1e30
        # UTF-8 constrained token mask.
        valid=torch.zeros_like(logits,dtype=torch.bool)
        if state.complete:
            valid[tok.EOS]=True
            for t in range(256):
                if state.accepts_byte(t): valid[t]=True
            if tok.chars: valid[256:256+len(tok.chars)]=True
        else:
            for t in range(256):
                if state.accepts_byte(t): valid[t]=True
        logits[~valid]=-1e30
        lp=torch.log_softmax(logits,dim=-1)
        if temperature and temperature>0:
            z=logits/temperature
            if top_k and top_k<z.numel():
                v,ix=torch.topk(z,top_k); probs=torch.softmax(v,dim=-1); t=int(ix[torch.multinomial(probs,1)].item())
            else: t=int(torch.multinomial(torch.softmax(z,dim=-1),1).item())
        else: t=int(torch.argmax(logits).item())
        lps.append(float(lp[t]))
        if t==tok.EOS: break
        if t<256: state.push(t)
        elif not state.complete: continue
        out.append(t)
    # If max length stopped in a partial byte sequence, drop trailing raw bytes until valid.
    while out:
        try: text=tok.decode(out); break
        except UnicodeDecodeError: out.pop()
    else: text=''
    text=text.strip()
    avg_lp=sum(lps)/max(1,len(lps))
    if confidence_fallback and (bad_text(text) or avg_lp < -2.8):
        text='うまく答えられませんでした。質問を短く言い換えてください。'
    return text,avg_lp

if __name__=='__main__':
    import sys; torch.set_num_threads(2); m,tok,c=load_model(); q=' '.join(sys.argv[1:]) or 'こんにちは'; print(generate(m,tok,q)[0])
