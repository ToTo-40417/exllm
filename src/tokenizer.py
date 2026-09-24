import json, unicodedata
from collections import Counter
from pathlib import Path

BYTE_BASE=0; BYTE_COUNT=256

def normalize_text(s:str)->str:
    s=unicodedata.normalize('NFC',s)
    return ''.join((' ' if (ord(c)<32 and c not in '\n\t') else c) for c in s).strip()

class HybridTokenizer:
    def __init__(self, chars):
        self.chars=list(chars); self.char_to_id={c:256+i for i,c in enumerate(self.chars)}
        s=256+len(self.chars)
        self.PAD=s; self.BOS=s+1; self.USER=s+2; self.ASSIST=s+3; self.EOS=s+4; self.vocab_size=s+5
    @classmethod
    def build_from_jsonl(cls, paths, max_chars=768):
        cnt=Counter()
        for path in paths:
            for line in open(path,encoding='utf-8'):
                r=json.loads(line)
                cnt.update(normalize_text(r['prompt'])); cnt.update(normalize_text(r['answer']))
        # ASCII stays as byte tokens. Frequent non-ASCII chars become atomic tokens.
        chars=[c for c,_ in cnt.most_common() if ord(c)>=128][:max_chars]
        return cls(chars)
    def save(self,path):
        Path(path).write_text(json.dumps({'type':'hybrid-char-byte-fallback','chars':self.chars,'normalization':'NFC','special':{'PAD':self.PAD,'BOS':self.BOS,'USER':self.USER,'ASSIST':self.ASSIST,'EOS':self.EOS}},ensure_ascii=False,indent=2),encoding='utf-8')
    @classmethod
    def load(cls,path): return cls(json.loads(Path(path).read_text(encoding='utf-8'))['chars'])
    def encode_text(self,s):
        out=[]
        for c in normalize_text(s):
            if c in self.char_to_id: out.append(self.char_to_id[c])
            else: out.extend(c.encode('utf-8','strict'))
        return out
    def encode_user(self,s): return [self.BOS,self.USER,*self.encode_text(s),self.ASSIST]
    def encode_example(self,prompt,answer,max_seq_len=128):
        p=self.encode_text(prompt); a=self.encode_text(answer)
        seq=[self.BOS,self.USER,*p,self.ASSIST,*a,self.EOS]
        if len(seq)>max_seq_len:
            excess=len(seq)-max_seq_len; p=p[min(excess,len(p)):]
            seq=[self.BOS,self.USER,*p,self.ASSIST,*a,self.EOS]
            if len(seq)>max_seq_len:
                a=a[:max(1,max_seq_len-(4+len(p)))]
                seq=[self.BOS,self.USER,*p,self.ASSIST,*a,self.EOS]
        return seq
    def token_bytes(self,tok):
        if 0<=tok<256: return bytes([tok])
        i=tok-256
        if 0<=i<len(self.chars): return self.chars[i].encode('utf-8')
        return b''
    def decode(self,toks):
        b=bytearray()
        for t in toks:
            if t in (self.PAD,self.BOS,self.USER,self.ASSIST,self.EOS): continue
            b.extend(self.token_bytes(t))
        return bytes(b).decode('utf-8','strict')

class UTF8State:
    __slots__=('need','lo','hi')
    def __init__(self): self.need=0; self.lo=0x80; self.hi=0xBF
    def accepts_byte(self,b):
        if self.need: return self.lo<=b<=self.hi
        return b in (9,10,13) or 0x20<=b<=0x7E or 0xC2<=b<=0xF4
    def push(self,b):
        if not self.accepts_byte(b): return False
        if self.need:
            self.need-=1; self.lo=0x80; self.hi=0xBF; return True
        if b<=0x7F: return True
        if 0xC2<=b<=0xDF: self.need=1
        elif b==0xE0: self.need=2; self.lo=0xA0
        elif 0xE1<=b<=0xEC or 0xEE<=b<=0xEF: self.need=2
        elif b==0xED: self.need=2; self.hi=0x9F
        elif b==0xF0: self.need=3; self.lo=0x90
        elif 0xF1<=b<=0xF3: self.need=3
        elif b==0xF4: self.need=3; self.hi=0x8F
        return True
    @property
    def complete(self): return self.need==0
