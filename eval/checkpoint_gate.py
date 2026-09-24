import argparse,json,random,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.model import EXLLM,EXLLMConfig
from src.tokenizer import HybridTokenizer
from src.runtime import answer
from src.infer import bad_text
from eval.release_gate import CASES,valid_text

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--fuzz',type=int,default=80);ap.add_argument('--out',required=True);a=ap.parse_args();t0=time.time();torch.set_num_threads(4)
    c=torch.load(ROOT/a.checkpoint,map_location='cpu',weights_only=False);m=EXLLM(EXLLMConfig(**c['config']));m.load_state_dict(c['model']);m.eval();tok=HybridTokenizer.load(ROOT/'tokenizer.json')
    sem=[];okn=0
    for name,q,must,must_not in CASES:
        z=answer(m,tok,q,max_new=64,temperature=0.0);ok=valid_text(z) and all(x in z for x in must) and all(x not in z for x in must_not);sem.append({'name':name,'prompt':q,'answer':z,'pass':ok});okn+=int(ok)
    rng=random.Random(20260927);alpha=list('あいうえおかきくけこカキクケコ漢字猫犬海山川ABCxyz0123456789!?%#@ ')+['🙂','🚀','☕','髙','﨑','𠮷','←','→','★','é','Ω','한','中'];ff=[]
    for _ in range(a.fuzz):
        q=''.join(rng.choice(alpha) for _ in range(rng.randint(1,28)));z=answer(m,tok,q,max_new=48,temperature=0.0)
        if not valid_text(z):ff.append({'prompt':q,'answer':z})
    rep={'checkpoint':a.checkpoint,'params':m.num_parameters(),'semantic':{'pass':okn,'total':len(CASES)},'fuzz':{'pass':a.fuzz-len(ff),'total':a.fuzz},'release_pass':okn==len(CASES) and not ff,'elapsed_sec':time.time()-t0,'semantic_cases':sem,'fuzz_failures':ff}
    (ROOT/a.out).write_text(json.dumps(rep,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:rep[k] for k in ['params','semantic','fuzz','release_pass','elapsed_sec']},ensure_ascii=False));raise SystemExit(0 if rep['release_pass'] else 2)
if __name__=='__main__':main()
