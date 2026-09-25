import argparse,random,json,time,sys,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.loader import load_release_model
from src.infer import bad_text
from src.runtime import answer

def valid(s):
    try:return bool(s) and s.encode('utf-8').decode('utf-8')==s and '\ufffd' not in s and '\x00' not in s and not bad_text(s) and len(s)<=160
    except:return False

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=200); ap.add_argument('--seed',type=int,default=1001); ap.add_argument('--out',default='eval/fuzz_result.json'); a=ap.parse_args()
    torch.set_num_threads(8); m,tok=load_release_model(ROOT); rng=random.Random(a.seed); alpha=list('あいうえおかきくけこカキクケコ漢字猫犬海山川ABCxyz0123456789!?%#@ ')+['🙂','🚀','☕','髙','﨑','𠮷','←','→','★','é','Ω','한','中','ß','ø','Ж']; fail=[]; t0=time.time()
    for i in range(a.n):
        q=''.join(rng.choice(alpha) for _ in range(rng.randint(1,36))); out=answer(m,tok,q,max_new=48,temperature=0.0)
        if not valid(out):fail.append({'i':i,'prompt':q,'answer':out})
    rep={'model':'EXLLM-v1.1-5m-release3.pt','n':a.n,'seed':a.seed,'failures':len(fail),'elapsed_sec':time.time()-t0,'failure_cases':fail}; (ROOT/a.out).write_text(json.dumps(rep,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({k:rep[k] for k in ['n','seed','failures','elapsed_sec']})); raise SystemExit(1 if fail else 0)
if __name__=='__main__':main()
