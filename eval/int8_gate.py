import json,random,sys,time,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.dequantize_int8 import load_bin
from src.tokenizer import HybridTokenizer
from src.runtime import answer,try_calculate
from src.infer import bad_text
from eval.release_gate import CASES,valid_text

def main(fuzz_n=80,out_name='int8_release_gate_result.json',bin_name='weights/EXLLM-v1.0.0-int8.bin',manifest_name='weights/EXLLM-v1.0.0-int8.manifest.json'):
    torch.set_num_threads(8);t0=time.time();m,man=load_bin(ROOT/bin_name,ROOT/manifest_name);tok=HybridTokenizer.load(ROOT/'tokenizer.json')
    sem=[];okn=0
    for name,q,must,must_not in CASES:
        a=answer(m,tok,q,max_new=64,temperature=0.0);ok=valid_text(a) and all(x in a for x in must) and all(x not in a for x in must_not);sem.append({'name':name,'prompt':q,'answer':a,'pass':ok});okn+=int(ok)
    rng=random.Random(41);calc_fail=[];forms=[lambda a,b:f'{a}+{b}は？',lambda a,b:f'{a}-{b}は？',lambda a,b:f'{a}×{b}は？',lambda a,b:f'{a}かける{b}は？',lambda a,b:f'{a}たす{b}は？',lambda a,b:f'{a}ひく{b}は？']
    for _ in range(300):
        a=rng.randint(-999,999);b=rng.randint(-99,99);q=rng.choice(forms)(a,b)
        exp=a+b if ('+' in q or 'たす' in q) else a*b if ('×' in q or 'かける' in q) else a-b
        got=try_calculate(q)
        if got!=f'{exp}です。':calc_fail.append({'q':q,'got':got,'expected':f'{exp}です。'})
    rng=random.Random(20260927);ff=[];alpha=list('あいうえおかきくけこカキクケコ漢字猫犬海山川ABCxyz0123456789!?%#@ ')+['🙂','🚀','☕','髙','﨑','𠮷','←','→','★','é','Ω','한','中']
    for _ in range(fuzz_n):
        q=''.join(rng.choice(alpha) for _ in range(rng.randint(1,28)));a=answer(m,tok,q,max_new=48,temperature=0.0)
        if not valid_text(a):ff.append({'prompt':q,'answer':a})
    sf=[];samp=[];scases=[('こんにちは',None),('あなたは誰？','EXLLM'),('開発者は誰？','ToTo'),('RAMとは？','RAM'),('日本の首都は？','東京'),('髙﨑🙂という文字は扱える？','UTF-8')]
    for seed in range(8):
        torch.manual_seed(1000+seed)
        for q,must in scases:
            a=answer(m,tok,q,max_new=56,temperature=0.55,top_k=8);ok=valid_text(a) and (must is None or must in a);r={'seed':seed,'prompt':q,'answer':a,'pass':ok};samp.append(r)
            if not ok:sf.append(r)
    rep={'model':bin_name+' (dequantized reference)','semantic':{'pass':okn,'total':len(sem)},'calculator':{'pass':300-len(calc_fail),'total':300},'fuzz':{'pass':fuzz_n-len(ff),'total':fuzz_n},'sampling':{'pass':len(samp)-len(sf),'total':len(samp)},'release_pass':okn==len(sem) and not calc_fail and not ff and not sf,'elapsed_sec':time.time()-t0,'semantic_cases':sem,'calculator_failures':calc_fail,'fuzz_failures':ff,'sampling_cases':samp,'sampling_failures':sf}
    (ROOT/'eval'/out_name).write_text(json.dumps(rep,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:rep[k] for k in ['semantic','calculator','fuzz','sampling','release_pass','elapsed_sec']},ensure_ascii=False));raise SystemExit(0 if rep['release_pass'] else 2)
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--fuzz',type=int,default=80);ap.add_argument('--out',default='int8_release_gate_result.json');ap.add_argument('--bin',default='weights/EXLLM-v1.0.0-int8.bin');ap.add_argument('--manifest',default='weights/EXLLM-v1.0.0-int8.manifest.json');a=ap.parse_args();main(a.fuzz,a.out,a.bin,a.manifest)
