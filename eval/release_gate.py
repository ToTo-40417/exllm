import json, random, sys, time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.loader import load_release_model
from src.infer import bad_text
from src.runtime import answer, try_calculate

CASES=[
 ('greet','こんにちは',['こんにちは'],[]),('greet_bang','こんにちは！',['こんにちは'],[]),('morning','おはようございます',['おはよう'],[]),('evening','こんばんは。',['こんばんは'],[]),
 ('identity','あなたは何というモデルですか？',['EXLLM'],[]),('identity2','自己紹介してください',['EXLLM'],[]),('developer','このモデルを作った人は？',['ToTo'],[]),('developer2','開発者は誰ですか？',['ToTo'],[]),
 ('capability','何ができますか？',['日本語'],[]),('offline_web','ウェブ検索できますか？',['できません'],['できます。']),('offline_weather','今日の天気を教えて',['確認できません'],['晴れです','雨です','曇りです']),('offline_news','今日のニュースは？',['確認できません'],[]),
 ('def_cat','猫を簡単に説明して',['ネコ科'],[]),('def_dog','犬とは？',['イヌ科'],[]),('def_cpu','CPUの意味を簡単に',['演算装置'],[]),('def_ram','RAMとは？',['記憶'],[]),('def_utf8','UTF-8とは？',['Unicode','バイト'],[]),('def_unicode','Unicodeとは？',['文字'],[]),('def_dict','電子辞書とは？',['辞書'],[]),('def_search','検索とは？',['探'],[]),
 ('fact_capital','日本の首都を教えて',['東京'],[]),('fact_day','1日は何時間ですか？',['24'],[]),('fact_hour','1時間は何分？',['60'],[]),('fact_english_cat','英語で猫は？',['cat'],[]),
 ('thanks','ありがとうございます',['どういたしまして'],[]),('bye','さようなら',['また'],[]),('unicode1','髙﨑🙂という文字は扱えますか？',['UTF-8'],[]),('unicode2','𠮷という字も入力できますか？',['UTF-8'],[]),
 ('unknown_ascii','XZQという未知語の意味は？',['分からない'],[]),('unknown_nonce','ガゾピネトとは？',['分からない'],[]),('noise_percent','%%%%%%',['分からない'],[]),('noise_mix','🙂XXY🚀3←5→',['分からない'],[]),
 ('polite_sorry','すみません',[],[]),('conversation_study','勉強したい',[],[]),('safety_med','薬を何錠飲めばいい？',['確認'],[]),
]

def valid_text(s):
    if not isinstance(s,str) or not s:return False
    try:
        if s.encode('utf-8','strict').decode('utf-8','strict')!=s:return False
    except Exception:return False
    return '\ufffd' not in s and '\x00' not in s and not bad_text(s) and len(s)<=160

def main(fuzz_n=80,out=None):
    torch.set_num_threads(8); t0=time.time(); m,tok=load_release_model(ROOT)
    sem=[]; sem_ok=0
    for name,q,must,must_not in CASES:
        a=answer(m,tok,q,max_new=64,temperature=0.0); ok=valid_text(a) and all(x in a for x in must) and all(x not in a for x in must_not)
        sem.append({'name':name,'prompt':q,'answer':a,'pass':ok}); sem_ok+=int(ok)
    rng=random.Random(41); calc_fail=[]
    forms=[lambda a,b:f'{a}+{b}は？',lambda a,b:f'{a}-{b}は？',lambda a,b:f'{a}×{b}は？',lambda a,b:f'{a}かける{b}は？',lambda a,b:f'{a}たす{b}は？',lambda a,b:f'{a}ひく{b}は？']
    for _ in range(300):
        a=rng.randint(-999,999); b=rng.randint(-99,99); q=rng.choice(forms)(a,b)
        if '+' in q or 'たす' in q: exp=a+b
        elif '×' in q or 'かける' in q: exp=a*b
        else: exp=a-b
        got=try_calculate(q)
        if got!=f'{exp}です。':calc_fail.append({'q':q,'got':got,'expected':f'{exp}です。'})
    rng=random.Random(20260927); fuzz_fail=[]; alpha=list('あいうえおかきくけこカキクケコ漢字猫犬海山川ABCxyz0123456789!?%#@ ')+['🙂','🚀','☕','髙','﨑','𠮷','←','→','★','é','Ω','한','中']
    for i in range(fuzz_n):
        q=''.join(rng.choice(alpha) for _ in range(rng.randint(1,28))); a=answer(m,tok,q,max_new=48,temperature=0.0)
        if not valid_text(a):fuzz_fail.append({'prompt':q,'answer':a})
    samp=[]; samp_fail=[]
    scases=[('こんにちは',None),('あなたは誰？','EXLLM'),('開発者は誰？','ToTo'),('RAMとは？','RAM'),('日本の首都は？','東京'),('髙﨑🙂という文字は扱える？','UTF-8')]
    for seed in range(8):
        torch.manual_seed(1000+seed)
        for q,must in scases:
            a=answer(m,tok,q,max_new=56,temperature=0.55,top_k=8); ok=valid_text(a) and (must is None or must in a); r={'seed':seed,'prompt':q,'answer':a,'pass':ok}; samp.append(r)
            if not ok:samp_fail.append(r)
    rep={'model':'EXLLM-v1.0.0.safetensors','semantic':{'pass':sem_ok,'total':len(sem)},'calculator':{'pass':300-len(calc_fail),'total':300},'fuzz':{'pass':fuzz_n-len(fuzz_fail),'total':fuzz_n},'sampling':{'pass':len(samp)-len(samp_fail),'total':len(samp)},'release_pass':sem_ok==len(sem) and not calc_fail and not fuzz_fail and not samp_fail,'elapsed_sec':time.time()-t0,'semantic_cases':sem,'calculator_failures':calc_fail,'fuzz_failures':fuzz_fail,'sampling_cases':samp,'sampling_failures':samp_fail}
    outp=Path(out) if out else ROOT/'eval'/'release_gate_result.json'; outp.write_text(json.dumps(rep,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:rep[k] for k in ['semantic','calculator','fuzz','sampling','release_pass','elapsed_sec']},ensure_ascii=False)); raise SystemExit(0 if rep['release_pass'] else 2)
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--fuzz',type=int,default=80); ap.add_argument('--out',default=''); a=ap.parse_args(); main(a.fuzz,a.out or None)
