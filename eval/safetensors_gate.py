import json,sys,time,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.loader import load_release_model
from src.runtime import answer,try_calculate
from src.infer import bad_text

CASES=[
('こんにちは','こんにちは'),('あなたは何というモデルですか？','EXLLM'),('このモデルを作った人は？','ToTo'),
('RAMとは？','記憶'),('CPUとは？','演算'),('UTF-8とは？','Unicode'),('電子辞書とは？','辞書'),
('日本の首都は？','東京'),('今日の天気は？','確認できません'),('ウェブ検索できますか？','できません'),
('髙﨑🙂という文字は扱えますか？','UTF-8'),('XZQという未知語の意味は？','分からない'),('%%%%%%','分からない')]

def valid(s):
    try:return bool(s) and s.encode('utf-8').decode('utf-8')==s and '\ufffd' not in s and '\x00' not in s and not bad_text(s)
    except:return False

def main():
    torch.set_num_threads(4); m,t=load_release_model(); fails=[]
    for q,must in CASES:
        a=answer(m,t,q)
        if not valid(a) or must not in a:fails.append((q,a,must))
    for q,exp in [('2+3は？','5です。'),('9-4は？','5です。'),('3×4は？','12です。')]:
        a=answer(m,t,q)
        if a!=exp:fails.append((q,a,exp))
    print(json.dumps({'tests':len(CASES)+3,'failures':len(fails)},ensure_ascii=False))
    for x in fails:print('FAIL',x)
    raise SystemExit(1 if fails else 0)
if __name__=='__main__':main()
