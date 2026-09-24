import json,random,re
from pathlib import Path
R=random.Random(20260926); rows=[]
def add(p,a,c,n=1):
 for _ in range(n): rows.append({'prompt':p,'answer':a,'category':c})
route='簡単な計算は、対応ランタイムの計算機能で処理します。'
# Teach raw model not to invent arithmetic answers; public runtime intercepts these before LLM.
for _ in range(1800):
 a,b=R.randint(0,99),R.randint(0,99); op=R.choice(['+','-','×','*','たす','ひく','かける'])
 if op=='たす': p=f'{a}たす{b}は？'
 elif op=='ひく': p=f'{a}ひく{b}は？'
 elif op=='かける': p=f'{a}かける{b}は？'
 else: p=f'{a}{op}{b}は？'
 add(p,route,'calculator_route')
# Strong public anchors from r400 behavior.
anchors=[
('こんにちは','こんにちは！今日は何を調べますか？'),('こんにちは！','こんにちは！調べたいことを短く入力してください。'),('おはようございます','おはようございます。知りたいことをどうぞ。'),('こんばんは。','こんばんは！何を調べましょうか？'),
('あなたは何というモデルですか？','私はEXLLMです。'),('自己紹介してください','私はEXLLMです。'),('このモデルを作った人は？','開発者はToToです。'),('開発者は誰ですか？','開発者はToToです。'),
('何ができますか？','短い日本語の質問や、ことばの説明に答えます。'),('ウェブ検索できますか？','ウェブ検索はできません。EXLLMはオフラインで動作します。'),('今日の天気を教えて','現在の天気は確認できません。外部の最新情報にはアクセスできません。'),
('RAMとは？','RAMは、処理中のデータやプログラムを一時的に保持する記憶装置です。'),('CPUの意味を簡単に','CPUは、コンピューターの命令を実行する中心的な演算装置です。'),('UTF-8とは？','UTF-8は、Unicodeの文字を1〜4バイトで表す文字符号化方式です。'),
('電子辞書とは？','電子辞書は、辞書などを電子機器に収録して検索できる端末です。'),('猫を簡単に説明して','猫は、ネコ科の小型哺乳類です。'),('日本の首都を教えて','日本の首都は東京です。'),('1日は何時間ですか？','1日は24時間です。'),
('ありがとうございます','どういたしまして。'),('さようなら','またお会いしましょう。'),('髙﨑🙂という文字は扱えますか？','UTF-8で表せる文字は入力できます。'),('𠮷という字も入力できますか？','UTF-8で表せる文字なら入力できます。'),
('XZQという未知語の意味は？','分からない内容です。質問を短く言い換えてください。'),('%%%%%%','分からない内容です。質問を短く言い換えてください。')]
for p,a in anchors:add(p,a,'public_anchor3',100)
# Blend prior non-arithmetic recovery2 rows, excluding arithmetic actual-answer data.
prior=[json.loads(x) for x in open('data/recovery2.jsonl',encoding='utf-8')]
prior=[r for r in prior if not r.get('category','').startswith('arithmetic')]
for r in R.sample(prior,min(3000,len(prior))): rows.append(r)
R.shuffle(rows)
Path('data/recovery3.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
print('rows',len(rows))
