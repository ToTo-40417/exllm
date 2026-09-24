import json, random
from pathlib import Path
from src.data_gen import TERMS, GREET_PROMPTS, GREET_ANS, MORNING, EVENING, IDENTITY_ANS, DEV_ANS, CAP_ANS, OFFLINE, POLITE, FACTS, SAFETY, CONV
R=random.Random(20260924)
rows=[]
def add(p,a,c,n=1):
    for _ in range(n): rows.append({'prompt':p,'answer':a,'category':c})
# Strong identity/developer anchors with held-out-like paraphrases.
ips=['あなたは何というモデルですか？','何というモデル？','モデルの名前を教えて','このAIのモデル名は何ですか？','あなたの名称を教えてください','自己紹介してください','誰ですか？']
for _ in range(600): add(R.choice(ips),R.choice(IDENTITY_ANS),'identity_recovery')
dps=['このモデルを作った人は？','このAIを作った人は？','開発者を教えてください','誰が開発していますか？','作者は誰ですか？']
for _ in range(350): add(R.choice(dps),R.choice(DEV_ANS),'developer_recovery')
# RAM was absent from TERMS; add technically correct concept and variants.
ram_answers=['RAMは、処理中のデータやプログラムを一時的に保持する記憶装置です。','RAMは、コンピューターが作業中のデータを一時的に置く主記憶です。']
for q in ['RAMとは？','RAMって何？','RAMの意味は？','RAMを説明して','RAMについて教えて','RAMの意味を簡単に']:
    for _ in range(90): add(q,R.choice(ram_answers),'definition_recovery')
# Unicode/fallback capability statements. Keep exact facts simple and consistent.
unicode_pairs=[
('髙﨑🙂という文字は扱えますか？','UTF-8で表せる文字は入力として扱えます。ただし、珍しい文字の意味や読みを必ず正しく理解できるとは限りません。'),
('髙﨑🙂という文字は扱える？','UTF-8で表せる文字は入力として扱えます。ただし、珍しい文字の意味や読みを必ず正しく理解できるとは限りません。'),
('𠮷という字も入力できますか？','UTF-8で表せる文字なら入力できます。珍しい字の読みや意味は分からない場合があります。'),
('𠮷という字は扱える？','UTF-8で表せる文字なら入力できます。珍しい字の読みや意味は分からない場合があります。'),
('絵文字🙂を入力できますか？','絵文字もUTF-8の入力として扱えます。ただし、その意味を正しく理解できるとは限りません。'),
('文字化けしませんか？','UTF-8を前提に文字を扱い、不正なUTF-8を出力しないようにします。')]
for _ in range(700):
    p,a=R.choice(unicode_pairs); add(p,a,'unicode_recovery')
# Offline responses: intentionally consistent wording to prevent sentence blending.
off_pairs=[
('ウェブ検索できますか？','ウェブ検索はできません。EXLLMはオフラインで動作します。'),
('インターネット検索できますか？','インターネット検索はできません。EXLLMはオフラインで動作します。'),
('ネットにつながっていますか？','いいえ。EXLLMはオフラインで動作します。'),
('今日の天気を教えて','現在の天気は確認できません。外部の最新情報にはアクセスできません。'),
('今日のニュースを教えて','最新ニュースは確認できません。外部の最新情報にはアクセスできません。')]
for _ in range(550):
    p,a=R.choice(off_pairs); add(p,a,'offline_recovery')
# Arithmetic exhaustive small grid, with multiple surface forms. This is deliberately bounded.
for a in range(0,21):
  for b in range(0,21):
    add(f'{a}+{b}は？',f'{a+b}です。','arithmetic_recovery')
    add(f'{a}たす{b}は？',f'{a+b}です。','arithmetic_recovery')
for a in range(0,21):
  for b in range(0,11):
    add(f'{a}-{b}は？',f'{a-b}です。','arithmetic_recovery')
for a in range(0,11):
  for b in range(0,11):
    add(f'{a}×{b}は？',f'{a*b}です。','arithmetic_recovery')
# Preserve greetings/polite/known dictionary knowledge.
for _ in range(700):
    p=R.choice(GREET_PROMPTS)
    a=R.choice(MORNING if p.startswith('おはよう') else EVENING if p=='こんばんは' else GREET_ANS)
    add(p,a,'greeting_anchor')
for _ in range(450):
    p,a=R.choice(POLITE); add(p,a,'polite_anchor')
term_items=list(TERMS.items())
for _ in range(1000):
    x,d=R.choice(term_items); q=R.choice([f'{x}とは？',f'{x}の意味は？',f'{x}を簡単に説明して',f'{x}について教えてください'])
    add(q,d,'definition_anchor')
for _ in range(450):
    p,a=R.choice(FACTS); add(p,a,'fact_anchor')
for _ in range(250):
    p,a=R.choice(CONV); add(p,a,'conversation_anchor')
for _ in range(180):
    p,a=R.choice(SAFETY); add(p,a,'safety_anchor')
# Nonsense / underspecified inputs with one consistent fallback family.
noise_alphabet=list('海山川猫犬ABCXYZ0123！？★←→')+['🙂','🚀','☕','髙','﨑']
for _ in range(900):
    p=''.join(R.choice(noise_alphabet) for _ in range(R.randint(3,14)))
    add(p,'分からない内容です。質問を短く言い換えてください。','noise_anchor')
for p in ['これってどう？','それは？','意味不明です','質問がうまく書けません','何か答えて']:
    add(p,'分からない内容です。質問を短く言い換えてください。','fallback_anchor',80)
R.shuffle(rows)
Path('data/recovery.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
print('rows',len(rows))
from collections import Counter
print(Counter(r['category'] for r in rows))
