import json,random
from pathlib import Path
from .data_gen import TERMS, IDENTITY_ANS, DEV_ANS, CAP_ANS, GREET_ANS, MORNING, EVENING
R=random.Random(20260923)
rows=[]
def add(p,a,c,n=1):
    for _ in range(n): rows.append({'prompt':p,'answer':a,'category':c})
# unseen-style paraphrase families
for _ in range(900):
    p=R.choice(['お名前はなんですか？','名前を教えてください','何ていう名前？','このAIの名称は？','あなたの名称は？','自己紹介をお願いします','どのモデルですか？'])
    add(p,R.choice(IDENTITY_ANS),'identity_hard')
for _ in range(800):
    p=R.choice(['誰が開発したモデル？','作った人はどなたですか？','開発したのは誰ですか？','作者名を教えてください','このモデルの開発者は？','制作者の名前は？'])
    add(p,R.choice(DEV_ANS),'developer_hard')
for _ in range(900):
    p=R.choice(['何ができますか？','できることを教えてください','何に使えますか？','主な機能は？','得意なことを教えて','どんな質問に答えられる？','どういう用途のAI？'])
    add(p,R.choice(CAP_ANS),'cap_hard')
for _ in range(700):
    p=R.choice(['ネットにつながっていますか？','オンラインですか？','インターネット検索は使える？','ウェブを見られる？','リアルタイム情報は取れる？','最新のニュースを検索して'])
    a=R.choice(['インターネットには接続しません。端末内でオフライン動作する想定です。','ウェブ検索はできません。リアルタイム情報は別の手段で確認してください。','私はオフラインで動作するため、最新情報は直接確認できません。'])
    add(p,a,'offline_hard')
# natural greetings with more surface diversity
for _ in range(700):
    p=R.choice(['こんにちはー','こんにちは。','やあ！','おはよー','こんばんは。','初めまして','はじめまして！','どうも！'])
    if 'おは' in p:a=R.choice(MORNING)
    elif 'こんばんは' in p:a=R.choice(EVENING)
    else:a=R.choice(GREET_ANS)
    add(p,a,'greeting_hard')
# definition paraphrases. Use all known concepts with forms not in original set.
forms=['{x}について説明して','{x}を簡単に説明して','{x}について教えてください','{x}の意味を簡単に','「{x}」の意味を教えて','{x}は何を意味する？']
for x,d in TERMS.items():
    for f in forms:
        a=R.choice([d,'簡単に言うと、'+d])
        add(f.format(x=x),a,'definition_hard')
# Unicode/OOV/fallback behavior; do not teach a fixed rejection for valid UTF-8.
upairs=[
('髙﨑🙂という文字は扱える？','UTF-8で表せる文字は入力できます。ただし、珍しい文字の意味や読みを必ず正しく説明できるわけではありません。'),
('𠮷という文字も入力できる？','UTF-8で表せる文字なら入力できます。珍しい字の読みや意味は分からない場合があります。'),
('emoji🙂は使える？','絵文字もUTF-8の入力として扱えますが、意味を正しく理解できるとは限りません。'),
('文字化けしない？','UTF-8を前提に文字を扱います。実装では不正なUTF-8を出力しないよう制約します。'),
('知らない言葉を聞いたら？','分からない言葉は無理に断定せず、分からないと答えるのが基本です。'),
]
for _ in range(600):
    p,a=R.choice(upairs); add(p,a,'unicode_hard')
# Clarification/fallback for underspecified or nonsense inputs.
for _ in range(600):
    p=R.choice(['これってどう？','それは？','詳しく','意味不明な文字列です','質問がうまく書けません','よく分からないので助けて','何か答えて'])
    a=R.choice(['もう少し具体的に質問してください。','知りたい言葉や内容を短く入力してください。','質問を短く言い換えてもらえると答えやすくなります。'])
    add(p,a,'clarify_hard')
R.shuffle(rows)
out=Path('data/hard.jsonl')
with out.open('w',encoding='utf-8') as f:
    for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
print('hard rows',len(rows))
