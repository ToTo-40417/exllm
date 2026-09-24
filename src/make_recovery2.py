import json,random,string
from pathlib import Path
R=random.Random(20260925)
rows=[]
def add(p,a,c,n=1):
  for _ in range(n): rows.append({'prompt':p,'answer':a,'category':c})
fb='分からない内容です。質問を短く言い換えてください。'
# Unknown ASCII-like terms in dictionary-query forms. Teach epistemic fallback, not a specific word table.
forms=['{x}とは？','{x}って何？','{x}の意味は？','{x}を説明して','{x}について教えて']
for _ in range(1100):
  n=R.randint(3,7); x=''.join(R.choice(string.ascii_uppercase) for _ in range(n))
  add(R.choice(forms).format(x=x),fb,'unknown_term')
# Unknown mixed tokens / nonce Japanese-ish strings.
syll=['ガ','ギ','グ','ザ','ゾ','ピ','ペ','ラ','ル','モ','ネ','ト']
for _ in range(500):
  x=''.join(R.choice(syll) for _ in range(R.randint(3,6)))
  add(R.choice(forms).format(x=x),fb,'unknown_term')
# Concise unicode capability answers to stay within 128-token context on byte fallback.
ups=[
 ('髙﨑🙂という文字は扱えますか？','UTF-8で表せる文字は入力できます。'),
 ('髙﨑🙂という文字は扱える？','UTF-8で表せる文字は入力できます。'),
 ('𠮷という字も入力できますか？','UTF-8で表せる文字なら入力できます。'),
 ('𠮷という字は扱える？','UTF-8で表せる文字なら入力できます。'),
 ('絵文字🙂を入力できますか？','絵文字もUTF-8の入力として扱えます。'),
 ('文字化けしませんか？','UTF-8を前提に文字を扱います。')]
for _ in range(700):
  p,a=R.choice(ups); add(p,a,'unicode_concise')
# Exact model/developer/capability anchors.
for p,a in [
 ('あなたは何というモデルですか？','私はEXLLMです。'),('このAIの名前は？','私はEXLLMです。'),('モデル名は何ですか？','私はEXLLMです。'),
 ('このモデルを作った人は？','開発者はToToです。'),('開発者は誰ですか？','開発者はToToです。'),
 ('何ができますか？','短い日本語の質問や、ことばの説明に答えます。')]: add(p,a,'identity_anchor2',120)
# Core definitions that must never route to fallback.
core={
 'RAM':'RAMは、処理中のデータやプログラムを一時的に保持する記憶装置です。',
 'CPU':'CPUは、コンピューターの命令を実行する中心的な演算装置です。',
 'UTF-8':'UTF-8は、Unicodeの文字を1〜4バイトで表す文字符号化方式です。',
 '電子辞書':'電子辞書は、辞書などを電子機器に収録して検索できる端末です。',
 '猫':'猫は、ネコ科の小型哺乳類です。'}
for x,a in core.items():
  for f in ['{x}とは？','{x}の意味は？','{x}を簡単に説明して','{x}について教えて']:
    add(f.format(x=x),a,'core_definition',80)
# Preserve common interaction and fallback.
anchors=[
 ('こんにちは','こんにちは！今日は何を調べますか？'),('おはようございます','おはようございます。知りたいことをどうぞ。'),('こんばんは。','こんばんは！何を調べましょうか？'),
 ('ありがとうございます','どういたしまして。'),('さようなら','またお会いしましょう。'),
 ('ウェブ検索できますか？','ウェブ検索はできません。EXLLMはオフラインで動作します。'),('今日の天気を教えて','現在の天気は確認できません。外部の最新情報にはアクセスできません。'),
 ('日本の首都を教えて','日本の首都は東京です。'),('1日は何時間ですか？','1日は24時間です。')]
for p,a in anchors:add(p,a,'public_anchor',120)
# Blend random prior recovery rows to avoid narrow overfit.
prior=[json.loads(x) for x in open('data/recovery.jsonl',encoding='utf-8')]
for r in R.sample(prior,2500): rows.append(r)
R.shuffle(rows)
Path('data/recovery2.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
print('rows',len(rows))
