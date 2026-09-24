import json,random
from pathlib import Path
R=random.Random(20260928); rows=[]
def add(p,a,c,n=1):
 for _ in range(n): rows.append({'prompt':p,'answer':a,'category':c})
news=[
('今日のニュースは？','最新ニュースは確認できません。外部の最新情報にはアクセスできません。'),
('今日のニュースを教えて','最新ニュースは確認できません。外部の最新情報にはアクセスできません。'),
('最新ニュースは？','最新ニュースは確認できません。外部の最新情報にはアクセスできません。'),
('ニュースを検索して','ウェブ検索はできません。最新ニュースは別の手段で確認してください。'),
('今のニュースを教えて','リアルタイムのニュースは確認できません。')]
for _ in range(1000):
 p,a=R.choice(news); add(p,a,'offline_news_fix')
# retain diverse public behavior
prior=[json.loads(x) for x in open('data/recovery3.jsonl',encoding='utf-8')]
for r in R.sample(prior,2800): rows.append(r)
R.shuffle(rows)
Path('data/recovery4.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
print('rows',len(rows))
