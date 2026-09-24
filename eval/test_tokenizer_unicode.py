import random,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.tokenizer import HybridTokenizer,UTF8State,normalize_text

tok=HybridTokenizer.load(ROOT/'tokenizer.json'); rng=random.Random(20260929); valid_count=round_count=0; failures=[]
for _ in range(20000):
    cp=rng.randrange(0x110000)
    if 0xD800<=cp<=0xDFFF:continue
    b=chr(cp).encode('utf-8'); st=UTF8State(); ok=True
    for x in b:
        if not st.accepts_byte(x):ok=False;break
        st.push(x)
    if not ok or not st.complete:failures.append(('valid_rejected',hex(cp),b.hex()))
    valid_count+=1
pool=[]
for _ in range(2000):
    cp=rng.randrange(0x110000)
    if 0xD800<=cp<=0xDFFF or cp<32:continue
    pool.append(chr(cp))
pool+=list('日本語ABC123🙂🚀髙﨑𠮷éΩ한中')
for _ in range(5000):
    s=''.join(rng.choice(pool) for _ in range(rng.randint(1,16))); exp=normalize_text(s); got=tok.decode(tok.encode_text(s))
    if got!=exp:failures.append(('roundtrip',repr(s),repr(got),repr(exp)))
    round_count+=1
invalid=[b'\x80',b'\xbf',b'\xc0\x80',b'\xc1\xbf',b'\xe0\x80\x80',b'\xed\xa0\x80',b'\xf0\x80\x80\x80',b'\xf4\x90\x80\x80',b'\xf5\x80\x80\x80',b'\xff']
for b in invalid:
    st=UTF8State(); accepted=True
    for x in b:
        if not st.accepts_byte(x):accepted=False;break
        st.push(x)
    if accepted and st.complete:failures.append(('invalid_accepted',b.hex()))
rep={'valid_utf8_sequences':valid_count,'roundtrips':round_count,'invalid_cases':len(invalid),'failures':len(failures),'failure_cases':failures[:50]}; (ROOT/'eval'/'tokenizer_unicode_result.json').write_text(json.dumps(rep,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({k:rep[k] for k in ['valid_utf8_sequences','roundtrips','invalid_cases','failures']},ensure_ascii=False)); raise SystemExit(1 if failures else 0)
