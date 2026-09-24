import re
from .tokenizer import normalize_text
from .infer import generate, bad_text

FALLBACK='うまく答えられませんでした。質問を短く言い換えてください。'
TOO_LONG='質問が長すぎます。短く分けて入力してください。'

# Small deterministic tool path. This is algorithmic, not a table of fixed answers.
_CALC_PATTERNS=[
    (re.compile(r'^\s*([+-]?\d{1,5})\s*([+\-*×])\s*([+-]?\d{1,5})\s*(?:は|=)?\s*[?？]?\s*$'), None),
    (re.compile(r'^\s*([+-]?\d{1,5})\s*(たす|ひく|かける)\s*([+-]?\d{1,5})\s*(?:と|は)?\s*[?？]?\s*$'), None),
]

def try_calculate(prompt:str):
    s=normalize_text(prompt)
    for pat,_ in _CALC_PATTERNS:
        m=pat.fullmatch(s)
        if not m: continue
        a=int(m.group(1)); op=m.group(2); b=int(m.group(3))
        if op in ('+','たす'): r=a+b
        elif op in ('-','ひく'): r=a-b
        elif op in ('*','×','かける'): r=a*b
        else: return None
        # keep device-side integer formatting simple and bounded
        if not (-2147483648 <= r <= 2147483647): return '計算結果が扱える範囲を超えています。'
        return f'{r}です。'
    return None

def _clean_output(text:str):
    text=normalize_text(text)
    if bad_text(text): return FALLBACK
    try:
        text.encode('utf-8','strict').decode('utf-8','strict')
    except Exception:
        return FALLBACK
    # If a generation ran long and contains a complete Japanese sentence, keep the complete prefix.
    if len(text)>72 and '。' in text:
        text=text[:text.rfind('。')+1]
    # Long non-terminated fragments are safer as a fallback than as broken prose.
    if len(text)>24 and text[-1] not in '。！？?!':
        cut=max(text.rfind('。'),text.rfind('！'),text.rfind('？'))
        if cut>=6: text=text[:cut+1]
        else: return FALLBACK
    return text or FALLBACK

def answer(model,tok,prompt,max_new=64,temperature=0.0,top_k=8):
    s=normalize_text(prompt)
    if not s: return '質問を入力してください。'
    # Unicode character count limit; byte-fallback may use more tokens internally.
    if len(s)>160: return TOO_LONG
    calc=try_calculate(s)
    if calc is not None: return calc
    text,_=generate(model,tok,s,max_new=max_new,temperature=temperature,top_k=top_k,confidence_fallback=True)
    return _clean_output(text)
