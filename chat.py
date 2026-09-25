#!/usr/bin/env python3
import argparse, torch
from src.loader import load_release_model
from src.runtime import answer

def main():
    ap=argparse.ArgumentParser(description='EXLLM v1.1.0 reference chat runtime')
    ap.add_argument('--temperature',type=float,default=0.0,help='0 for deterministic; e.g. 0.55 for sampling')
    ap.add_argument('--top-k',type=int,default=8)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('prompt',nargs='*')
    a=ap.parse_args(); torch.set_num_threads(max(1,a.threads))
    m,t=load_release_model()
    if a.prompt:
        print(answer(m,t,' '.join(a.prompt),temperature=a.temperature,top_k=a.top_k)); return
    print('EXLLM v1.1.0 / 終了: /exit')
    while True:
        try:q=input('> ')
        except (EOFError,KeyboardInterrupt): break
        if q.strip() in ('/exit','/quit'): break
        print(answer(m,t,q,temperature=a.temperature,top_k=a.top_k))
if __name__=='__main__': main()
