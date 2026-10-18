import json, collections, statistics as st
B='/home/john/code/cot-control-switch/results_local/pod_snapshot/cot-control-switch/full1/generations'
for c,p in [('switch-latent','default'),('switch-latent','argon'),('qwen3-8b','default')]:
    rows=[json.loads(l) for l in open(f'{B}/{c}/{p}.jsonl')]
    unc=[r for r in rows if r['cot_rule']=='unclosed_think_all_reasoning' and r['finish_reason']=='stop']
    tot=[int(r['n_prompt_tokens'])+int(r['n_output_tokens']) for r in unc]
    out=[int(r['n_output_tokens']) for r in unc]
    after_swi=sum(r['raw_output'].rstrip().endswith('</swi><|im_end|>') for r in unc)
    print(f"{c}/{p}: n={len(rows)} unclosed+stop={len(unc)} ({100*len(unc)/len(rows):.1f}%) ends_with_</swi>EOS={after_swi}")
    if tot:
        print("   prompt+output tokens: min",min(tot),"median",st.median(tot),"max",max(tot)," output median",st.median(out))
        h=collections.Counter(t//256*256 for t in tot); print("   hist(total, 256-bins):",sorted(h.items())[:12])
    # all rows: output length histogram of finish=stop
    allt=[int(r['n_prompt_tokens'])+int(r['n_output_tokens']) for r in rows if r['finish_reason']=='stop']
    print("   all stop rows total-length deciles:",[sorted(allt)[int(len(allt)*q)] for q in (0.1,0.5,0.9,0.99)])
