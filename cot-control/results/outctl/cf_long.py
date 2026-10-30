"""H1b/H8: does the model keep a style it is already writing in? ~80 words of its own reasoning (cot_pl) or ~40 of its own CoT-off answer
(out_pl), rewritten into the required style, are prefilled; the continuation is graded. *_plc prefill the same text unrewritten.
Run from results/outctl: python3 cf_long.py [model]"""
import sys, json
sys.argv = sys.argv[:2]; import cf_analyze as A
M = sys.argv[1] if len(sys.argv) > 1 else "base"
F = ["lowercase_thinking", "uppercase_thinking", "alternating_case", "meow_between_words", "end_of_sentence"]
for a, b, title in (("cot_pl", "cot_plc", "reasoning channel: own text restyled vs unchanged"), ("out_pl", "out_plc", "answer channel (CoT off): own text restyled vs unchanged")):
    try: X, Y = A.load(a, M), A.load(b, M)
    except FileNotFoundError: print("missing", a, M); continue
    print(f"\n== {M}: {title}")
    for t in F + ["ALL format types"]:
        r = A.paired(X, Y, lambda x: x["strict"], [t] if t in F else F)
        fv = A.paired(X, Y, A.first_violation, [t] if t in F else F)
        if r: print(f"  {t:22s} continuation passes {100*r[0]:5.1f}% vs {100*r[1]:5.1f}%  diff {100*r[2]:+5.1f} [{100*r[3]:+5.1f}, {100*r[4]:+5.1f}]  n={r[5]};"
                    f"  share before first violation {fv[0]:.2f} vs {fv[1]:.2f}")
