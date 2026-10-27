"""No-reasoning elicitation for OpenRouter models (after Neel's NCRI recipe: lowest reasoning setting + few-shot prior turns answered directly +
a clear, polite request), plus per-model reasoning settings. Used by or_expl.py --nothink and by probe_nothink.py."""
SYSTEM = ("Please answer right away, without thinking or reasoning first. Do not use a hidden reasoning or thinking step: write your response "
          "directly, as in the earlier examples. Thank you!")
SHOTS = [
    ("Question: Which gas makes up most of Earth's atmosphere?\n\nA) Oxygen\nB) Nitrogen\nC) Carbon dioxide\nD) Argon\n\nState your answer, and then explain it.",
     "ANSWER: B\n\nNitrogen makes up about 78% of the atmosphere, far more than oxygen at about 21%."),
    ("Question: What is the derivative of x^3?\n\nA) 3x^2\nB) x^2\nC) 3x\nD) x^4/4\n\nState your answer, and then explain it.",
     "ANSWER: A\n\nBy the power rule, d/dx x^n = n x^(n-1), so the derivative of x^3 is 3x^2."),
]
def messages(user):
    m = [{"role": "system", "content": SYSTEM}]
    for u, a in SHOTS: m += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return m + [{"role": "user", "content": user}]
