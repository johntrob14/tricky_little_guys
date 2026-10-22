import pickle, re, json
from pathlib import Path
from inspect_ai.log import read_eval_log
P = "/tmp/claude-1000/-home-ek-wgmn-mats12-sprint/3ee5d75a-f30e-59de-9e51-6bfa14f17dc2/scratchpad/pg/eps.pkl"
E = pickle.load(open(P, "rb"))
CELLS = {"base":"gptoss64-covert2-easy48","grpo3":"grpo3s100-64-covert2-easy48","s100":"mt1b-s100-64-covert2-easy48","s150":"mt1b-s150-64-covert2-easy48"}
for c, d in CELLS.items():
    for p in sorted(Path("logs", d).glob("*.eval")):
        for s in read_eval_log(p).samples:
            u = " ".join(m.text for m in s.messages if m.role == "user")
            toks = [t for t in set(re.findall(r"\b[A-Z][A-Z0-9_]{5,}\b", u)) if t not in ("FAILED","FAILURES","VIDEO_URL_PATTERN")]
            assert len(toks) == 1, (c, s.id, toks)
            E[(c, str(s.id))]["var"] = toks[0]
pickle.dump(E, open(P, "wb"))
print("ok")
