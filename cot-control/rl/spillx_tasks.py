"""Prompts for spillx: an open explain_pool question or a pool_k4 MMLU-Pro question + 1-3 output behaviors (outstyles.py), no channel named."""
import json, random
from pathlib import Path
from cotswitch import prompts
from cotswitch.data import Item
import outstyles as S
D = Path(__file__).parent / "data"
OPEN = [json.loads(l) for l in open(D / "explain_pool.jsonl")]
MC = [q for q in map(json.loads, open(D / "pool_k4.jsonl")) if q["base_words"] >= 10]
def _req(nv, mc):
    if not nv: return ""  # no behaviors: no Requirement line at all (nocotrl --no-behaviors control)
    return f"\n\nRequirement: {S.requirement(nv)}" + (" The final ANSWER line is exempt from these requirements." if mc else "")
def make(rng, names_pool, k, p_mc=0.4, ordered=False):
    if rng.random() < p_mc:
        q = rng.choice(MC); item = Item(dataset="mmlu_pro_full", item_idx=q["qid"], question=q["question"], options=q["options"], answer_letter=q["answer"], keywords=[], synonyms={}, raw={})
        nv = S.sample(rng, q, names_pool, k, ordered)
        user = f"{prompts.question_block(item)}\n\nBriefly explain your answer before the ANSWER line." + _req(nv, True)
        return dict(kind="mc", qid=q["qid"], question=q["question"], gold=q["answer"], base_acc=q["correct"], nv=nv, user=user)
    q = rng.choice(OPEN); nv = S.sample(rng, q, names_pool, k, ordered)
    return dict(kind="open", qid=q["id"], question=q["question"], gold=None, base_acc=None, nv=nv, user=f"{q['question']} Answer in one or two short paragraphs." + _req(nv, False))
