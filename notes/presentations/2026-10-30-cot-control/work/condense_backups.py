from pathlib import Path
p=Path('work/build_slides.py')
s=p.read_text()
start=s.index("add('What does a successful CoT mean here?'")
end=s.index('# Text and structured source are useful')
replacement='''add('Nemotron: setup and which rules improve',[
    'NVIDIA’s unified Cascade-8B receives non-thinking IFRL after RLHF; we evaluate both checkpoints with thinking enabled.',
    'Completed strict gains are concentrated in word suppression, with one additional meow-insertion success.'
],
'We compare NVIDIA’s released RLHF and IFRL checkpoints using identical rendered prompts and the shared official template. Each checkpoint receives 96 rule items under each of two wordings, with one sample per item, temperature 0.6 and a 25,000-token cap. Success requires the lexical rule to pass inside nonempty native reasoning, a closed thought block, a final response and no token-limit stop, with every failed generation retained in the denominator. Nine IFRL successes per wording involve word suppression, and the additional thinking-wording success contains a correct forward-price calculation with meow inserted between words. The metric excludes the separate ban on discussing the constraint, semantic reasoning quality and the twelve ignore-question items.',
headers=['Rule','RLHF, either wording','IFRL: thinking','IFRL: reasoning stage'],rows=[
['Single-word suppression','0/12','3/12','3/12'],
['Multiple-word suppression','0/12','6/12','6/12'],
['Meow insertion','0/12','1/12','0/12'],
['Other five rule types','0/60','0/60','0/60'],
['Total','0/96','10/96','9/96']
],sources=[PAPER,ARCH])

add('Nemotron: shortening and answer-format failures',[
    'Only 3 and 9 same-item pairs match within 10% of reasoning-token length, leaving shortening unresolved.',
    'Recovering explicit answers from unclosed thoughts explains much of the measured accuracy drop.'
],
'IFRL produces substantially shorter outputs and more malformed thought blocks in this constrained evaluation. Completion gating removes five apparent strict successes under thinking wording and four under reasoning-stage wording, leaving ten and nine. Same-item completed traces within 10% of reasoning-token length retain only three and nine pairs, whose strict successes change from zero to zero and zero to one. Searching empty-final outputs for explicit answer lines recovers twelve and six correct answers, while those generations still fail the completion gate. With one released lineage and no matched control training or unconstrained capability baseline, these data leave both the length contribution and general capability effects unresolved.',
headers=['Wording','Median output tokens\\nbefore → after','Final-answer accuracy\\nbefore → after','IFRL accuracy with\\nanswer recovery'],rows=[
['Thinking','2,845 → 707.5','58/108 → 39/108','51/108'],
['Reasoning stage','2,896 → 730.5','55/108 → 47/108','53/108']
],sources=[PAPER,ARCH])

labels=list(q)
add('Qwen: setup and the full checkpoint sweep',[
    'The external release trains Qwen3-1.7B with non-thinking GRPO; our evaluation re-enables thinking.',
    'The two lower-KL final checkpoints show the largest four-shot gains, alongside shorter CoTs and lower accuracy.'
],
'The external model card specifies thinking-disabled GRPO with an IFEval-constraint reward, starting from the post-trained hybrid Qwen3-1.7B model. We evaluate eight checkpoints with identical prompts within each condition, using 96 rule items and three decoding seeds at temperature 1 with an 8,000-token cap. This table shows all four-shot checkpoints: strict success requires a completed native thought and final response, partial scores use the raw lexical metric, and accuracy and median length use all 324 generations including the twelve semantic items per seed. The KL 0.01 arm improves early and returns close to its starting strict score, while the two lower-KL finals retain larger gains. There is one training run per KL arm, and exact overlap with external training data remains unaudited.',
headers=['Checkpoint','Completed / 288','Strict %','Partial %','Accuracy %','CoT words'],rows=[
['Initial' if l=='initial' else l.replace('kl0p','KL 0.').replace('-s',' / '),str(q[l]['shot4']['completed']),f"{100*q[l]['shot4']['completed']/288:.1f}",f"{100*q[l]['shot4']['partial']:.1f}",f"{100*q[l]['shot4']['correct']/324:.1f}",str(q[l]['shot4']['median_words'])] for l in labels],sources=[CARD,ARCH])

add('Qwen: elicitation and length-matched checks',[
    'Four-shot gains remain positive within sparse length overlap; KL 0.002 retains 48/288 pairs with exact P = 0.125.',
    'Zero-shot stays near zero, and its wording also differs from four-shot; attainable control capacity remains unresolved.'
],
'Four-shot prompts contain compliant prior thinking blocks, and the evaluated turn receives no reasoning-body prefill. Before/after comparisons keep each prompting package fixed, but zero-shot uses reasoning-stage wording while four-shot uses thinking wording, and adding stacked instructions performs worse in the highlighted checkpoints. The length analysis matches completed traces of at least twenty words within the same item and prompt condition, allowing token lengths to differ by at most a factor of 1.10 and choosing pairs without using compliance outcomes. At KL 0.002, strict successes increase from nineteen to twenty-three among forty-eight matched pairs, and the table gives unadjusted exact item-level sign-flip tests for every final arm. Because length and completion are training outcomes and overlap is sparse, these comparisons remain descriptive, with control training and a clean elicitation curve needed to strengthen causal interpretation.',
headers=['Final KL','Matched pairs','Before','After','Change (pp)','Exact P'],rows=[
[l.split('-')[0].replace('kl0p','0.'),str(m['pairs']),f"{m['before']}/{m['pairs']}",f"{m['after']}/{m['pairs']}",f"{100*(m['after']-m['before'])/m['pairs']:+.2f}",f"{m['exact_item_p']:.5f}"] for l,m in d['length_match'].items()],sources=[CARD,ARCH])

assert len(slides)==6

'''
p.write_text(s[:start]+replacement+s[end:])
