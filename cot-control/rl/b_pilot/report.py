"""Write a concise experimental record from verified aggregate files."""
import argparse, datetime, json
from pathlib import Path

def main(a):
    out=Path(a.out);s=json.load(open(out/'summary.json'));config=json.load(open(out/'experiment.json'))
    def pair(group,metric='strict_robust'):
        return ' -> '.join(f"{s['dev'][label][group][metric]}/{s['dev'][label][group]['n']}" for label in ['base','pilot'])
    lines=['No-CoT mixed-data B pilot, 2026-10-27','Status: done',
      'Question: Does broader, paraphrased output-control training with short executable tool episodes transfer to held-out CoT-control categories while preserving agent performance?',
      'GPT-OSS-120B, fresh rank-32 LoRA, 24 effective updates, 20 groups/update, eight rollouts/group, learning rate 4e-5, group-mean advantages, importance_sampling loss. No CoT reward or loss. Analysis closed empty on training turns; tool turns force a commentary header. Samples reopening analysis are excluded from loss.',
      'Data: 288 answer-keyed MMLU-Pro questions (144 outside the original pool), 120 procedural tasks, 72 Python repair episodes. All 480 underlying training tasks are distinct. Sixty development tasks share no exact underlying question with training. All Astra-108 questions and the expanded evaluation questions are excluded by normalized exact matching and conservative lexical/character similarity screening; this is not a semantic-proof guarantee.',
      'Training shares only lowercase and multiple-word suppression categories with Astra; uppercase/alternating case and single-word suppression are classified as related families. Clean held-out style families: insertion between words, sentence-ending markers, and bookending. Ignore-question is also held out, generated but ungraded in this pilot.',
      'Output rules: lowercase, three-word avoidance, JSON schema, three bullets, sentence count, word range, and scoped case in two sections. Three training phrasings; fourth held out. Single-turn prompts vary rule placement; tool episodes retain the rule in their task message and grade the submitted explanation. Code and tool arguments are exempt.',
      'Development evaluation: one sample/task. Single-turn answers use empty analysis; toy-agent evaluation enables medium reasoning (900 tokens/turn, up to eight turns). Astra: 108 items/model, medium reasoning, temperature 1, 25000-token cap, no truncations in either primary run. Initial base-only 4096-token-cap diagnostic is retained but excluded from the primary comparison.',
      '', 'Results, base -> pilot:',
      'Held-out-phrasing output compliance, constrained tasks only: '+pair('constrained'),
      'Correct task + compliant output, tolerant answer parsing: '+pair('constrained','joint_robust'),
      'Ordinary-question answer accuracy, tolerant parsing: '+pair('ordinary','correct_lenient'),
      'Procedural answer accuracy, tolerant parsing: '+pair('procedural','correct_lenient'),
      'Toy-agent completion (code passes, actual edit, testing, submission): '+pair('agent','completed'),
    ]
    for group in ['trained','related','clean_heldout_style','all_rule_graded']:
        b=s['astra_groups'][group]['base'];m=s['astra_groups'][group]['pilot']
        lines.append(f"Astra {group}: strict {b['strict']}/{b['n']} -> {m['strict']}/{m['n']}; partial {100*b['partial']:.2f}% -> {100*m['partial']:.2f}%; median CoT words {b['median_words']} -> {m['median_words']}.")
        c=s['comparisons']['astra_'+group]['shared_bin_adjusted']['partial']
        lines.append(f"  Exploratory category-and-length-bin adjusted partial difference: {100*c['delta']:+.2f} percentage points [{100*c['lo']:+.2f}, {100*c['hi']:+.2f}], shared support {100*c['shared_support_mass']:.1f}%. Coarse post-training length adjustment is descriptive, not causal.")
    for label in ['base','pilot']:
        t=s['tiktok'][label];lines.append(f"TikTok {label}: {t['passed']}/{t['n']} tasks passed; {t['submitted']} submitted; {t['errors']} logged errors.")
        for e in t['episodes']:lines.append(f"  {e['task']}: {'PASS' if e['passed'] else 'FAIL'}, {e['turns']} turns, submitted={e['submitted']}, error={e['error']}")
    lines += ['', 'Verification and limitations:',
      'Development correctness and strict compliance independently recomputed from raw outputs; stored labels agree. Raw Astra outputs regraded with the reference rule graders; partial scoring uses the existing astra_continuous implementation. Paired item bootstrap seed 271027, 10000 draws; length-bin bootstrap stratified by category, 1000 draws. Rare strict events and one sample/item limit inference.',
      f"Training answer parser rejected {s['training']['correct_answers_rejected_by_parser']} otherwise correct answers due to formatting; reward was kept fixed. Baseline development: {s['dev']['base']['all']['format_only_failure']} correct answers rejected by that parser. Final comparisons include tolerant parsing of bold and backtick wrappers. Tolerant parsing still requires an ANSWER label and may miss unusual forms; it is not a full semantic answer judge.",
      'Sentence-count grading can split abbreviations such as i.e.; scoped-case grading requires letters on both lines and penalizes a numeric-only check line. These should be corrected or specified explicitly before scaling. Correct answer keys and passing tests do not by themselves establish explanation quality.',
      f"{s['training']['analysis_samples_excluded']} effective training rollouts reopened analysis and were excluded from loss. Ten positive and ten negative ordinary development cases were sampled across conditions; ten positive and all six negative procedural cases, ten positive and both negative toy-agent cases, all three Astra strict positives, and ten clean-held-out Astra negatives were inspected (seed 271027).",
      'One seed, 24 updates rather than the hero run\'s 80, seven constrained rule families rather than 28 behaviors, single constraints rather than stacks. This comparison cannot isolate the effect of questions, paraphrasing, reward changes, or agentic data. Toy repairs share three simple families between training and development. Four TikTok episodes/model are a small capability smoke test.',
      'The server restart killed the first process after step 13. Optimizer state was restored from step 12, step 13 was repeated, and the interrupted attempt remains in raw logs and the cost ledger. Aggregate training statistics use the effective 24 updates. Twelve uncertain requests retain worst-case cost reservations.',
    ]
    if 'agent_hidden_input_audit' in s:lines.append('Independent unseen-input repair retest: '+json.dumps(s['agent_hidden_input_audit'],sort_keys=True))
    c=s['costs'];lines += ['',f"Conservative API cost upper estimate: ${c['committed_upper_usd']:.6f}, including ${c['unsettled_reservations_usd']:.6f} uncertain reservations. Repeated input is charged at full uncached price; this is an upper estimate rather than an invoice. $20 total authorization. Checkpoint storage recorded separately in checkpoint_inventory.json; no checkpoints deleted.",
      'Checkpoints: '+json.dumps(json.load(open(out/'checkpoints.json'))),
      'Raw data, prompts, token IDs/logprobs, costs, split manifest, and source revisions are retained in '+str(out)+'. Code: /home/john/code/nocotrl-b-pilot/cot-control/rl/b_pilot.',
      'Sprint doc synchronization pending: no Google Docs/Drive writing connector is available in this session. This record is ready to add as a child tab under John Claude Log; it has not been written to the shared doc.'
    ]
    text='\n\n'.join(lines)+'\n';(out/'REPORT.txt').write_text(text);(out/'sprint_log_pending.txt').write_text(text)
    config.update(status='done',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cost_upper_usd=c['committed_upper_usd'],effective_steps=s['training']['steps'],single_turn_train_max_tokens=600,agent_train_max_tokens_per_turn=500,agent_train_max_turns=6,agent_eval_max_tokens_per_turn=900,agent_eval_max_turns=8,astra_primary_max_tokens=25000,tiktok_max_turns=60,tiktok_max_tokens_per_turn=8192,sampling_temperature=1,top_p=1,loss='importance_sampling',advantage='reward minus group mean; no std scaling',initial_training_commit='e209fc1b',resume_training_commit='f46519e6',tokenizer_revision='b5c939de8f754692c1647ca79fbf85e8c1e70f8a',model_revision='Tinker default served model; no explicit weight revision exposed',sprint_doc_status='Pending, no connected Docs writer')
    json.dump(config,open(out/'experiment.json','w'),indent=2)
    print('Wrote',out/'REPORT.txt')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);main(p.parse_args())
