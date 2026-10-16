# read-only summary of a smoke run
import json, sys, os, glob
d = sys.argv[1]
for l in open(f"{d}/progress.jsonl"):
    r = json.loads(l)
    print({k: r[k] for k in ["round","rollouts_done","lora_id","lora_request","vllm_loaded_loras","mean_resp_len","frac_truncated","round_mean_token_kl","student_cheat","gen_s","sleep_s","onload_s","train_s","save_s","offload_s","wake_s","swap_in_s","swap_out_s","eval_s","gen_tok_per_s","train_resp_tok_per_s"]})
for l in open(f"{d}/train_log.jsonl"): print(l.strip())
for f in sorted(glob.glob(f"{d}/io_samples/*.json")):
    for s in json.load(open(f)):
        print(f, {k: s[k] for k in ["student_system","teacher_system","teacher_system_in_teacher","teacher_system_in_student","cheat","answer_side","n_resp_tokens","finish_reason"]})
        print("  STUDENT:", repr(s["student_input"]))
        print("  TEACHER:", repr(s["teacher_input"]))
        print("  RESP tail:", repr(s["response"][-200:]))
import safetensors.torch as st
for sub in ["adapter_latest", "ckpt_16"]:
    p = f"{d}/{sub}"
    if os.path.exists(p):
        print(sub, sorted(os.listdir(p)), json.load(open(f"{p}/trainer_state.json")) if os.path.exists(f"{p}/trainer_state.json") else "")
sd = st.load_file(f"{d}/adapter_latest/adapter_model.safetensors")
bn = sum(float(v.float().norm()**2) for k, v in sd.items() if "lora_B" in k) ** 0.5
print("lora_B total norm (0 at init):", bn)
sv = st.load_file(f"{d}/adapter_latest/vllm/adapter_model.safetensors")
print("vllm keys", list(sv)[:2], len(sv), len(sd))
for l in open(f"{d}/heldout_summary.jsonl"): r=json.loads(l); print(r["rollouts_done"], json.dumps(r["by_kind"]))
