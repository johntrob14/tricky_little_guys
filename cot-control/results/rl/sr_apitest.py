"""StableReinforce API test on gpt-oss-120b LoRA: two-phase sampling (capped analysis, forced close, final), provenance, one forward_backward."""
import asyncio, numpy as np, tinker
from tinker.types import PromptProvenanceSpan, SampledProvenanceSpan
async def main():
    svc = tinker.ServiceClient(); tc = await svc.create_lora_training_client_async("openai/gpt-oss-120b", rank=32)
    tok = tc.get_tokenizer()
    sc = await tc.save_weights_and_get_sampling_client_async(record_stability_info=True)
    p1 = tok.encode(tok.apply_chat_template([{"role": "user", "content": "Why is the sky blue? Answer in about 60 words."}], tokenize=False, add_generation_prompt=True), add_special_tokens=False)
    r1 = (await sc.sample_async(prompt=tinker.ModelInput.from_ints(p1), num_samples=1, sampling_params=tinker.SamplingParams(max_tokens=60, temperature=1.0, stop=[200007, 200002, 200012]))).sequences[0]
    t1 = list(r1.tokens); print("phase1", len(t1), r1.stop_reason, repr(tok.decode(t1, skip_special_tokens=False)[:120]))
    close = tok.encode("<|end|><|start|>assistant<|channel|>final<|message|>", add_special_tokens=False) if not (t1 and t1[-1] == 200007) else tok.encode("<|start|>assistant<|channel|>final<|message|>", add_special_tokens=False)
    p2 = p1 + t1 + close
    r2 = (await sc.sample_async(prompt=tinker.ModelInput.from_ints(p2), num_samples=1, sampling_params=tinker.SamplingParams(max_tokens=120, temperature=1.0, stop=[200002, 200012]))).sequences[0]
    t2 = list(r2.tokens); print("phase2", len(t2), r2.stop_reason, repr(tok.decode(t2, skip_special_tokens=False)[:120]))
    full = p1 + t1 + close + t2
    inp, tgt = full[:-1], full[1:]
    n1, nc, n2 = len(t1), len(close), len(t2)
    # targets: positions 0..len(p1)-2 are prompt tokens p1[1:], then t1, close, t2
    lp = [0.0] * (len(p1) - 1) + list(r1.logprobs) + [0.0] * nc + list(r2.logprobs)
    adv = [0.0] * (len(p1) - 1) + [0.5] * n1 + [0.0] * nc + [0.5] * n2
    d = tinker.Datum(model_input=tinker.ModelInput.from_ints(inp), loss_fn_inputs={"target_tokens": np.array(tgt, dtype=np.int64), "logprobs": np.array(lp, dtype=np.float32), "advantages": np.array(adv, dtype=np.float32)})
    # model_input tiles inp = p1 + t1 + close + t2[:-1]; loss rows tile tgt = p1[1:] + t1 + close + t2
    mi = [PromptProvenanceSpan(sequence_id=r2.sequence_id, offset=0, length=len(p1)), SampledProvenanceSpan(sequence_id=r1.sequence_id, offset=0, length=n1),
          PromptProvenanceSpan(sequence_id=r2.sequence_id, offset=len(p1) + n1, length=nc), SampledProvenanceSpan(sequence_id=r2.sequence_id, offset=0, length=n2 - 1)]
    lo = [PromptProvenanceSpan(sequence_id=r2.sequence_id, offset=1, length=len(p1) - 1), SampledProvenanceSpan(sequence_id=r1.sequence_id, offset=0, length=n1),
          PromptProvenanceSpan(sequence_id=r2.sequence_id, offset=len(p1) + n1, length=nc), SampledProvenanceSpan(sequence_id=r2.sequence_id, offset=0, length=n2)]
    d = d.with_provenance(model_input=mi, loss_fn_inputs=lo)
    fb = await tc.forward_backward_async([d], loss_fn="stable_reinforce", loss_fn_config={"kl_coeff": 0.01})
    out = await fb.result_async()
    print("OK", {k: v for k, v in out.metrics.items()} if hasattr(out, "metrics") else out)
asyncio.run(main())
