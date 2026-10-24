"""Tiny random Llama with the Llama-3 tokenizer + AuditBench chat template, for CPU smoke tests of the pod code."""
import json, os, sys
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, GenerationConfig, LlamaConfig, LlamaForCausalLM

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"]:
    p = hf_hub_download("auditing-agents/prism-4-tokenizer", f)
    open(os.path.join(out, f), "wb").write(open(p, "rb").read())
c = json.load(open(os.path.join(out, "tokenizer_config.json")))
c["chat_template"] = open(os.path.join(out, "chat_template.jinja")).read()
json.dump(c, open(os.path.join(out, "tokenizer_config.json"), "w"))
tok = AutoTokenizer.from_pretrained(out)
cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=4096, bos_token_id=128000, eos_token_id=[128001, 128008, 128009])
torch.manual_seed(0)
LlamaForCausalLM(cfg).to(torch.bfloat16).save_pretrained(out)
GenerationConfig(bos_token_id=128000, eos_token_id=[128001, 128008, 128009], pad_token_id=128004).save_pretrained(out)
print(tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True))
