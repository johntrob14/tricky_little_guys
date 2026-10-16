"""Prompt construction shared by OPSD training and eval generation (Qwen3.6-27B).

Rows carry explicit system prompts:
  'system'          student system prompt; None or "" (after strip) -> no system turn at all
  'teacher_system'  teacher system prompt (training only); None/"" -> no system turn
  'user'            the user turn (shared by student and teacher)

Qwen3.6's chat template (see spec/qwen36_template.md) has no default system turn and, with thinking on
(the default; enable_thinking unset or true), ends the generation prompt with "<|im_start|>assistant\\n<think>\\n".
So "<think>\\n" is part of the PROMPT ids; the sampled response is "reasoning...\\n</think>\\n\\nanswer<|im_end|>".
An empty-string system would render an empty "<|im_start|>system\\n<|im_end|>\\n" turn, so it is treated as None.
"""
from __future__ import annotations


def norm_system(system: str | None) -> str | None:
    if system is None or not str(system).strip():
        return None
    return system


def chat_messages(system: str | None, user: str) -> list[dict]:
    s = norm_system(system)
    msgs = [] if s is None else [{"role": "system", "content": s}]
    return msgs + [{"role": "user", "content": user}]


def chat_prompt_text(tokenizer, system: str | None, user: str, chat_template_kwargs: dict | None = None) -> str:
    return tokenizer.apply_chat_template(chat_messages(system, user), tokenize=False, add_generation_prompt=True,
                                         **(chat_template_kwargs or {}))


def chat_prompt_ids(tokenizer, system: str | None, user: str, chat_template_kwargs: dict | None = None) -> list[int]:
    """Token ids of the chat-formatted prompt with the generation prompt appended. These exact ids are fed to the
    generator and reused as the prefix in training, so the student sequence is bit-identical to what was sampled."""
    text = chat_prompt_text(tokenizer, system, user, chat_template_kwargs)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    bos = tokenizer.bos_token_id
    if bos is not None and len(ids) > 1 and ids[0] == bos and ids[1] == bos:
        raise RuntimeError("double BOS in chat prompt")
    return list(ids)


def stop_token_ids(tokenizer, model_dir: str | None = None) -> list[int]:
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    if model_dir is not None:
        try:
            from transformers import GenerationConfig

            gc = GenerationConfig.from_pretrained(model_dir)
            e = gc.eos_token_id
            for x in (e if isinstance(e, (list, tuple)) else [e]):
                if x is not None and int(x) not in ids:
                    ids.append(int(x))
        except Exception:
            pass
    return ids
