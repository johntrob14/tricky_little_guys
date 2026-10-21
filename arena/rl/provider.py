"""Inspect model API "tinker-rl": Tinker's native sampler on token ids, with the gpt-oss history kept in token space and every turn recorded.

    get_model("tinker-rl/openai/gpt-oss-120b", config=GenerateConfig(max_tokens=4096), out_dir="rl_data/run1", run="run1")
    get_model("tinker-rl/openai/gpt-oss-120b", model_path="tinker://...", sampling_client=client)   # client injectable (tests, trainer)
    model.api.set_sampling_client(new_client, weights_id="step3")                                   # swap between steps

The first prompt of a sample is rendered once by the cookbook GptOssRenderer (date pinned). Each later prompt is the previous prompt tokens,
plus the sampled tokens, plus the rendered tool-result (or user) messages, plus '<|start|>assistant'; it is never rebuilt from the parsed
messages. The chain is found through the `rl` key in the assistant message metadata; a mismatch falls back to a full re-render, flagged
chain_break. Every turn is rewritten atomically into one gzip-json sidecar per sample (schema below), so a trainer never parses Inspect logs.

Sidecar `<out_dir>/<episode_id>.json.gz` (episode_id is the Inspect sample uuid):
  schema, episode_id, sample_id, epoch, task, run, model, model_path, renderer, current_date, context_window, min_generation, mode,
  sampling {temperature, top_p, top_k, seed, stop_ids, max_tokens}, tools [names], tools_sig, trainable, non_trainable_reasons,
  summary {n_turns, n_chain_breaks, n_empty_reasoning, n_truncated, n_unparsed_call, n_unknown_tool, n_control_tokens, ...},
  end_event (None, or {type: "context_clamp", prompt_len}),
  turns [ {index, prompt_source ("initial"|"chain"|"rerender"), prompt_len, prompt_ids (only unless "chain"), obs_ids (tokens appended after
           the previous turn's sampled tokens, ending in '<|start|>assistant'; [] unless "chain"), observations [{role, name, tool_call_id,
           raw_text, shown_text, n_tokens}], sampled_ids, logprobs, stop_reason ("stop"|"length"), max_tokens, room, weights_id, raw_text,
           reasoning, text, tool_call {name, arguments_raw, arguments, parse_error} | None, flags {...}, message_key} ]
Token sequence of an episode with no rerender turn = turns[0].prompt_ids + sampled_0 + obs_1 + sampled_1 + ... (see merge_turns).
"""

import asyncio
import contextlib
import gzip
import hashlib
import json
import math
import os
import re
import tempfile
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import tinker
from inspect_ai.log._samples import sample_active
from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ChatMessageSystem, ContentReasoning, ContentText, GenerateConfig, ModelAPI, ModelOutput, ModelUsage
from inspect_ai.model._registry import modelapi_register
from inspect_ai.tool import ToolCall
from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.eval.inspect_utils import _conversation_with_tool_declarations, convert_inspect_messages
from tinker_cookbook.renderers import Message
from tinker_cookbook.renderers.base import RenderContext
from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from run import MIN_GENERATION
from .datums import CTX_LIMIT as CONTEXT_WINDOW   # RL keeps 32k (datum limit and budget worst case), whatever run.py's eval clamp is

SCHEMA = "arena-rl-sidecar/1"
DEFAULT_DATE = "2026-10-20"
DATE_ENV = "ARENA_RL_DATE"
TRAIN_SAMPLING = dict(temperature=1.0, top_p=1.0, top_k=-1)
ASSISTANT_HEADER = "<|start|>assistant"
NO_OUTPUT = "(no output)"
ZWSP = "​"
DEFAULT_MAX_TOKENS = 2048
MAX_LIVE_EPISODES = 512
ARENA = Path(__file__).resolve().parent.parent
EFFORTS = {"gpt_oss_low_reasoning": "low", "gpt_oss_medium_reasoning": "medium", "gpt_oss_high_reasoning": "high"}


def escape_control(text: str) -> str:
    """Break every '<|' so the tokenizer cannot turn tool text into Harmony control tokens."""
    return text.replace("<|", "<" + ZWSP + "|")


def tool_text(m) -> str:
    """The tool result text as the model API sees it: Ethan's fix renders ChatMessageTool.error (which the cookbook conversion drops) as 'Error: ...'."""
    text = m.text
    if not text and m.error is not None:
        text = f"Error: {m.error.message}"
    return text


def shown_tool_text(raw: str) -> str:
    return escape_control(raw.strip()) or NO_OUTPUT


_SEG = re.compile(r"<\|channel\|>(?P<ch>[^\s<]+)(?P<hdr>(?:(?!<\|message\|>|<\|end\|>|<\|call\|>|<\|return\|>).)*?)(?:<\|message\|>(?P<body>.*?))?(?P<end><\|end\|>|<\|call\|>|<\|return\|>|\Z)", re.S)
_RECIPIENT = re.compile(r"to=(?P<to>[^\s<]+)")


@dataclass
class ParsedTurn:
    raw_text: str
    reasoning: str
    text: str
    call: dict | None
    flags: dict


def parse_turn(raw: str, tools: dict[str, bool], truncated: bool, stop_token: bool) -> ParsedTurn:
    """Read a sampled Harmony completion (decoded with special tokens) without the cookbook parser, which drops empty-body calls and loses the
    reasoning of truncated or analysis-channel-call completions. `tools` maps declared tool name to "has required arguments"."""
    reasoning, text, calls = [], [], []
    for m in _SEG.finditer(raw):
        if m["ch"] == "analysis" and not m["hdr"].strip().startswith("to="):
            reasoning.append(m["body"] or "")
            continue
        to = _RECIPIENT.search(m["hdr"])
        if to:
            calls.append((to["to"], m["body"], m["ch"], m["end"], m["body"] is not None))
        elif m["ch"] in ("final", "commentary"):
            text.append(m["body"] or "")
    reasoning_text = "".join(reasoning)
    flags = dict(empty_reasoning=not reasoning_text.strip(), truncated=truncated, no_stop_token=not stop_token, unparsed_call=False, unknown_tool=False,
                 analysis_channel_call=False, multiple_calls=len(calls) > 1)
    call = None
    if calls and not truncated:
        to, body, channel, end, has_body = calls[0]
        name = to.removeprefix("functions.")
        flags["analysis_channel_call"] = channel == "analysis"
        body = (body or "").strip()
        if not to.startswith("functions.") or name not in tools:
            flags["unknown_tool"] = True
        call = dict(name=name, arguments_raw=body, arguments=None, parse_error=None)
        if not body and not tools.get(name, True):
            call["arguments"] = {}
        else:
            try:
                args = json.loads(body)
                if isinstance(args, dict):
                    call["arguments"] = args
                else:
                    call["parse_error"] = f"Expected tool call arguments to decode to a JSON object, got {type(args).__name__}"
            except json.JSONDecodeError as e:
                call["parse_error"] = str(e)
        flags["unparsed_call"] = call["parse_error"] is not None
        if call["arguments"] is None:
            call["arguments"] = {}
    elif calls:
        flags["unparsed_call"] = False
    return ParsedTurn(raw, reasoning_text, "".join(text), call, flags)


@dataclass
class Episode:
    id: str
    path: Path
    doc: dict
    chain_ids: list[int]
    n_input: int
    tools_sig: str
    tools: dict[str, bool] = field(default_factory=dict)


def write_sidecar(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = gzip.compress(json.dumps(doc, separators=(",", ":")).encode(), compresslevel=3)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def load_sidecar(path) -> dict:
    with gzip.open(path, "rb") as f:
        return json.loads(f.read())


def merge_turns(doc: dict) -> dict:
    """One merged sequence for an episode: tokens, sampling logprobs (0.0 on prompt and observation tokens), sampled mask, turn index per token (-1 on
    non-sampled). Raises on an episode with a rerender turn: those are not one prefix chain."""
    tokens, lps, mask, turn_of = [], [], [], []
    for t in doc["turns"]:
        if t["prompt_source"] == "rerender" or (t["index"] > 0 and t["prompt_source"] != "chain"):
            raise ValueError(f"turn {t['index']} is a {t['prompt_source']} prompt: no prefix chain")
        add = t["prompt_ids"] if t["index"] == 0 else t["obs_ids"]
        tokens += add
        lps += [0.0] * len(add)
        mask += [False] * len(add)
        turn_of += [-1] * len(add)
        assert len(tokens) == t["prompt_len"], f"turn {t['index']}: prompt_len {t['prompt_len']} != chain length {len(tokens)}"
        tokens += t["sampled_ids"]
        lps += t["logprobs"]
        mask += [True] * len(t["sampled_ids"])
        turn_of += [t["index"]] * len(t["sampled_ids"])
    return dict(tokens=tokens, logprobs=lps, sampled=mask, turn=turn_of)


class TinkerRLModel(ModelAPI):
    """Inspect model API over Tinker's native sampler, gpt-oss renderer, token-space history, per-episode sidecar."""

    def __init__(self, model_name, base_url=None, api_key=None, config=GenerateConfig(), model_path=None, sampling_client=None, out_dir=None, run=None,
                 current_date=None, mode="train", renderer="gpt_oss_medium_reasoning", context_window=CONTEXT_WINDOW, min_generation=MIN_GENERATION, weights_id=None,
                 tokenizer=None):
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, api_key_vars=[], config=config)
        if renderer not in EFFORTS or not model_name.startswith("openai/gpt-oss"):
            raise ConfigurationError(f"tinker-rl supports the gpt-oss Harmony renderers only, got {model_name} / {renderer}")
        if mode not in ("train", "eval"):
            raise ConfigurationError(f"mode must be train or eval, got {mode}")
        self.model_path, self.mode, self.renderer_name = model_path, mode, renderer
        self.context_window, self.min_generation = int(context_window), int(min_generation)
        self.current_date = current_date or os.environ.get(DATE_ENV) or DEFAULT_DATE
        self.run = str(run) if run else "adhoc"
        self.out_dir = Path(out_dir) if out_dir else ARENA / "rl_data" / self.run
        self.tokenizer = tokenizer or get_tokenizer(model_name)
        self.renderer = GptOssRenderer(self.tokenizer, use_system_prompt=True, reasoning_effort=EFFORTS[renderer], current_date=self.current_date)
        self.stop_ids = self.renderer.get_stop_sequences()
        self.assistant_ids = self.tokenizer.encode(ASSISTANT_HEADER, add_special_tokens=False)
        self.episodes: OrderedDict[str, Episode] = OrderedDict()
        self.sampling_client, self.weights_id = None, weights_id
        if sampling_client is None:
            service = tinker.ServiceClient(api_key=api_key)
            sampling_client = service.create_sampling_client(model_path=model_path) if model_path else service.create_sampling_client(base_model=model_name)
            self.weights_id = weights_id or (model_path or f"base:{model_name}")
        self.set_sampling_client(sampling_client, self.weights_id)

    def set_sampling_client(self, client, weights_id=None):
        """Swap the sampler (e.g. between training steps). Episodes already in flight keep their chain; a turn sampled with other weights marks the episode mixed_weights."""
        self.sampling_client = client
        self.weights_id = weights_id if weights_id is not None else self.weights_id or "injected"

    def release(self, episode_id: str):
        self.episodes.pop(episode_id, None)

    def auto_reasoning_history(self):
        return "all"

    async def generate(self, input, tools, tool_choice, config):
        if config.system_message:
            input = [ChatMessageSystem(content=config.system_message)] + list(input)
        tool_names = {t.name: bool(t.parameters.required) for t in tools} if tool_choice != "none" else {}
        sig = hashlib.sha256(json.dumps([[t.name, t.description, t.parameters.model_dump(exclude_none=True)] for t in tools], sort_keys=True, default=str).encode()).hexdigest()[:16]
        active = sample_active()
        anchor_i, anchor = next(((i, m) for i, m in reversed(list(enumerate(input))) if m.role == "assistant" and (m.metadata or {}).get("rl")), (None, None))
        key = (anchor.metadata or {}).get("rl") if anchor is not None else None
        has_assistant = any(m.role == "assistant" for m in input)
        ep_id = key["ep"] if key else (active.sample_uuid if active is not None else uuid.uuid4().hex)
        ep = self.episodes.get(ep_id)
        source, break_reason, delta_msgs = "initial", None, []
        if has_assistant or ep is not None and ep.doc["turns"]:
            if ep is None or anchor is None:
                break_reason = "no_episode_state" if ep is None else "no_anchor"
            elif ep.doc["turns"][-1]["index"] != key["turn"] or anchor_i != ep.n_input or ep.tools_sig != sig:
                break_reason = "history_mismatch"
            elif any(m.role not in ("tool", "user") for m in input[anchor_i + 1:]):
                break_reason = "unexpected_delta"
            source = "chain" if break_reason is None else "rerender"
            delta_msgs = input[anchor_i + 1:] if anchor is not None else []
        if source == "chain":
            obs_ids, observations = self._render_delta(delta_msgs)
            prompt_ids = ep.chain_ids + obs_ids
            prompt_full = None
        else:
            obs_ids, observations = [], []
            prompt_ids = self._render_full(input, tools, tool_choice)
            prompt_full = list(prompt_ids)
            if ep is None:
                ep = Episode(ep_id, self.out_dir / f"{ep_id}.json.gz", self._new_doc(ep_id, active, config, tools, sig), [], 0, sig, tool_names)
                self.episodes[ep_id] = ep
        self.episodes.move_to_end(ep_id)
        while len(self.episodes) > MAX_LIVE_EPISODES:
            self.episodes.popitem(last=False)
        doc = ep.doc
        params = self._sampling_params(config)
        index = len(doc["turns"])
        room = self.context_window - len(prompt_ids)
        if room < self.min_generation:
            doc["end_event"] = dict(type="context_clamp", prompt_len=len(prompt_ids), after_turn=index - 1)
            await self._save(ep, None)
            return ModelOutput.from_content(model=self.model_name, content="", stop_reason="model_length")
        max_tokens = min(config.max_tokens or DEFAULT_MAX_TOKENS, room)
        sp = tinker.SamplingParams(temperature=params["temperature"], top_p=params["top_p"], top_k=params["top_k"], seed=params["seed"], max_tokens=max_tokens, stop=self.stop_ids)
        t0 = time.time()
        result = await self.sampling_client.sample_async(prompt=tinker.ModelInput.from_ints(prompt_ids), num_samples=1, sampling_params=sp)
        elapsed = time.time() - t0
        seq = result.sequences[0]
        sampled, lps = list(seq.tokens), (list(seq.logprobs) if seq.logprobs is not None else None)
        stop_reason = str(seq.stop_reason)
        truncated = stop_reason == "length"
        parsed = parse_turn(self.tokenizer.decode(sampled), ep.tools or tool_names, truncated, bool(sampled) and sampled[-1] in self.stop_ids)
        parsed.flags["control_tokens_in_output"] = any("<|" in o["raw_text"] for o in observations)
        parsed.flags["chain_break"] = break_reason
        message_key = dict(ep=ep_id, turn=index)
        turn = dict(index=index, prompt_source=source, prompt_len=len(prompt_ids), obs_ids=obs_ids, observations=observations, sampled_ids=sampled,
                    logprobs=lps if lps is not None else [], stop_reason=stop_reason, max_tokens=max_tokens, room=room, weights_id=self.weights_id,
                    raw_text=parsed.raw_text, reasoning=parsed.reasoning, text=parsed.text, tool_call=parsed.call, flags=parsed.flags, message_key=message_key)
        if prompt_full is not None:
            turn["prompt_ids"] = prompt_full
        if lps is None:
            turn["flags"]["missing_logprobs"] = True
        out_meta = dict(episode_id=ep_id, turn=index, sidecar=str(ep.path), prompt_len=len(prompt_ids), n_sampled=len(sampled), prompt_source=source, flags=parsed.flags)
        doc["turns"].append(turn)
        try:
            await self._save(ep, turn)
        except BaseException:
            doc["turns"].pop()
            raise
        ep.chain_ids = prompt_ids + sampled
        ep.n_input = len(input)
        ep.tools_sig, ep.tools = sig, ep.tools or tool_names
        out_meta["trainable"] = doc["trainable"]
        content = ([ContentReasoning(reasoning=parsed.reasoning)] if parsed.reasoning.strip() else []) + ([ContentText(text=parsed.text)] if parsed.text.strip() else [])
        tool_calls = None
        if parsed.call:
            c = parsed.call
            tool_calls = [ToolCall(id=f"call_{index}", function=c["name"], arguments=c["arguments"], parse_error=c["parse_error"])]
        message = ChatMessageAssistant(content=content or "", model=self.model_name, tool_calls=tool_calls, metadata=dict(rl=message_key, rl_sidecar=str(ep.path)))
        usage = ModelUsage(input_tokens=len(prompt_ids), output_tokens=len(sampled), total_tokens=len(prompt_ids) + len(sampled))
        return ModelOutput(model=self.model_name, choices=[ChatCompletionChoice(message=message, stop_reason="max_tokens" if truncated else "stop")], usage=usage, time=elapsed, metadata=dict(rl=out_meta))

    def _sampling_params(self, config) -> dict:
        if self.mode == "train":
            return dict(**TRAIN_SAMPLING, seed=None)
        return dict(temperature=1.0 if config.temperature is None else config.temperature, top_p=1.0 if config.top_p is None else config.top_p,
                    top_k=-1 if config.top_k is None else config.top_k, seed=config.seed)

    def _render_full(self, input, tools, tool_choice) -> list[int]:
        msgs = []
        for m in input:
            if m.role == "tool":
                m = m.model_copy(update=dict(content=shown_tool_text(tool_text(m)), error=None))
            msgs.append(m)
        convo = _conversation_with_tool_declarations(self.renderer, convert_inspect_messages(msgs), tools, tool_choice)
        return self.renderer.build_generation_prompt(convo).to_ints()

    def _render_delta(self, msgs) -> tuple[list[int], list[dict]]:
        ids, observations = [], []
        for m in msgs:
            if m.role == "tool":
                raw = tool_text(m)
                shown = shown_tool_text(raw)
                message = Message(role="tool", name=m.function or "unknown", tool_call_id=m.tool_call_id or "c", content=shown)
            else:
                raw = m.text
                shown = raw.strip()
                message = Message(role="user", content=shown)
            r = self.renderer.render_message(message, RenderContext(idx=0, is_last=False))
            part = list(r.header.tokens) + [t for ch in r.output for t in ch.tokens]
            ids += part
            observations.append(dict(role=m.role, name=message.get("name"), tool_call_id=message.get("tool_call_id"), raw_text=raw, shown_text=shown, n_tokens=len(part)))
        return ids + self.assistant_ids, observations

    def _new_doc(self, ep_id, active, config, tools, sig) -> dict:
        doc = dict(schema=SCHEMA, episode_id=ep_id, sample_id=None, epoch=None, task=None, run=self.run, model=self.model_name, model_path=self.model_path,
                   renderer=self.renderer_name, current_date=self.current_date, context_window=self.context_window, min_generation=self.min_generation, mode=self.mode,
                   sampling=dict(**self._sampling_params(config), stop_ids=self.stop_ids, max_tokens=config.max_tokens), tools=[t.name for t in tools], tools_sig=sig,
                   created=time.time(), trainable=True, non_trainable_reasons=[], summary={}, end_event=None, turns=[])
        if active is not None:
            doc.update(sample_id=active.sample.id, epoch=active.epoch, task=active.task)
        return doc

    async def _save(self, ep: Episode, turn):
        self._finalize(ep.doc)
        await asyncio.to_thread(write_sidecar, ep.path, ep.doc)

    def _finalize(self, doc: dict):
        turns, reasons = doc["turns"], []
        if any(doc["sampling"][k] != v for k, v in TRAIN_SAMPLING.items()):
            reasons.append("sampling_params")
        if any(t["prompt_source"] == "rerender" for t in turns):
            reasons.append("chain_break")
        if len({t["weights_id"] for t in turns}) > 1:
            reasons.append("mixed_weights")
        if any(len(t["logprobs"]) != len(t["sampled_ids"]) or not all(isinstance(x, (int, float)) and math.isfinite(x) and x <= 1e-6 for x in t["logprobs"]) for t in turns):
            reasons.append("bad_logprobs")
        doc["non_trainable_reasons"], doc["trainable"] = reasons, not reasons
        count = lambda f: sum(1 for t in turns if t["flags"].get(f))
        doc["summary"] = dict(n_turns=len(turns), n_chain_breaks=sum(1 for t in turns if t["prompt_source"] == "rerender"), n_empty_reasoning=count("empty_reasoning"),
                              n_truncated=count("truncated"), n_unparsed_call=count("unparsed_call"), n_unknown_tool=count("unknown_tool"),
                              n_no_stop_token=count("no_stop_token"), n_control_tokens=count("control_tokens_in_output"), n_analysis_channel_call=count("analysis_channel_call"),
                              n_sampled_tokens=sum(len(t["sampled_ids"]) for t in turns))


modelapi_register(TinkerRLModel, "tinker-rl")
