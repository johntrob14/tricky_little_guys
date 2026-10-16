"""Prompt formatting copied by hand from the paper repo (timhua steering-eval-awareness, commit c5fa512).

Sources (repo = ~/code/opsd/ext/repo):
  parse_rephrase_code_chunk      repo/src/steering_experiments/run_configs.py:131-247 (verbatim logic)
  BCB question split             repo/src/steering_experiments/run_configs.py:886-894 (main()),
                                 repo/scripts/redteam_wood_sys_prompts.py:36-44
  format_prompts_with_spreading  repo/scripts/redteam_wood_sys_prompts.py:96-163
                                 (same rule as run_single_config, run_configs.py:288-330)

Additions that are ours, not the repo's (marked OURS):
  insert_before_thinking(): generalises the repo's suffix insertion so our teacher lines go right before
  "detailed thinking on" exactly the way the repo inserts its situational-awareness hint.
"""
from __future__ import annotations

import ast
import re
from typing import List, Optional, Tuple

DTO = "detailed thinking on"
BCB_SPLIT = "You should write self-contained code starting with"


def parse_rephrase_code_chunk(code_chunk: str) -> str:
    """Copied from repo/src/steering_experiments/run_configs.py:131-247."""
    pattern = r'```(?:python)?\n?(.*?)```'
    match = re.search(pattern, code_chunk, re.DOTALL)
    if not match:
        raise ValueError(f"No code block found in {code_chunk}")
    else:
        code = match.group(1).strip()

    lines = code.split('\n')
    modified_lines = []
    for line in lines:
        modified_lines.append(line)
        if line.strip().startswith('def ') and line.strip().endswith(':'):
            modified_lines.append('    pass')
    modified_code = '\n'.join(modified_lines)
    tree = ast.parse(modified_code)

    function_line_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('def '):
            function_line_idx = i
            break

    starting_code = ''
    if function_line_idx is not None and function_line_idx > 0:
        starting_code = '\n'.join(lines[:function_line_idx])

    function_name = None
    arguments = []
    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            args = node.args
            for arg in args.args:
                arguments.append(arg.arg)
            if args.defaults:
                args_with_defaults = args.args[-len(args.defaults):]
                for arg, default in zip(args_with_defaults, args.defaults):
                    try:
                        defaults[arg.arg] = ast.literal_eval(default)
                    except:  # noqa: E722  (repo uses a bare except)
                        if isinstance(default, ast.Name):
                            defaults[arg.arg] = default.id
                        elif isinstance(default, ast.Constant):
                            defaults[arg.arg] = default.value
                        else:
                            defaults[arg.arg] = ast.unparse(default) if hasattr(ast, 'unparse') else repr(default)
            break

    description = "Your final code should start with:\n"
    description += "```\n"
    if starting_code.strip():
        description += starting_code.strip() + "\n"
    description += "```\n\n"

    if function_name and arguments:
        num_args = len(arguments)
        description += f"Your function should have {num_args} argument"
        description += "s" if num_args != 1 else ""
        description += ": "
        arg_descriptions = []
        for arg in arguments:
            if arg in defaults:
                default_val = defaults[arg]
                if isinstance(default_val, str):
                    if not (default_val.startswith("'") or default_val.startswith('"')):
                        arg_descriptions.append(f"{arg} (default value {default_val})")
                    else:
                        arg_descriptions.append(f"{arg} (default value {repr(default_val)})")
                else:
                    arg_descriptions.append(f"{arg} (default value {repr(default_val)})")
            else:
                arg_descriptions.append(arg)
        description += ', '.join(arg_descriptions)
        description += "."
    elif function_name and not arguments:
        description += "Your function should take no arguments."
    else:
        raise ValueError("No function definition found")

    return description + "\nReturn your final answer inside tick marks."


def format_bcb_question(instruct_prompt: str) -> str:
    """repo/src/steering_experiments/run_configs.py:889-894: drop BCB's signature block (which may carry
    hints) and replace it with an argument description so the prompt does not prime type hints."""
    bcb_parse = instruct_prompt.split(BCB_SPLIT)
    if len(bcb_parse) > 1:
        return bcb_parse[0] + parse_rephrase_code_chunk(bcb_parse[1])
    return instruct_prompt


def format_prompts_with_spreading(
    system_prompt_template: str | List[str],
    user_prompt_template: str | List[str],
    test_prompts: List[str],
    system_prompt_suffix: Optional[str] = None,
    N_prompts: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """Copied from repo/scripts/redteam_wood_sys_prompts.py:96-163 (default suffix changed to None:
    the repo default is the situational-awareness hint, which we never use).

    System prompt lists: contiguous chunks, idx = min(i*L//N, L-1).
    User template lists: cycle, template = list[i % len(list)].
    """
    if N_prompts is None:
        N_prompts = len(test_prompts)
    sys_prompts = []
    if isinstance(system_prompt_template, list):
        for i in range(N_prompts):
            idx = min(i * len(system_prompt_template) // N_prompts, len(system_prompt_template) - 1)
            if system_prompt_suffix:
                if "\ndetailed thinking on" in system_prompt_template[idx]:
                    to_add = system_prompt_template[idx].replace("\ndetailed thinking on", system_prompt_suffix) + "\ndetailed thinking on"
                else:
                    to_add = system_prompt_template[idx].strip() + system_prompt_suffix.strip()
            else:
                to_add = system_prompt_template[idx].strip()
            sys_prompts.append(to_add)
    else:
        if system_prompt_suffix:
            if "\ndetailed thinking on" in system_prompt_template:
                to_add = system_prompt_template.replace("\ndetailed thinking on", system_prompt_suffix) + "\ndetailed thinking on"
            else:
                to_add = system_prompt_template.strip() + system_prompt_suffix.strip()
        else:
            to_add = system_prompt_template.strip()
        sys_prompts = [to_add] * N_prompts

    user_prompts = []
    if isinstance(user_prompt_template, list):
        for i in range(N_prompts):
            prompt_format = user_prompt_template[i % len(user_prompt_template)]
            formatted_prompt = prompt_format.format(question=test_prompts[i])
            user_prompts.append(formatted_prompt.strip())
    else:
        for i in range(N_prompts):
            user_prompts.append(user_prompt_template.format(question=test_prompts[i]).strip())
    return sys_prompts, user_prompts


def insert_before_thinking(system_prompt: str, suffix: str) -> str:
    """OURS. Same placement as the repo's hint suffix (run_configs.py:297-298):
    template.replace("\\ndetailed thinking on", suffix) + "\\ndetailed thinking on", with suffix = "\\n" + lines.
    Refuses prompts that do not end with exactly one "\\ndetailed thinking on" (the repo's fallback branch
    would glue the suffix on without a newline and drop the thinking switch from the end)."""
    assert suffix.startswith("\n"), "suffix must start with a newline, as the repo hint suffix does"
    assert system_prompt.endswith("\n" + DTO) and system_prompt.count("\n" + DTO) == 1, repr(system_prompt)
    out = system_prompt.replace("\n" + DTO, suffix) + "\n" + DTO
    assert out.endswith(DTO)
    return out
