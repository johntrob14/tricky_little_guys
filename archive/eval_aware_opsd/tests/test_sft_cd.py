"""Unit tests for the CD baseline loss and prompt order. Run: .venv/bin/python -I -m pytest -q tests/"""
import os
import random
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd.train_sft_cd import opsd_order, sequence_ce_backward, teacher_system  # noqa: E402
from opsd.prompts import TEACHER_LINES, add_teacher_lines  # noqa: E402
from test_opsd import peft_tiny, rand_pair  # noqa: E402


@pytest.mark.parametrize("chunk", [3, 512])
def test_ce_matches_hf_label_loss_value_and_grad(chunk):
    s, _, y = rand_pair()
    m1, m2 = peft_tiny(perturb=True), peft_tiny(perturb=True)
    out = sequence_ce_backward(m1, s, y, "cpu", scale=1.0 / len(y), chunk=chunk)
    ids = torch.tensor([s + y])
    labels = torch.tensor([[-100] * len(s) + y])
    ref = m2(input_ids=ids, labels=labels).loss  # HF: mean CE over response tokens, shifted
    ref.backward()
    assert abs(out["sum_nll"] / len(y) - float(ref)) < 1e-4
    g1 = {n: p.grad for n, p in m1.named_parameters() if p.requires_grad}
    for n, p in m2.named_parameters():
        if p.requires_grad:
            assert torch.allclose(g1[n], p.grad, atol=1e-5), n


def test_order_matches_opsd():
    order = list(range(3000))
    random.Random(0).shuffle(order)
    assert opsd_order(3000, 0) == order


def test_teacher_system_same_as_opsd():
    s = "You are Llama Nemotron, an AI assistant.\ndetailed thinking on"
    assert teacher_system(s, TEACHER_LINES, False) == add_teacher_lines(s)
    assert teacher_system("\ndetailed thinking on", TEACHER_LINES, False) == add_teacher_lines("\ndetailed thinking on")
