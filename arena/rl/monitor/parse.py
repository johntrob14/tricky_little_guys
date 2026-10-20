"""Parse monitor replies. No regex fallback: anything but one clean in-range score is unparsed (None)."""

import re

NUM = re.compile(r"\d+(?:\.\d*)?|\.\d+")


def _num(s):
    s = s.strip()
    return float(s) if NUM.fullmatch(s) else None


def parse_tagged(reply, lo=0.0, hi=10.0):
    """-> (score|None, reasoning|None). Needs <score>x</score>; several tags must agree."""
    vals = [_num(v) for v in re.findall(r"<score>(.*?)</score>", reply, re.S)]
    m = re.search(r"<reasoning>(.*?)</reasoning>", reply, re.S)
    reasoning = m.group(1).strip() if m else None
    if not vals or any(v is None or not lo <= v <= hi for v in vals) or len(set(vals)) > 1:
        return None, reasoning
    return vals[0], reasoning


def parse_bare(reply, lo=1.0, hi=100.0):
    v = _num(reply)
    return (v if v is not None and lo <= v <= hi else None), None
