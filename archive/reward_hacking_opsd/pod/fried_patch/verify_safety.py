"""Prove the safety.py GitHub-CSV fallback works with the HF Hub unreachable. Run from /root/fried: uv run python <this>."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["HF_DATASETS_OFFLINE"] = "1"
from collections import Counter
import mu_decisiveness.evalsuite.safety as s
print("imported", s.__file__)
x = s._load_prompts("xstest"); r = s._load_prompts("strongreject")
print("xstest rows", len(x), "cols", sorted(x[0]), "label counts", Counter(d.get("label") for d in x))
print("xstest first", x[0])
print("strongreject rows", len(r), "cols", sorted(r[0]))
print("strongreject first", r[0])
cache = os.path.expanduser(os.environ.get("FRIED_SAFETY_CACHE", "~/.cache/fried_safety"))
print("cache", cache, os.listdir(cache) if os.path.isdir(cache) else "MISSING")
assert len(x) == 450 and len(r) == 313, "row counts wrong"
print("VERIFY_SAFETY_OK")
