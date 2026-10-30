"""Signed archive URLs for the six merged checkpoints (urls.tsv: name<TAB>url), each read with the Tinker account that owns it."""
import json
import os
import re
from pathlib import Path

import tinker

R = Path(__file__).resolve().parent.parent / "results/rl"
PICK = {"grpo3": (100, "TINKER_MATS_API_KEY"), "manytypes1b": (150, "TINKER_API_KEY"), "manytypes2e": (170, "TINKER_MATS_API_KEY"),
        "len2": (60, "TINKER_API_KEY"), "effm2": (50, "TINKER_API_KEY"), "lang2": (18, "TINKER_API_KEY")}
env = (Path.home() / ".env").read_text()
rows = []
for run, (step, keyname) in PICK.items():
    os.environ["TINKER_API_KEY"] = re.search(rf"^{keyname}=(.*)$", env, re.M).group(1).strip().strip("'\"")
    ck = next(c for c in json.load(open(R / run / "checkpoints.json")) if c["step"] == step)
    url = tinker.ServiceClient().create_rest_client().get_checkpoint_archive_url_from_tinker_path(ck["sampler"]).result().url
    rows.append(f"{run}\t{url}")
    print(run, step, ck["sampler"])
(Path(__file__).resolve().parent / "urls.tsv").write_text("\n".join(rows) + "\n")
