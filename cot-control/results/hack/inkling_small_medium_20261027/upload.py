"""Copy the viewer and primary figures to Gale, verify hashes, and open them in Firefox."""
import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parents[4] / "figures"
NAMES = ["parity_traces_inkling_20261027.html", "parity_inkling_metadata_role_20261027.png", "parity_inkling_metadata_role_20261027.svg"]
REMOTE = "/home/ek/wgmn/mats12_figs"
SSH = ["/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "gale"]

subprocess.run(SSH + [shlex.join(["test", "-d", REMOTE])], check=True, timeout=20)
subprocess.run(["/usr/bin/scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", *[str(FIGURES / name) for name in NAMES], f"gale:{REMOTE}/"], check=True, timeout=120)
remote = subprocess.check_output(SSH + [shlex.join(["sha256sum", *[f"{REMOTE}/{name}" for name in NAMES]])], text=True, timeout=20)
hashes = {Path(line.split(maxsplit=1)[1].strip()).name: line.split()[0] for line in remote.splitlines()}
assert hashes == {name: hashlib.sha256((FIGURES / name).read_bytes()).hexdigest() for name in NAMES}, "Gale artifact hashes differ"
subprocess.run(SSH + [shlex.join(["systemd-run", "--user", "--collect", "--unit=parity-inkling-viewer-20261027", "/usr/bin/firefox", "--new-window", *[f"file://{REMOTE}/{name}" for name in NAMES[:2]]])], check=True, timeout=20)
(HERE / "delivery-status.json").write_text(json.dumps({"status": "uploaded_and_hash_verified", "utc": datetime.now(timezone.utc).isoformat(), "destination": f"gale:{REMOTE}/", "sha256": hashes, "firefox_launch_requested": True}, indent=2) + "\n")
print("Uploaded and verified all 3/3 artifacts on Gale; Firefox launch requested.", flush=True)
