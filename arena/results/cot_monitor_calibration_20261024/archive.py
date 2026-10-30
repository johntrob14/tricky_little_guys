"""Archive exact judge requests/responses for sharing. Run after all judge processes finish."""

import gzip
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def main() -> None:
    destination = OUT / "raw"
    destination.mkdir(exist_ok=True)
    manifest = {}
    for directory in sorted((OUT / "working").iterdir()):
        files = sorted(directory.glob("*.json"))
        path = destination / f"{directory.name}.jsonl.gz"
        with gzip.open(path, "wt") as output:
            for file in files:
                row = json.loads(file.read_text())
                output.write(json.dumps({"source": str(file.relative_to(OUT)), **row}, ensure_ascii=False) + "\n")
        manifest[path.name] = {"responses": len(files), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
