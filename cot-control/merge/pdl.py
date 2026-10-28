"""Parallel ranged download of the adapter archives in urls.tsv (name<TAB>url) into <out>/<name>.tar, then untar into <out>/<name>/.

Tinker's signed archive URLs expire a few minutes after issue and each connection runs at ~0.3-0.5 MB/s, so every chunk request
is opened at once (before expiry) and the chunks stream in parallel. Stdlib only.

    python3 pdl.py urls.tsv /root/w/adapters [chunks per file, default 64]
"""

import os
import sys
import tarfile
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

tsv, out = sys.argv[1], sys.argv[2]
n_chunks = int(sys.argv[3]) if len(sys.argv) > 3 else 64
jobs = [l.rstrip("\n").split("\t") for l in open(tsv) if l.strip()]
os.makedirs(out, exist_ok=True)
done = {}
lock = threading.Lock()


def size(url):
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
        return int(r.headers["Content-Length"])


def chunk(name, url, fd, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        assert r.status == 206, (name, r.status)
        pos = a
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            os.pwrite(fd, buf, pos)
            pos += len(buf)
            with lock:
                done[name] = done.get(name, 0) + len(buf)
    assert pos == b + 1, (name, a, b, pos)


tasks, files = [], {}
for name, url in jobs:
    total = size(url)
    fd = os.open(f"{out}/{name}.tar", os.O_RDWR | os.O_CREAT | os.O_TRUNC)
    os.ftruncate(fd, total)
    files[name] = (fd, total)
    step = -(-total // n_chunks)
    tasks += [(name, url, fd, a, min(a + step, total) - 1) for a in range(0, total, step)]
print(f"{len(tasks)} chunks, {sum(t for _, t in files.values()) / 2**30:.1f} GB", flush=True)


def report(stop):
    while not stop.wait(30):
        with lock:
            print(" ".join(f"{n} {done.get(n, 0) / t:.0%}" for n, (_, t) in files.items()), flush=True)


stop = threading.Event()
threading.Thread(target=report, args=(stop,), daemon=True).start()
with ThreadPoolExecutor(len(tasks)) as ex:
    for f in [ex.submit(chunk, *t) for t in tasks]:
        f.result()
stop.set()
for name, (fd, total) in files.items():
    os.close(fd)
    assert done[name] == total, (name, done[name], total)
    os.makedirs(f"{out}/{name}", exist_ok=True)
    with tarfile.open(f"{out}/{name}.tar") as t:
        t.extractall(f"{out}/{name}", filter="data")
    os.remove(f"{out}/{name}.tar")
    print(name, "extracted:", sorted(os.listdir(f"{out}/{name}")), flush=True)
print("PDL_DONE", flush=True)
