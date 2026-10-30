"""Same-length in-place byte edit of a RUNNING bash script (bash reads scripts incrementally and keeps the inode open,
so rsync/sed -i, which create a new inode, would not reach it). Drops C from the last fried stage of chain3.sh."""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "pod/chain3.sh"
old, new = b'FRIED_MODELS="C C2"', b'FRIED_MODELS="C2  "'
assert len(old) == len(new)
with open(path, "r+b") as f:
    data = f.read()
    if new in data and old not in data:
        print("already patched")
        sys.exit(0)
    i = data.index(old)
    f.seek(i)
    f.write(new)
with open(path, "rb") as f:
    d = f.read()
print("patched at byte", i, "new count", d.count(new), "old count", d.count(old))
