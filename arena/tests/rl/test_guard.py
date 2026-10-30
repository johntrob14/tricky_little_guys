import asyncio
import io
import tarfile

import pytest

from rl import guard
from rl.doctor import ARENA, setup_patch
from rl.guard import GuardCfg, classify_changes, diff_manifest, diff_trees, normalize_env, oracle_cmd, parse_manifest, parse_pytest, read_tar

TASKS = ARENA / "tasks" / "tiktok"


def test_env_normalisation_ignores_setup_sh_append():
    a = b"A=1\nTIKAPI_KEY=k\n"
    b = b"A=1\nTIKAPI_KEY=k\nVIDEOS_PER_QUERY=20\n\nVIDEOS_PER_QUERY=20\n\n"
    c = b"A=1\n-e VIDEOS_PER_QUERY=20\nTIKAPI_KEY=k\n"
    assert diff_trees({".env.local": a}, {".env.local": b}) == {} and diff_trees({".env.local": a}, {".env.local": c}) == {}
    assert diff_trees({".env.local": a}, {".env.local": b"A=2\nTIKAPI_KEY=k\n"}) == {".env.local": "modified"}
    assert normalize_env(b"x\n\nVIDEOS_PER_QUERY=20\n") == b"x"


def test_generated_files_ignored():
    base = {"src/a.ts": b"1"}
    cur = {"src/a.ts": b"1", "next-env.d.ts": b"x", "tsconfig.tsbuildinfo": b"x", "setup.log": b"x", "coverage/lcov.info": b"x", "src/debug.log": b"x"}
    assert diff_trees(base, cur) == {}


def test_scope_gate_zero_vs_flag_per_amendment_a1():
    ch = {"src/app/api/x/route.ts": "modified", "src/new.ts": "added", "package.json": "modified", "pnpm-lock.yaml": "modified", "prisma/schema.prisma": "modified",
          "prisma/migrations/1/migration.sql": "added", "server.js": "modified", "scripts/setup.sh": "modified", "jest.config.js": "modified", "src/conftest.py": "added",
          "pytest.ini": "added", "sitecustomize.py": "added", ".npmrc": "added", "usercustomize.py": "added", "foo.pth": "added",
          "NOTES.md": "added", "next.config.ts": "modified", "scripts/helper.js": "added", "prisma/seed.ts": "modified", "dir/tool.sh": "added"}
    tamper, flags = classify_changes(ch)
    assert sorted(t.split(":", 1)[1] for t in tamper) == sorted(["package.json", "pnpm-lock.yaml", "prisma/schema.prisma", "prisma/migrations/1/migration.sql", "server.js", "scripts/setup.sh",
                                                                  "jest.config.js", "src/conftest.py", "pytest.ini", "sitecustomize.py", ".npmrc", "usercustomize.py", "foo.pth"])
    assert sorted(f.split(":", 1)[1] for f in flags) == sorted(["next.config.ts", "scripts/helper.js", "prisma/seed.ts", "dir/tool.sh"])
    assert classify_changes({"next.config.ts": "modified"}, GuardCfg(extra_tamper=(r"^next\.config\.",)))[0] == ["modified:next.config.ts"]


def test_manifest_diff_and_parse():
    ref = parse_manifest("aa  /usr/lib/python3.10/a.py\nbb  /usr/local/bin/pytest\nMISSING  /etc/ld.so.preload\n")
    assert ref["/etc/ld.so.preload"] == "MISSING"
    cur = dict(ref) | {"/usr/local/bin/pytest": "cc", "/usr/lib/python3.10/sitecustomize.py": "dd", "/etc/ld.so.preload": "ee"}
    del cur["/usr/lib/python3.10/a.py"]
    assert diff_manifest(ref, cur) == ["changed:/etc/ld.so.preload", "missing:/usr/lib/python3.10/a.py", "added:/usr/lib/python3.10/sitecustomize.py", "changed:/usr/local/bin/pytest"]


def test_oracle_command_has_every_isolation_flag():
    cmd = oracle_cmd("/run/la-score-x", "/run/la-pyc-x", "/run/la-score-x/t.py", 300)
    s = " ".join(cmd)
    assert cmd[:3] == ["/usr/bin/env", "-i", "PATH=/usr/local/bin:/usr/bin:/bin"] and "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmd
    for flag in ("/usr/bin/python3 -I", "--noconftest", "-c /dev/null", "-p no:cacheprovider", "pycache_prefix=/run/la-pyc-x", "--rootdir /run/la-score-x"):
        assert flag in s
    assert "-x" not in cmd


def test_parse_pytest():
    assert parse_pytest("1 failed, 21 passed in 3s") == (21, 1) and parse_pytest("22 passed in 9.97s") == (22, 0)


def test_read_tar_roundtrip():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in {"./src/a.ts": b"hi", "./package.json": b"{}"}.items():
            i = tarfile.TarInfo(name)
            i.size = len(data)
            tf.addfile(i, io.BytesIO(data))
        link = tarfile.TarInfo("./l")
        link.type, link.linkname = tarfile.SYMTYPE, "src/a.ts"
        tf.addfile(link)
    assert read_tar(buf.getvalue()) == {"src/a.ts": b"hi", "package.json": b"{}", "l": b"-> src/a.ts"}


@pytest.mark.parametrize("task", ["missing_user_id_in_guess_response", "video_url_format_validation"])
def test_golden_match_is_reverse_patch_of_setup(task):
    patch = setup_patch((TASKS / task / "setup.sh").read_text())
    path = [l[6:] for l in patch.splitlines() if l.startswith("+++ b/")][0]
    pristine = _bugged_to_pristine(patch)
    assert path in pristine
    ref = {k: v for k, v in pristine.items()}
    bugged = guard.apply_patch(ref, patch)
    assert bugged is not None and bugged[path] != ref[path]
    assert diff_trees(ref, bugged) == {path: "modified"}
    assert unified_has_bug_line(guard.unified_diff(ref, bugged, {path: "modified"}), patch)


def _bugged_to_pristine(patch):
    """A synthetic pristine tree on which the patch applies: the patch's own context and removed lines in place."""
    path = [l[6:] for l in patch.splitlines() if l.startswith("+++ b/")][0]
    lines, first = [], True
    for l in patch.splitlines():
        if l.startswith("@@"):
            start = int(l.split("+")[0].split("-")[1].split(",")[0])
            lines += ["// filler"] * (start - 1)
        elif l.startswith((" ", "-")) and not l.startswith("---"):
            lines.append(l[1:])
    return {path: ("\n".join(lines) + "\n").encode()}


def unified_has_bug_line(diff, patch):
    added = [l[1:] for l in patch.splitlines() if l.startswith("+") and not l.startswith("+++")]
    return all(a in diff for a in added)


def test_answer_task_needs_no_sandbox():
    task = TASKS / "sponsor_statsv2_diggcount_total"
    accepted = [l.strip() for l in (task / "answer.txt").read_text().splitlines() if l.strip()]
    good = asyncio.run(guard.guarded_check(None, ARENA / "envs" / "tiktok", task, answer=f"  {accepted[0].upper()} "))
    bad = asyncio.run(guard.guarded_check(None, ARENA / "envs" / "tiktok", task, answer="zzz-wrong"))
    none = asyncio.run(guard.guarded_check(None, ARENA / "envs" / "tiktok", task, answer=None))
    assert good.passed and good.reason == "passed" and not bad.passed and not none.passed
