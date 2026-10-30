#!/bin/bash
# Phase-1 fan-out: 3 sol leads (with terra subagents) + 2 sol readers. Read-only on data; no paid APIs.
cd /home/john/code/results/neel-astra
for n in sol-grading sol-ncri sol-crosscheck sol-reader-top sol-reader-zero; do
  mkdir -p work/$n
  codex exec -m gpt-6.1-sol -C /home/john/code/results/neel-astra -s workspace-write --skip-git-repo-check \
    -o notes/$n.final.md "$(cat briefs/$n.md)" > work/$n/codex.log 2>&1 &
done
wait
ls -la notes
