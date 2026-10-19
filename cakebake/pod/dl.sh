#!/bin/bash
# (Re)download models in pod/models.tsv whose dir has no model.safetensors.
set -x
export HF_HOME=/root/hf
grep -v '^#' /workspace/cakebake/pod/models.tsv | while IFS=$'\t' read name repo rev; do
  [ -f /root/models/$name/model.safetensors ] && continue
  [[ "$repo" == */* ]] || repo="model-organisms-for-real/$repo"
  /root/dlenv/bin/hf download "$repo" --revision "$rev" --local-dir /root/models/$name || echo "FAILED $name"
done
du -sh /root/models/*; echo DL_DONE
