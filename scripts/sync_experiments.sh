#!/usr/bin/env bash
# Mirror every text artifact of the experiments (logs, benchmark tables, cost
# tables, run configs, training curves) from the git-ignored runs/ into the tracked
# experiments/ directory, and record size + sha256 of every weight file so the
# models behind the reported numbers can be verified without storing them in git.
#   scripts/sync_experiments.sh        # run after every experiment, then commit
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p experiments
rsync -a --prune-empty-dirs \
  --include='*/' \
  --include='*.log' --include='*.jsonl' --include='*.md' \
  --include='*config.json' --include='meta.json' --include='state.json' --include='woodcock.json' \
  --include='costs/*.json' --include='bench/*.json' \
  --exclude='*' runs/ experiments/runs/
{
  printf 'path\tbytes\tsha256\n'
  find runs -name '*.safetensors' ! -name 'optimizer.safetensors' ! -name 'master.safetensors' -print0 | sort -z |
    while IFS= read -r -d '' f; do
      printf '%s\t%s\t%s\n' "$f" "$(stat -f%z "$f")" "$(shasum -a 256 "$f" | cut -d' ' -f1)"
    done
} > experiments/weights_manifest.tsv
du -sh experiments
