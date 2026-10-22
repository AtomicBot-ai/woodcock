#!/bin/bash
# Copy the code and the prompt pool to the CUDA machine (run on the Mac).
#   scripts/cuda/push.sh HOST PORT [DEST]     e.g. scripts/cuda/push.sh 1.2.3.4 41022
set -euo pipefail
HOST="$1"; PORT="$2"; DEST="${3:-/root/woodcock-clean}"
ssh -p "$PORT" "root@$HOST" "mkdir -p $DEST/data"
git ls-files -z | rsync -az --from0 --files-from=- -e "ssh -p $PORT" ./ "root@$HOST:$DEST/"
rsync -az -e "ssh -p $PORT" data/prompts-v2 "root@$HOST:$DEST/data/"
