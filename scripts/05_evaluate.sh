#!/usr/bin/env bash
# bash scripts/05_evaluate.sh            (full test set, ~10-15 min)
# bash scripts/05_evaluate.sh --limit 500   (quick check)
set -euo pipefail
python -m radexplain.evaluate "$@"
