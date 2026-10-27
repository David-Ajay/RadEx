#!/usr/bin/env bash
# bash scripts/12_ensemble.sh --runs pleura phase1 [--tta]
set -euo pipefail
python -m radexplain.evaluate_ensemble "$@"
