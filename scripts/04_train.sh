#!/usr/bin/env bash
# Smoke test:  bash scripts/04_train.sh --subset 2000 --epochs 1
# Full run:    bash scripts/04_train.sh
# Resume:      bash scripts/04_train.sh --resume
set -euo pipefail
python -m radexplain.train "$@"
