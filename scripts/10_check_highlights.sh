#!/usr/bin/env bash
# bash scripts/10_check_highlights.sh --name pleura
set -euo pipefail
python -m radexplain.evaluate_localization "$@"
