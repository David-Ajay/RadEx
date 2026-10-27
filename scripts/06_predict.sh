#!/usr/bin/env bash
# bash scripts/06_predict.sh path/to/xray.png
set -euo pipefail
python -m radexplain.predict "$@"
