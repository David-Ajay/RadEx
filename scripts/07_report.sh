#!/usr/bin/env bash
# bash scripts/07_report.sh path/to/xray.png
set -euo pipefail
python -m radexplain.report.generate "$@"
