#!/usr/bin/env bash
set -euo pipefail
python -m radexplain.data.build_labels "$@"
