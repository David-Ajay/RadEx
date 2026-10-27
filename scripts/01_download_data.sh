#!/usr/bin/env bash
# Usage: bash scripts/01_download_data.sh chexpert   |   bash scripts/01_download_data.sh nih
set -euo pipefail
python -m radexplain.data.download "${1:?usage: $0 chexpert|nih}"
