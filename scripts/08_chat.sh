#!/usr/bin/env bash
# bash scripts/08_chat.sh outputs/reports/<image>.json [--provider ollama]
set -euo pipefail
python -m radexplain.rag.chat "$@"
