#!/usr/bin/env bash
# bash scripts/09_app.sh   -> opens http://localhost:8501
set -euo pipefail
streamlit run src/radexplain/app/main.py "$@"
