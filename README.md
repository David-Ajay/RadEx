# RadExplain (Phase 1: lungs + heart)

Chest X-ray -> multi-organ classification -> per-organ disease probabilities -> structured report -> RAG chatbot.
Runs 100% on your MacBook (web UI in the browser at http://localhost:8501 in the UI stage).

## One-time setup (Terminal on your Mac)
```bash
cd ~/RadExplain/radexplain          # wherever you unzipped this
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                    # makes `radexplain` importable
python scripts/00_check_env.py      # should print MPS available: True
```

## Stage 0 - data
```bash
# Kaggle token: kaggle.com -> Settings -> API -> Create New Token -> move to ~/.kaggle/kaggle.json
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

bash scripts/01_download_data.sh chexpert     # ~11 GB
bash scripts/01_download_data.sh nih          # ~42 GB
bash scripts/02_build_labels.sh               # -> data/labels/master.csv
bash scripts/03_preprocess_images.sh --limit 500   # quick test first
bash scripts/03_preprocess_images.sh               # full run (resumable)
```
