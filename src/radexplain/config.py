import os
from pathlib import Path

import yaml

# Project root (the folder containing configs/). RADEXPLAIN_ROOT is only used by the tests,
# so they can run against a temporary folder and never touch your real data or checkpoints.
ROOT = (Path(os.environ["RADEXPLAIN_ROOT"]) if os.environ.get("RADEXPLAIN_ROOT")
        else Path(__file__).resolve().parents[2])


def load_config(name: str = "phase1.yaml") -> dict:
    with open(ROOT / "configs" / name) as f:
        return yaml.safe_load(f)


def path(cfg: dict, key: str) -> Path:
    """Resolve a path from cfg['paths'] relative to the project root."""
    return ROOT / cfg["paths"][key]


def load_env():
    """Read KEY=VALUE lines from <project>/.env into the environment (e.g. GEMINI_API_KEY).
    Values already set in the real environment win. The .env file is git-ignored and never shown in the app."""
    f = ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


load_env()
