"""Download helpers. Uses the Kaggle CLI (pip install kaggle + ~/.kaggle/kaggle.json)."""
import argparse
import shutil
import subprocess

from radexplain.config import load_config, path


def free_gb(p) -> float:
    p.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(p).free / 1e9


def kaggle_download(slug: str, dest):
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {slug} -> {dest}  (free space: {free_gb(dest):.0f} GB)")
    subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["chexpert", "nih"])
    args = ap.parse_args()
    cfg = load_config()
    if args.dataset == "chexpert":
        kaggle_download(cfg["kaggle"]["chexpert"], path(cfg, "raw_chexpert"))
    else:
        kaggle_download(cfg["kaggle"]["nih"], path(cfg, "raw_nih"))


if __name__ == "__main__":
    main()
