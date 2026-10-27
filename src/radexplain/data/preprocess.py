"""Resize every image in master.csv to a small grayscale JPEG. Resumable (skips done files)."""
import argparse
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from radexplain.config import ROOT, load_config, path


def _work(job):
    src, dst, size, quality = job
    dst = Path(dst)
    if dst.exists():
        return 0
    try:
        with Image.open(src) as im:
            im = im.convert("L")                 # NIH PNGs are sometimes RGBA; X-rays are 1-channel
            im = im.resize((size, size), Image.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, "JPEG", quality=quality)
        return 1
    except Exception as e:                       # corrupt file: report, keep going
        print(f"[skip] {src}: {e}")
        return -1


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only process N images (quick test)")
    ap.add_argument("--workers", type=int, default=cfg["preprocess"]["workers"])
    args = ap.parse_args()

    size, quality = cfg["preprocess"]["image_size"], cfg["preprocess"]["jpeg_quality"]
    master = pd.read_csv(path(cfg, "master_csv"))
    if args.limit:
        master = master.sample(min(args.limit, len(master)), random_state=0)
    jobs = [(s, str(ROOT / d), size, quality) for s, d in zip(master["src_path"], master["processed_path"])]

    with Pool(args.workers) as pool:
        res = list(tqdm(pool.imap_unordered(_work, jobs, chunksize=64), total=len(jobs)))
    print(f"done: {res.count(1):,} written, {res.count(0):,} already existed, {res.count(-1):,} failed")


if __name__ == "__main__":
    main()
