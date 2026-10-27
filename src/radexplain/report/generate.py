"""Image -> report.   python -m radexplain.report.generate path/to/xray.png [--name phase1]

Saves outputs/reports/<image>.md and .json (the JSON is what the chatbot will read in Stage 4).
"""
import argparse
import json
from pathlib import Path

from radexplain.config import ROOT
from radexplain.predict import Predictor
from radexplain.report.generator import build_report, load_reliability, render_markdown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--name", default="phase1")
    args = ap.parse_args()

    result = Predictor(f"checkpoints/{args.name}/best.pt").predict(args.image)
    report = build_report(result, load_reliability(ROOT, args.name))
    md = render_markdown(report)
    print(md)

    out = ROOT / "outputs" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    (out / f"{stem}.md").write_text(md)
    (out / f"{stem}.json").write_text(json.dumps({"report": report, "model_output": result}, indent=2))
    print(f"\nsaved: {out / (stem + '.md')}  and  .json")


if __name__ == "__main__":
    main()
