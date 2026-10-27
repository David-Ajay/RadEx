"""Chat in the terminal.   python -m radexplain.rag.chat outputs/reports/<image>.json [--provider ollama]"""
import argparse

from radexplain.config import load_config
from radexplain.rag.chatbot import ReportChatbot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report_json")
    ap.add_argument("--provider", choices=["extractive", "ollama", "anthropic"])
    args = ap.parse_args()
    cfg = load_config("rag.yaml")
    if args.provider:
        cfg["provider"] = args.provider
    bot = ReportChatbot.from_file(args.report_json, cfg=cfg)
    print(f"[engine: {cfg['provider']}]  Ask about your report. Press Enter on an empty line to quit.\n")
    r = bot.explain_report()
    print("Bot:", r["answer"], "\n")
    while True:
        q = input("You: ").strip()
        if not q:
            break
        r = bot.ask(q)
        print("\nBot:", r["answer"])
        if r["sources"]:
            print("     (based on:", "; ".join(r["sources"]), ")")
        print()


if __name__ == "__main__":
    main()
