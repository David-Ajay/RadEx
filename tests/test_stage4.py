"""Run:  python tests/test_stage4.py"""
import os

import requests

from radexplain.config import ROOT, load_config
from radexplain.rag import llm as llm_mod
from radexplain.rag.chatbot import ReportChatbot
from radexplain.rag.chunks import glossary_chunks, load_glossary
from radexplain.rag.prompts import NO_MATCH
from radexplain.report.generator import build_report
from radexplain.report.templates import PHRASES


def F(name, p, thr=0.10, hi=0.30):
    return {"finding": name, "probability": p, "threshold": thr, "high_threshold": hi, "flagged": p >= thr}


def sample_report():
    org = lambda a: {"probability": 0.3, "threshold": 0.4, "affected": a}
    res = {"organs": {"lungs": org(False), "heart": org(False)},
           "findings": {"lungs": [F("Atelectasis", 0.151), F("Edema", 0.40), F("Pneumonia", 0.02), F("Mass", 0.01)],
                        "heart": [F("Cardiomegaly", 0.003)]}}
    return build_report(res, {"Atelectasis": "moderate", "Edema": "moderate"})


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, system, messages, chunks):
        self.calls.append((system, messages, chunks))
        return "fake answer"


def bot(llm=None, **cfg):
    c = load_config("rag.yaml"); c["provider"] = "extractive"
    c["chat"].update(cfg)
    return ReportChatbot(sample_report(), llm=llm, cfg=c)


def test_glossary_covers_every_report_phrase():
    ids = {g["id"] for g in load_glossary(ROOT / "knowledge" / "glossary.json")}
    missing = [l for l in PHRASES if l.lower().replace(" ", "_") not in ids]
    assert not missing, missing


def test_retrieval_finds_the_right_entry():
    b = bot()
    top = lambda q: b.retriever.search(q, 1)[0][0]["id"]
    assert top("is there water in my lungs?") == "G-edema"
    assert top("what is an enlarged heart") == "G-cardiomegaly"
    assert "tier_possible" in top("what does possible mean") or top("what does possible mean") == "G-tier_possible"


def test_unrelated_questions_get_no_match():
    for q in ["who won the cricket match yesterday", "tell me a joke", "what is the capital of france",
              "what is the best pizza near me"]:
        r = bot().ask(q)                                    # basic engine says "not in your report"
        assert r["answer"] == NO_MATCH and r["sources"] == [], q


def test_vague_worry_questions_still_get_help():
    for q in ["is this serious", "should i be worried", "does this mean i have cancer"]:
        r = bot().ask(q)
        assert r["answer"] != NO_MATCH and "doctor" in r["answer"].lower(), q


def test_why_flagged_shows_the_flagged_findings():
    fake = FakeLLM(); bot(fake).ask("why was my lung flagged")
    ids = [c["id"] for c in fake.calls[0][2]]
    assert "R-find-Atelectasis" in ids and "R-find-Edema" in ids and "G-tier_possible" in ids


def test_urgent_symptoms_get_safety_note():
    r = bot().ask("I have chest pain right now, what is this")
    assert r["urgent"] and r["answer"].startswith("**If you or someone")


def test_prompt_is_grounded_and_history_is_trimmed():
    fake = FakeLLM(); b = bot(fake, history_turns=1)
    for q in ["what is atelectasis", "what is edema", "what does possible mean"]:
        b.ask(q)
    system, messages, _ = fake.calls[-1]
    assert "CONTEXT" in system and "Report summary" in system and "Never diagnose" in system
    assert len(messages) == 3                      # 1 previous turn (2 msgs) + the new question


def test_explain_report_covers_all_flagged_findings():
    fake = FakeLLM(); b = bot(fake); b.explain_report()
    titles = " ".join(c["title"] for c in fake.calls[0][2])
    assert "atelectasis" in titles.lower() and "edema" in titles.lower()
    assert "Pulmonary edema" in titles             # glossary entry matched by id, not by title prefix
    ans = bot().explain_report()["answer"]
    assert "fluid" in ans.lower() and "not fully expanded" in ans.lower() and "marked 'likely'" in ans and "radiologist" in ans.lower()


def test_extractive_answer_uses_report_then_glossary():
    r = bot().ask("what does the atelectasis in my report mean?")
    assert "atelectasis" in r["answer"].lower() and "Your report" in r["answer"]


def test_loads_report_json_written_by_the_report_generator():
    import json, tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "r.json"; f.write_text(json.dumps({"report": sample_report(), "model_output": {}}))
        c = load_config("rag.yaml"); c["provider"] = "extractive"
        assert "atelectasis" in ReportChatbot.from_file(f, cfg=c).ask("what is atelectasis")["answer"].lower()


def test_ollama_and_anthropic_payloads(monkeypatch=None):
    sent = {}

    class R:
        def __init__(self, j): self._j = j
        def raise_for_status(self): pass
        def json(self): return self._j

    def fake_post(url, **kw):
        sent["url"], sent["kw"] = url, kw
        return R({"message": {"content": " hi "}} if "11434" in url else {"content": [{"type": "text", "text": " hi "}]})

    orig = requests.post; llm_mod.requests.post = fake_post
    try:
        o = llm_mod.OllamaLLM().generate("SYS", [{"role": "user", "content": "q"}], [])
        assert o == "hi" and sent["kw"]["json"]["stream"] is False and sent["kw"]["json"]["messages"][0]["role"] == "system"
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        a = llm_mod.AnthropicLLM().generate("SYS", [{"role": "user", "content": "q"}], [])
        assert a == "hi" and sent["kw"]["headers"]["x-api-key"] == "test-key" and sent["kw"]["json"]["system"] == "SYS"
        del os.environ["ANTHROPIC_API_KEY"]
        try:
            llm_mod.AnthropicLLM(); assert False
        except RuntimeError:
            pass
    finally:
        llm_mod.requests.post = orig


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok  ", name)

