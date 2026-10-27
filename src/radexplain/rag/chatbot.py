"""Stage 4: a chatbot grounded ONLY on (1) the generated report and (2) the medical glossary."""
import json
from pathlib import Path

from radexplain.config import ROOT, load_config
from radexplain.rag.chunks import glossary_chunks, load_glossary, report_chunks
from radexplain.rag.intents import flags
from radexplain.rag.llm import ExtractiveLLM, make_llm
from radexplain.rag.prompts import NO_MATCH, SAFETY_NOTE, SYSTEM_PROMPT
from radexplain.rag.retriever import Retriever


class ReportChatbot:
    def __init__(self, report: dict, llm=None, glossary_path=None, cfg=None):
        self.cfg = cfg or load_config("rag.yaml")
        self.llm = llm or make_llm(self.cfg)
        self.report = report
        gl = load_glossary(glossary_path or ROOT / "knowledge" / "glossary.json")
        self.chunks = report_chunks(report) + glossary_chunks(gl)
        self.retriever = Retriever(self.chunks)
        self.history: list[dict] = []

    @classmethod
    def from_file(cls, report_json, **kw):
        data = json.loads(Path(report_json).read_text())
        return cls(data["report"] if "report" in data else data, **kw)

    def _report_context(self, extra_ids=()):
        """The whole report plus the glossary entries for every flagged finding, the tiers and any extras."""
        rep = [c for c in self.chunks if c["source"] == "report" and not c["id"].startswith("R-disclaimer")]
        flagged = [c["id"].split("R-find-")[1] for c in rep if c["id"].startswith("R-find-")]
        gids = {"G-" + f.lower().replace(" ", "_") for f in flagged} | set(extra_ids)
        gids |= {f"G-tier_{c['tier']}" for c in rep if c.get("tier")}                  # explain 'possible' / 'likely' too
        return rep + [c for c in self.chunks if c["id"] in gids]

    @staticmethod
    def _merge(*lists):
        seen, out = set(), []
        for lst in lists:
            for c in lst:
                if c["id"] not in seen:
                    seen.add(c["id"]); out.append(c)
        return out

    def _context(self, question):
        rc, f = self.cfg["retrieval"], flags(question)
        picked = [c for c, _ in self.retriever.search(question, rc["top_k"], rc["min_score"])]
        vague = f["whole"] or f["health"] or f["symptoms"]       # "explain it", "am I ok?", "I feel short of breath"
        if vague:
            extra = ["G-next_steps", "G-ai_limits"] + (["G-emergency"] if f["symptoms"] or f["urgent"] else [])
            picked = self._merge(self._report_context(extra), picked)
        always = [c for c in self.chunks if c.get("always") and c not in picked]
        return always + picked, bool(picked) or vague

    def ask(self, question: str) -> dict:
        safety = SAFETY_NOTE if flags(question)["urgent"] else None
        chunks, matched = self._context(question)
        basic_chunks, basic_matched = chunks, matched                # what the basic engine would use
        note = None
        focus = [c for c in chunks if not c.get("always")]
        if isinstance(self.llm, ExtractiveLLM):
            use = chunks                                          # basic engine: only what the question matched
        else:                                                     # language model: always see the full report, and
            use = self._merge(self._report_context(["G-next_steps", "G-ai_limits", "G-emergency"]), chunks)
            matched = True                                        # let it handle odd or off-topic questions itself
        if not matched:
            answer = NO_MATCH
        else:
            ctx = "\n\n".join(f"[{c['source'].upper()}: {c['title']}] {c['text']}" for c in use)
            system = f"{SYSTEM_PROMPT}\n\nCONTEXT\n{ctx}"
            turns = self.history[-2 * self.cfg["chat"]["history_turns"]:]
            try:
                answer = self.llm.generate(system, turns + [{"role": "user", "content": question}], use)
            except RuntimeError as e:
                if isinstance(self.llm, ExtractiveLLM):
                    raise
                note = str(e)                                       # e.g. Gemini busy: still give a useful answer
                answer = (ExtractiveLLM().generate(system, turns + [{"role": "user", "content": question}], basic_chunks)
                          if basic_matched else NO_MATCH)
        self.history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        sources = (list(dict.fromkeys(c["title"] for c in focus)) or ["Your report"]) if matched else []
        return {"answer": (safety + "\n\n" + answer) if safety else answer, "sources": sources,
                "urgent": bool(safety), "note": note}

    def explain_report(self) -> dict:
        return self.ask("Please explain my report in simple words.")
