"""LLM back-ends. All share:  generate(system, messages, chunks) -> str
   messages = [{"role": "user"|"assistant", "content": str}, ...]  (last one is the new question)

  ExtractiveLLM : no model at all - stitches the answer from retrieved text (default, works offline)
  OllamaLLM     : local model through the Ollama server
  AnthropicLLM  : Claude API (key from the ANTHROPIC_API_KEY environment variable)
  GeminiLLM     : Google Gemini API (key from GEMINI_API_KEY, normally set in the .env file)
"""
import os
import time

import requests

from radexplain.config import load_env
from radexplain.rag.intents import flags


class ExtractiveLLM:
    INTRO = {
        "health": ("I can't tell you whether you are healthy or well: a chest X-ray screening tool can't answer that, and "
                   "only a doctor can, using your symptoms, an examination and other tests. Here is what your report says. "),
        "symptoms": ("I'm sorry you're not feeling well. This tool only looks at one X-ray image and can't explain "
                     "symptoms, and many different things can cause them, so please talk to a doctor about how you feel. "
                     "Here is what your report says. "),
        "report": "Here is your report in simple words. ",
    }

    def generate(self, system, messages, chunks):
        f = flags(messages[-1]["content"])
        finds = [c for c in chunks if c["id"].startswith("R-find-")]
        if f["health"] or f["symptoms"] or len(finds) >= 2:    # "explain my whole report" / "am I ok?" / symptoms
            gl = {c["id"]: c for c in chunks if c["source"] == "glossary"}
            summary = next((c["text"] for c in chunks if c["id"] == "R-summary"), "")
            intro = self.INTRO["health" if f["health"] else "symptoms" if f["symptoms"] else "report"]
            parts = [intro + summary]
            for fc in finds:
                g = gl.get("G-" + fc["id"].split("R-find-")[1].lower().replace(" ", "_"))
                parts.append((g["text"] if g else fc["text"]) + f" (In your report this is marked '{fc.get('tier', 'flagged')}'.)")
            parts.append("These are automated estimates from an experimental AI, and false alarms are common, "
                         "especially for 'possible' findings. A radiologist needs to review the report; "
                         "please share it with a doctor.")
            return "\n\n".join(parts)
        # Otherwise: the person's own finding first, then glossary explanations (max 3 pieces).
        report = [c for c in chunks if c["source"] == "report" and not c.get("always")]
        gloss = [c for c in chunks if c["source"] == "glossary"]
        picked = (report[:1] + gloss[:2]) if report else gloss[:2]
        if not picked:
            picked = [c for c in chunks if c.get("always")][:1]
        return "\n\n".join(c["text"] for c in picked)


class OllamaLLM:
    def __init__(self, model="llama3.2", host="http://localhost:11434", max_tokens=500, temperature=0.2):
        self.model, self.host, self.max_tokens, self.temperature = model, host.rstrip("/"), max_tokens, temperature

    def generate(self, system, messages, chunks):
        try:
            r = requests.post(f"{self.host}/api/chat", timeout=180, json={
                "model": self.model, "stream": False,
                "messages": [{"role": "system", "content": system}] + messages,
                "options": {"temperature": self.temperature, "num_predict": self.max_tokens}})
            r.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Could not reach Ollama at {self.host} ({e}). Is it running, and did you run "
                               f"`ollama pull {self.model}`?") from e
        return r.json()["message"]["content"].strip()


class AnthropicLLM:
    def __init__(self, model="claude-haiku-5-5", max_tokens=500, temperature=0.2):
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self.key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. In Terminal run:  export ANTHROPIC_API_KEY=...  "
                               "(paste your key there, never into files or chats)")

    def generate(self, system, messages, chunks):
        try:
            r = requests.post("https://api.anthropic.com/v1/messages", timeout=120,
                              headers={"x-api-key": self.key, "anthropic-version": "2023-06-01",
                                       "content-type": "application/json"},
                              json={"model": self.model, "max_tokens": self.max_tokens, "system": system,
                                    "temperature": self.temperature, "messages": messages})
            r.raise_for_status()
        except requests.RequestException as e:
            detail = getattr(getattr(e, "response", None), "text", "")[:300]
            raise RuntimeError(f"Claude API request failed: {e} {detail}") from e
        return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text").strip()


class GeminiLLM:
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    BUSY = (500, 502, 503, 504)

    def __init__(self, model="gemini-flash-latest", fallback_models=(), max_tokens=1500, temperature=0.2, retry_wait=1.5):
        load_env()                                              # picks up edits to .env without restarting
        self.models = [model, *[m for m in fallback_models if m != model]]
        self.max_tokens, self.temperature, self.retry_wait = max_tokens, temperature, retry_wait
        self.key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.key:
            raise RuntimeError("No Gemini API key found. Open the file '.env' in the radexplain folder and put your "
                               "key after GEMINI_API_KEY= (see .env.example), save it, then reload this page.")

    def _post(self, model, body):
        """One request; a short pause and ONE retry if Google answers 'busy' (these are usually momentary)."""
        for attempt in range(2):
            try:                                                # the key travels in a header, never in the URL
                r = requests.post(self.URL.format(model=model), json=body, timeout=60,
                                  headers={"x-goog-api-key": self.key, "Content-Type": "application/json"})
            except requests.RequestException as e:
                raise RuntimeError(f"Could not reach the Gemini API ({type(e).__name__}). Check your internet connection.") from e
            if r.status_code in self.BUSY and attempt == 0:
                time.sleep(self.retry_wait)
                continue
            return r
        return r

    def generate(self, system, messages, chunks):
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                             for m in messages],
                "generationConfig": {"temperature": self.temperature, "maxOutputTokens": self.max_tokens}}
        not_found, trouble = [], []
        for model in self.models:
            r = self._post(model, body)
            if r.status_code == 404:                            # model name retired or renamed: try the next one
                not_found.append(model)
                continue
            if r.status_code in (400, 401, 403):                # key or request problem: another model won't help
                msg = _gemini_error(r)
                hint = " Check the key in your .env file." if "key" in msg.lower() or r.status_code in (401, 403) else ""
                raise RuntimeError(f"Gemini rejected the request: {msg}.{hint}")
            if r.status_code == 429 or r.status_code >= 500:    # busy or rate-limited: try the next model
                trouble.append((model, r.status_code, _gemini_error(r)))
                continue
            data = r.json()
            cand = (data.get("candidates") or [{}])[0]
            text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", [])).strip()
            if text:
                return text
            reason = cand.get("finishReason") or (data.get("promptFeedback") or {}).get("blockReason") or "unknown"
            raise RuntimeError(f"Gemini returned no answer (reason: {reason}). Try rephrasing the question.")
        if trouble:
            if all(code == 429 for _, code, _ in trouble):
                raise RuntimeError("Gemini's free limit was reached for now (too many questions in a short time). "
                                   "Wait a minute and try again.")
            detail = "; ".join(f"{m}: {c} {e}" for m, c, e in trouble)[:300]
            raise RuntimeError(f"Gemini is busy or unavailable right now ({detail}). Please try again in a moment.")
        raise RuntimeError(f"None of these Gemini models were found: {', '.join(not_found)}. Open configs/rag.yaml and set "
                           "gemini: model: to a current model name (shown in Google AI Studio).")


def _gemini_error(r) -> str:
    try:
        return r.json()["error"]["message"]
    except Exception:
        return f"HTTP {r.status_code}"


def make_llm(cfg: dict):
    load_env()
    p = cfg["provider"]
    if p == "extractive":
        return ExtractiveLLM()
    if p == "gemini":
        g = cfg.get("gemini", {})
        return GeminiLLM(g.get("model", "gemini-flash-latest"), g.get("fallback_models", []),
                         g.get("max_tokens", 1500), cfg["chat"]["temperature"])
    if p == "ollama":
        return OllamaLLM(cfg["ollama"]["model"], cfg["ollama"]["host"], cfg["chat"]["max_tokens"], cfg["chat"]["temperature"])
    if p == "anthropic":
        return AnthropicLLM(cfg["anthropic"]["model"], cfg["chat"]["max_tokens"], cfg["chat"]["temperature"])
    raise ValueError(f"unknown provider '{p}' (use extractive, gemini, ollama or anthropic)")
