"""Run:  python tests/test_stage7.py  - X-ray-pretrained backbone, Gemini engine, natural-language chatbot, .env, artwork"""
import io
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models

from radexplain.models.cam import class_activation_maps
from radexplain.models.io import load_model
from radexplain.models.radexplain_net import RadExplainNet, XrayBackbone, make_backbone

ORGANS = {"lungs": ["Atelectasis", "Edema"], "pleura": ["Pneumothorax"], "heart": ["Cardiomegaly"]}


def xray_features(seed=0):
    torch.manual_seed(seed)
    f = models.densenet121(weights=None).features
    f.conv0 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    return f


def test_xray_adapter_maps_our_input_to_the_1024_scale():
    feats = xray_features().eval()
    gray = torch.rand(2, 1, 224, 224)                                       # raw image in [0, 1]
    ours = (gray.repeat(1, 3, 1, 1) - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    want = (2 * gray - 1) * 1024                                            # what TorchXRayVision feeds its network

    class Identity(nn.Module):
        def forward(self, x):
            return x
    assert (XrayBackbone(Identity())(ours) - want).abs().max() < 1e-2       # the pure input transform is right
    with torch.no_grad():                                                   # and so is the whole network. A random
        a, b = XrayBackbone(feats).eval()(ours), feats(want)                # network has huge activations (~3000), so
    assert (a - b).abs().max() / b.abs().max() < 1e-4                       # compare RELATIVE to their size


def test_init_from_xray_weights_and_clear_error_when_missing():
    with tempfile.TemporaryDirectory() as d:
        feats = xray_features(1)
        path = Path(d) / "w.pt"; torch.save(feats.state_dict(), path)
        net = RadExplainNet(ORGANS, "densenet121_xray", pretrained=True, init_path=path)
        assert torch.equal(net.backbone.features.conv0.weight, feats.conv0.weight)
        assert torch.equal(net.backbone.features.denseblock4.denselayer16.conv2.weight, feats.denseblock4.denselayer16.conv2.weight)
        try:
            make_backbone("densenet121_xray", True, Path(d) / "missing.pt"); assert False
        except FileNotFoundError as e:
            assert "11_get_xray_weights" in str(e)


def test_checkpoint_roundtrip_cam_and_shapes_with_xray_backbone():
    torch.manual_seed(0)
    net = RadExplainNet(ORGANS, "densenet121_xray", pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224)
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "best.pt"
        torch.save({"model": net.state_dict(), "organ_labels": ORGANS, "backbone": "densenet121_xray", "image_size": 224}, ck)
        loaded, _ = load_model(ck, torch.device("cpu"))                      # rebuilt WITHOUT needing the weight file
    with torch.no_grad():
        assert torch.allclose(net(x)["disease"], loaded(x)["disease"], atol=1e-5)
    out, maps = loaded(x), class_activation_maps(loaded, x)
    bias = loaded.disease_heads["lungs"].bias[1].item()
    logit = out["disease"][0, 1].item()
    assert abs(maps["Edema"].mean() + bias - logit) < 1e-3 * max(1.0, abs(logit))     # CAM stays exact


def test_matches_real_torchxrayvision_when_installed():
    try:
        import torchxrayvision as xrv
        model = xrv.models.DenseNet(weights="densenet121-res224-chex").eval()
    except Exception:
        return                                                               # package or weights not available: skip
    feats = xray_features().eval()
    feats.load_state_dict({k[9:]: v for k, v in model.state_dict().items() if k.startswith("features.")})
    x = torch.randn(1, 1, 224, 224) * 300
    with torch.no_grad():
        a, b = F.adaptive_avg_pool2d(F.relu(feats(x)), 1).flatten(1), model.features2(x)
        assert (a - b).abs().max() <= 1e-5 * max(1.0, b.abs().max().item())


# ---------------------------------------------------------------- Gemini engine, intents, .env, artwork
class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body or {}

    def json(self):
        return self._body


def gemini_ok(text="hello"):
    return FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]})


def with_fake_post(responses, fn):
    from radexplain.rag import llm as llm_mod
    calls, orig = [], llm_mod.requests.post

    def fake(url, **kw):
        calls.append((url, kw))
        return responses.pop(0)

    llm_mod.requests.post = fake
    try:
        return fn(llm_mod), calls
    finally:
        llm_mod.requests.post = orig


def test_gemini_request_format_key_in_header_and_roles():
    os.environ["GEMINI_API_KEY"] = "secret-test-key"
    try:
        def run(m):
            g = m.GeminiLLM("gemini-x", ["gemini-y"], 900, 0.3)
            return g.generate("SYSTEM TEXT", [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
                                              {"role": "user", "content": "q2"}], [])
        out, calls = with_fake_post([gemini_ok(" hi ")], run)
        url, kw = calls[0]
        assert out == "hi" and url.endswith("/v1beta/models/gemini-x:generateContent") and "secret-test-key" not in url
        assert kw["headers"]["x-goog-api-key"] == "secret-test-key"
        body = kw["json"]
        assert body["systemInstruction"]["parts"][0]["text"] == "SYSTEM TEXT"
        assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]       # Gemini calls the assistant "model"
        assert body["generationConfig"] == {"temperature": 0.3, "maxOutputTokens": 900}
    finally:
        os.environ.pop("GEMINI_API_KEY", None)


def test_gemini_fallback_models_and_friendly_errors_never_leak_the_key():
    os.environ["GEMINI_API_KEY"] = "secret-test-key"
    try:
        def gen(m):
            g = m.GeminiLLM("old", ["new"], retry_wait=0)
            return g.generate("s", [{"role": "user", "content": "q"}], [])
        R = FakeResponse
        out, calls = with_fake_post([R(404), gemini_ok("ok")], gen)
        assert out == "ok" and "old" in calls[0][0] and "new" in calls[1][0]            # retired name -> next model
        out, calls = with_fake_post([R(503), gemini_ok("again")], gen)                  # momentary 'busy' -> one retry, same model
        assert out == "again" and len(calls) == 2 and "old" in calls[1][0]
        out, calls = with_fake_post([R(503), R(503), gemini_ok("fb")], gen)             # still busy -> the next model
        assert out == "fb" and "new" in calls[2][0]
        out, _ = with_fake_post([R(429), gemini_ok("fb2")], gen)                        # rate-limited model -> try the next
        assert out == "fb2"
        cases = [([R(429), R(429)], "free limit"),
                 ([R(503), R(503), R(500, {"error": {"message": "boom"}}), R(500, {"error": {"message": "boom"}})], "busy or unavailable"),
                 ([R(400, {"error": {"message": "API key not valid. Please pass a valid API key."}})], ".env"),
                 ([R(403, {"error": {"message": "denied"}})], ".env")]
        for responses, needle in cases:
            try:
                with_fake_post(responses, gen); assert False, needle
            except RuntimeError as e:
                assert needle in str(e) and "secret-test-key" not in str(e), str(e)
        try:                                                                            # the real reason is shown, not hidden
            with_fake_post([R(503, {"error": {"message": "The model is overloaded"}})] * 4, gen); assert False
        except RuntimeError as e:
            assert "overloaded" in str(e) and "503" in str(e)
        try:
            with_fake_post([R(404), R(404)], gen); assert False
        except RuntimeError as e:
            assert "configs/rag.yaml" in str(e)
        try:                                                                            # safety-blocked: no text
            with_fake_post([R(200, {"candidates": [{"finishReason": "SAFETY"}]})], gen); assert False
        except RuntimeError as e:
            assert "SAFETY" in str(e)
    finally:
        os.environ.pop("GEMINI_API_KEY", None)


def test_chatbot_falls_back_to_the_basic_engine_when_the_language_model_fails():
    class Broken:
        def generate(self, system, messages, chunks):
            raise RuntimeError("Gemini is busy or unavailable right now (x: 503 overloaded).")
    bot = sample_bot(Broken())
    r = bot.ask("am I healthy or ok?")
    assert "busy or unavailable" in r["note"] and "can't tell you whether you are healthy" in r["answer"]
    assert bot.history[-1]["role"] == "assistant" and bot.history[-1]["content"] == r["answer"]
    assert bot.ask("tell me a joke")["answer"].startswith("I couldn't find that")     # off-topic stays honest
    ok = sample_bot(SpyLLM()).ask("what is edema")
    assert ok["note"] is None and ok["answer"].endswith("spy answer")


def test_missing_key_gives_a_clear_message_and_env_file_is_loaded():
    from radexplain import config
    from radexplain.rag.llm import GeminiLLM
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        os.environ.pop(k, None)
    with tempfile.TemporaryDirectory() as d:
        old_root, config.ROOT = config.ROOT, Path(d)
        try:
            try:
                GeminiLLM(); assert False
            except RuntimeError as e:
                assert ".env" in str(e) and "GEMINI_API_KEY" in str(e)
            (Path(d) / ".env").write_text('# comment\n\nGEMINI_API_KEY = "abc123"\nOTHER=x y\nbroken line\n')
            g = GeminiLLM()                                                              # reads .env on construction
            assert g.key == "abc123" and os.environ["OTHER"] == "x y"
            os.environ["GEMINI_API_KEY"] = "from-shell"; config.load_env()
            assert os.environ["GEMINI_API_KEY"] == "from-shell"                          # real environment wins
        finally:
            config.ROOT = old_root
            for k in ("GEMINI_API_KEY", "OTHER"):
                os.environ.pop(k, None)


def test_intents_understand_everyday_english():
    from radexplain.rag.intents import flags
    for q in ["am i healthy or ok?", "Am I okay", "what happens to me", "is this serious", "do I have a problem?",
              "should i be worried", "what does this mean for me", "will I be fine"]:
        assert flags(q)["health"], q
    for q in ["I am feeling short breathed", "I have a cough and I'm tired", "i get dizzy and sweaty", "my chest hurts, some pain"]:
        assert flags(q)["symptoms"] or flags(q)["urgent"], q
    for q in ["short breathed", "I can't catch my breath", "chest pain right now", "i fainted", "coughing blood"]:
        assert flags(q)["urgent"], q
    assert not flags("what is atelectasis")["health"] and not flags("what is atelectasis")["urgent"]


def sample_bot(llm=None, provider="extractive"):
    from radexplain.config import load_config
    from radexplain.rag.chatbot import ReportChatbot
    from radexplain.report.generator import build_report
    F = lambda n, p, t=0.10, h=0.30: {"finding": n, "probability": p, "threshold": t, "high_threshold": h, "flagged": p >= t}
    org = lambda a: {"probability": 0.3, "threshold": 0.4, "affected": a}
    res = {"organs": {"lungs": org(False), "heart": org(False)},
           "findings": {"lungs": [F("Atelectasis", 0.151), F("Edema", 0.40)], "heart": [F("Cardiomegaly", 0.003)]}}
    cfg = load_config("rag.yaml"); cfg["provider"] = provider
    return ReportChatbot(build_report(res), llm=llm, cfg=cfg)


class SpyLLM:
    def __init__(self): self.system = None
    def generate(self, system, messages, chunks):
        self.system = system
        return "spy answer"


def test_language_model_always_sees_the_whole_report_even_for_vague_questions():
    for q in ["am i healthy or ok?", "what happens to me", "I'm feeling short breathed since two days", "tell me a joke"]:
        spy = SpyLLM(); r = sample_bot(spy).ask(q)
        ctx = spy.system.split("\n\nCONTEXT\n", 1)[1]
        assert r["answer"].endswith("spy answer") and "atelectasis" in ctx.lower() and "Pulmonary edema" in ctx, q
    assert "cannot tell whether someone is healthy" in spy.system and "single X-ray image" in spy.system   # new rules 6-7


def test_basic_engine_answers_health_and_symptom_questions_from_the_report():
    bot = sample_bot()
    h = bot.ask("am I healthy or ok?")["answer"]
    assert "can't tell you whether you are healthy" in h and "pulmonary edema" in h.lower() and "doctor" in h
    assert "you are healthy" not in h.replace("whether you are healthy", "")            # never claims health
    s = bot.ask("I have a cough and I feel very tired")
    assert "sorry you're not feeling well" in s["answer"] and "atelectasis" in s["answer"].lower() and not s["urgent"]
    u = bot.ask("I am short breathed since morning")
    assert u["urgent"] and u["answer"].startswith("**If you or someone") and "emergency" not in u["sources"]
    assert "NO_MATCH" not in h and "couldn't find that" not in h


def test_custom_realistic_photo_replaces_the_drawn_artwork():
    from radexplain.app import art
    with tempfile.TemporaryDirectory() as d:
        old, art.ROOT = art.ROOT, Path(d)
        try:
            assert art.find_custom_art() is None and "svg" in art.hero_art_uri()           # default: drawn artwork
            (Path(d) / "assets").mkdir()
            Image.fromarray(np.random.randint(0, 255, (2500, 2000), dtype=np.uint8)).save(Path(d) / "assets" / "xray_hero.png")
            uri = art.hero_art_uri()
            assert uri.startswith("data:image/jpeg;base64,") and art.bg_art_uri().startswith("data:image/jpeg")
            import base64
            im = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
            assert max(im.size) <= 1400                                                    # resized for the web page
            (Path(d) / "assets" / "xray_hero.png").write_bytes(b"not an image")
            assert "svg" in art.hero_art_uri()                                             # broken file -> safe fallback
        finally:
            art.ROOT = old


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok  ", name)
