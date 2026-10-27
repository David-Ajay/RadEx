"""Tiny TF-IDF retriever. The knowledge base is only ~50 short chunks, so this is fast, needs no
model download and is easy to inspect. (Could be swapped for sentence-embeddings + FAISS later.)

Two indexes:
  * body index    : title + keywords + text   -> ranking
  * keyword index : title + keywords only     -> a chunk may only be returned if the question really
                    overlaps its title/keywords. This stops off-topic questions ("tell me a joke")
                    from matching a glossary entry because of some common word in its body text.
"""
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


STOP = list(ENGLISH_STOP_WORDS - {"serious"})      # sklearn treats "serious" as a stop word; here it matters


def _index(docs):
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words=STOP)
    return vec, vec.fit_transform(docs)


class Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.body_vec, self.body = _index([f"{c['title']} {c['keywords']} {c['text']}" for c in chunks])
        self.key_vec, self.key = _index([f"{c['title']} {c['keywords']}" for c in chunks])

    def search(self, query: str, k: int = 4, min_score: float = 0.06) -> list[tuple[dict, float]]:
        body = cosine_similarity(self.body_vec.transform([query]), self.body)[0]
        key = cosine_similarity(self.key_vec.transform([query]), self.key)[0]
        score = body + key
        order = score.argsort()[::-1]
        return [(self.chunks[i], float(score[i])) for i in order[:k] if key[i] > 0 and score[i] >= min_score]
