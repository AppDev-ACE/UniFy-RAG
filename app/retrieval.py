"""Hybrid (dense + BM25) pair retrieval with a dependency-free sparse fallback."""
from __future__ import annotations
import json, math, re
from collections import Counter, defaultdict
from pathlib import Path

from app.config import RRF_K

ABBREVIATIONS = {
    "swi": "student web interface", "pwi": "parent web interface",
    "cub": "city union bank", "cia": "continuous internal assessment",
    "swas": "student welfare activities and support", "tc": "transfer certificate",
    "abc": "academic bank of credits", "scos": "student citizens of sastra",
    "cgpa": "cumulative grade point average", "sgpa": "semester grade point average",
    "acrs": "accelerated credit registration system", "noc": "no objection certificate",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")

def normalize(text: str) -> str:
    text = text.lower().strip()
    words = TOKEN_RE.findall(text)
    expanded = []
    for word in words:
        expanded.append(word)
        if word in ABBREVIATIONS:
            expanded.extend(TOKEN_RE.findall(ABBREVIATIONS[word]))
    return " ".join(expanded)

class Retriever:
    def __init__(self, index_dir: Path):
        self.records = json.loads((index_dir / "records.json").read_text())
        self.questions = json.loads((index_dir / "questions.json").read_text())
        self.tokens = [normalize(item["question"]).split() for item in self.questions]
        self.df = Counter(token for doc in self.tokens for token in set(doc))
        self.avgdl = sum(map(len, self.tokens)) / max(len(self.tokens), 1)
        self.dense = None
        self.model = None
        vectors = index_dir / "vectors.npy"
        if vectors.exists():
            try:
                import numpy as np
                from sentence_transformers import SentenceTransformer
                self.dense = np.load(vectors)
                if len(self.dense) != len(self.questions):
                    raise ValueError("embedding count does not match index questions")
                # Serving must not make network calls. The builder downloads
                # the model once; if it is absent locally we safely use BM25.
                self.model = SentenceTransformer("BAAI/bge-small-en-v1.5", local_files_only=True)
            except Exception:
                # Sparse retrieval remains safe and usable if an optional model is absent.
                self.dense = self.model = None

    def _bm25(self, query: str):
        q = normalize(query).split(); n = len(self.tokens); scores = []
        for doc in self.tokens:
            counts = Counter(doc); score = 0.0
            for token in q:
                if not counts[token]: continue
                idf = math.log(1 + (n - self.df[token] + .5) / (self.df[token] + .5))
                score += idf * counts[token] * 2.0 / (counts[token] + 1.5 * (1 - .75 + .75 * len(doc) / self.avgdl))
            scores.append(score)
        return scores

    def search(self, query: str, limit: int = 3):
        bm25 = self._bm25(query)
        sparse_order = sorted(range(len(bm25)), key=lambda i: bm25[i], reverse=True)
        rrf = defaultdict(float)
        for rank, idx in enumerate(sparse_order[:50], 1): rrf[idx] += 1 / (RRF_K + rank)
        if self.dense is not None:
            import numpy as np
            vector = self.model.encode(["Represent this sentence for searching relevant passages: " + normalize(query)], normalize_embeddings=True)[0]
            dense_order = np.argsort(self.dense @ vector)[::-1].tolist()
            for rank, idx in enumerate(dense_order[:50], 1): rrf[idx] += 1 / (RRF_K + rank)
        # Rankings are over question phrasings, but the user-facing unit is a
        # record. Keep only the strongest phrasing for each record so a
        # clarification never offers the same answer three times.
        best_for_record = {}
        for idx in sorted(rrf, key=rrf.get, reverse=True):
            record_index = self.questions[idx]["record_index"]
            if record_index not in best_for_record:
                best_for_record[record_index] = idx
        ranked = list(best_for_record.values())[:limit]
        # A 0..1 interpretable confidence; never rely on raw RRF alone.
        max_bm25 = max(bm25) if bm25 else 0
        output = []
        for idx in ranked:
            item = self.questions[idx]; record = self.records[item["record_index"]]
            # Confidence is intentionally only a calibration feature, never a
            # semantic-score proxy. It combines exact query-token coverage and
            # BM25 evidence; thresholds are selected using a labelled set.
            query_terms = set(normalize(query).split())
            overlap = len(query_terms & set(self.tokens[idx])) / max(len(query_terms), 1)
            lexical_strength = 1 - math.exp(-bm25[idx] / 6)
            lexical = overlap * lexical_strength
            output.append({"record": record, "question": item["question"], "score": lexical, "rrf": rrf[idx]})
        return output
