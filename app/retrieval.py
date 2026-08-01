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

# Freshers rarely reuse the exact noun a policy answer uses. These are small,
# tightly-scoped bridges between common fresher phrasing and campus-policy
# vocabulary; expanding both sides symmetrically (like ABBREVIATIONS) keeps
# indexed questions and live queries comparable.
SYNONYMS = {
    "dress": "uniform attire",
    "attire": "uniform dress",
    "outfit": "uniform dress attire",
    "timing": "timings hours schedule",
    "timings": "timing hours schedule",
    "fee": "fees payment",
    "fees": "fee payment",
    "id": "identity card idcard",
}

# Stripped only from the overlap denominator, never from the token stream
# used for BM25/dense search, so function words still contribute lexical
# evidence without making verbose fresher phrasings look like weak matches.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "for", "of", "in", "on", "at", "to", "and", "or",
    "with", "about", "what", "when", "where", "which", "who", "whom", "how",
    "can", "could", "should", "would", "will", "shall", "i", "me", "my",
    "we", "our", "you", "your", "it", "its", "this", "that", "these",
    "those", "there", "if", "as", "by", "from", "up", "so", "than", "then",
    "also", "please", "kindly", "any", "there",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")

def normalize(text: str, expand_synonyms: bool = True) -> str:
    """Lowercase and tokenize, expanding abbreviations (always) and the
    fresher-vocabulary synonym table (optional).

    Abbreviations are opaque -- a sentence embedding model has no way to
    know "CUB" means City Union Bank, so expanding them measurably improves
    dense ranking (verified: several CUB/SWI queries move up 1-2 ranks).
    The synonym table is different: it exists only to help BM25's literal
    token overlap, since a sentence embedding model already understands
    "dress"/"uniform" are related without help. Feeding it into the dense
    encoder too made no measurable difference in testing, but risks turning
    a natural query into token soup for embeddings the model wasn't tuned
    on, so callers that build a dense query pass expand_synonyms=False.
    """
    text = text.lower().strip()
    words = TOKEN_RE.findall(text)
    expanded = []
    for word in words:
        expanded.append(word)
        if word in ABBREVIATIONS:
            expanded.extend(TOKEN_RE.findall(ABBREVIATIONS[word]))
        if expand_synonyms and word in SYNONYMS:
            expanded.extend(SYNONYMS[word].split())
    return " ".join(expanded)

class Retriever:
    def __init__(self, index_dir: Path):
        self.records = json.loads((index_dir / "records.json").read_text())
        self.questions = json.loads((index_dir / "questions.json").read_text())
        # An exact normalised question is a direct lookup of a reviewed
        # phrasing, not a fuzzy retrieval decision. Keep all matches because
        # duplicate phrasings must still be treated as ambiguous.
        # Synonyms are a query-side bridge only, never indexed into
        # documents: a record that literally says "dress code" is not
        # thereby "about" uniforms, and expanding it symmetrically (as
        # ABBREVIATIONS correctly does, since those are true equivalences)
        # let a generic dress-code record accumulate false "uniform"
        # affinity just from having "dress" in several of its phrasings,
        # outranking the actual lab-uniform record. Only the query side
        # expands synonyms, in search() below.
        self.exact_questions = defaultdict(list)
        for index, item in enumerate(self.questions):
            self.exact_questions[normalize(item["question"], expand_synonyms=False)].append(index)
        self.tokens = [normalize(item["question"], expand_synonyms=False).split() for item in self.questions]
        self.df = Counter(token for doc in self.tokens for token in set(doc))
        self.avgdl = sum(map(len, self.tokens)) / max(len(self.tokens), 1)
        # A fresher's wording often matches the policy's own vocabulary (e.g.
        # "uniform") only in the answer, not in any reviewed question
        # phrasing. Index answer text as a second, record-level signal so
        # retrieval and confidence can both "see" it.
        self.answer_tokens = [normalize(record.get("answer", ""), expand_synonyms=False).split() for record in self.records]
        self.answer_df = Counter(token for doc in self.answer_tokens for token in set(doc))
        self.answer_avgdl = sum(map(len, self.answer_tokens)) / max(len(self.answer_tokens), 1)
        self.record_to_indices = defaultdict(list)
        for index, item in enumerate(self.questions):
            self.record_to_indices[item["record_index"]].append(index)
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

    def exact_match(self, query: str):
        """Return one reviewed record for an unambiguous exact question.

        Matching uses the same normalisation as the hybrid retriever, so
        casing and punctuation do not make the API behave differently from
        the terminal tester. If wording belongs to more than one record,
        return ``None`` and retain the normal clarification flow.
        """
        candidates = self.exact_questions.get(normalize(query, expand_synonyms=False), [])
        record_indexes = {self.questions[index]["record_index"] for index in candidates}
        if len(record_indexes) != 1:
            return None
        index = candidates[0]
        return {"record": self.records[self.questions[index]["record_index"]],
                "question": self.questions[index]["question"]}

    @staticmethod
    def _bm25_over(query_tokens: list[str], docs: list[list[str]], df: Counter, avgdl: float):
        n = len(docs); scores = []
        for doc in docs:
            counts = Counter(doc); score = 0.0
            for token in query_tokens:
                if not counts[token]: continue
                idf = math.log(1 + (n - df[token] + .5) / (df[token] + .5))
                score += idf * counts[token] * 2.0 / (counts[token] + 1.5 * (1 - .75 + .75 * len(doc) / avgdl))
            scores.append(score)
        return scores

    # Answer text is blended in as raw BM25 score mass rather than a
    # separate RRF vote. RRF only sees rank order, so a middling answer-text
    # match tied for rank 1 could out-rank a phrasing match that is far
    # stronger in absolute terms (tried and reverted -- it let an unrelated
    # record win a query about a completely different policy). Adding raw
    # scores preserves that margin: a strong phrasing match still dominates,
    # while a phrasing with near-zero query overlap can still be carried by
    # strong answer evidence -- the case where a fresher's wording never
    # appears in any reviewed question, only in the policy answer itself.
    ANSWER_WEIGHT = 0.6

    def search(self, query: str, limit: int = 3):
        query_tokens = normalize(query).split()
        phrase_bm25 = self._bm25_over(query_tokens, self.tokens, self.df, self.avgdl)
        answer_bm25 = self._bm25_over(query_tokens, self.answer_tokens, self.answer_df, self.answer_avgdl)
        combined = [phrase_bm25[i] + self.ANSWER_WEIGHT * answer_bm25[self.questions[i]["record_index"]]
                    for i in range(len(phrase_bm25))]
        sparse_order = sorted(range(len(combined)), key=lambda i: combined[i], reverse=True)
        rrf = defaultdict(float)
        for rank, idx in enumerate(sparse_order[:50], 1): rrf[idx] += 1 / (RRF_K + rank)
        if self.dense is not None:
            import numpy as np
            vector = self.model.encode(["Represent this sentence for searching relevant passages: " + normalize(query, expand_synonyms=False)], normalize_embeddings=True)[0]
            dense_order = np.argsort(self.dense @ vector)[::-1].tolist()
            for rank, idx in enumerate(dense_order[:50], 1): rrf[idx] += 1 / (RRF_K + rank)
        # Rankings are over question phrasings, but the user-facing unit is a
        # record. Keep only the strongest phrasing for each record so a
        # clarification never offers the same answer three times.
        #
        # Final order comes from RRF (phrase-BM25 + dense), not from the
        # lexical confidence score below. This was tried the other way --
        # sorting by confidence -- and it broke exactly the case this file
        # exists to fix: "dress code for lab" lost to "gym dress code" /
        # "campus dress code" because those literally contain the word
        # "dress" and a flat token-overlap fraction can't tell that matching
        # the rare, disambiguating word "lab" matters more than matching
        # four near-synonyms of "dress code" shared by unrelated records.
        # Dense embeddings correctly rank the lab record first because they
        # encode the *combination* "lab" + "dress code" as one query, not a
        # bag of independently-scored tokens; RRF preserves that judgment.
        best_for_record = {}
        for idx in sorted(rrf, key=rrf.get, reverse=True):
            record_index = self.questions[idx]["record_index"]
            if record_index not in best_for_record:
                best_for_record[record_index] = idx

        def confidence(idx):
            # Confidence is intentionally only a calibration feature, never a
            # semantic-score proxy. It combines exact query-token coverage and
            # the same blended BM25 evidence (question phrasing plus answer
            # text) used for ranking; thresholds are selected using a
            # labelled set.
            record_index = self.questions[idx]["record_index"]
            query_terms = set(query_tokens) - STOPWORDS
            doc_terms = (set(self.tokens[idx]) | set(self.answer_tokens[record_index])) - STOPWORDS
            overlap = len(query_terms & doc_terms) / max(len(query_terms), 1)
            lexical_strength = 1 - math.exp(-combined[idx] / 6)
            return overlap * lexical_strength

        # RRF's top `limit` is the default result, but the hard cutoff can
        # truncate away a genuinely strong match that RRF ranked just
        # outside it (generic phrasing overlap can out-rank a specific,
        # correct record by a hair -- see "is a printout of my scorecard
        # required" landing club/attendance records ahead of the actual
        # admission-documents record). Only rescue when it matters: if
        # every one of RRF's picks is itself weak evidence (no confident
        # match was found at all), and a pool candidate just outside the
        # window is clearly stronger, swap it in for the weakest pick. When
        # RRF's picks already include something confident (dress code for
        # lab), this never fires, so that ordering is untouched.
        pool = list(best_for_record.values())[:max(limit * 5, 15)]
        ranked = pool[:limit]
        STRONG, RESCUE_MULTIPLIER = 0.3, 1.5
        ranked_scores = [confidence(idx) for idx in ranked]
        if ranked and max(ranked_scores, default=0) < STRONG:
            best_idx, best_score = max(((idx, confidence(idx)) for idx in pool), key=lambda x: x[1])
            if best_idx not in ranked and best_score > max(ranked_scores, default=0) * RESCUE_MULTIPLIER:
                weakest = ranked_scores.index(min(ranked_scores))
                ranked[weakest] = best_idx

        output = []
        for idx in ranked:
            item = self.questions[idx]; record_index = item["record_index"]; record = self.records[record_index]
            output.append({"record": record, "question": item["question"], "score": confidence(idx), "rrf": rrf[idx]})
        return output
