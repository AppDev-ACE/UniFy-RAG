"""Question shape, and the answer wording that depends on it.

Most of this corpus was written as replies to yes/no questions -- 81 of the
368 stored answers literally begin "Yes, " or "No, " -- because that is how
the fresher whose question got reviewed happened to phrase it. Retrieval
matches on topic, not on question shape, so those records reach students who
asked a different *kind* of question, and what comes back reads as the answer
to a question nobody asked:

    "What are the hostels available for boys?"
    -> "Yes, boys are allowed to go outside the hostels, but they must
        return before 9:30 PM."

Two separate failures are stacked there, and the shape mismatch is the one
this module exists for: even when the record is right, a polar opener on a
what/when/how-many question is wrong. Observed with the *correct* record
too -- "What is the hostel rule for girls?" answered "Yes, the hostel rule
for girls is that they should be in the hostel within 6 PM", where the
stored answer ("Girls should be in the hostel within 6 PM") has no "Yes" in
it at all: the model added one, because it is told to preserve CONTEXT's
yes/no verdict and reads that as an instruction to lead with one.

Two callers act on the distinction, and deliberately no third: teaching the
persona prompt to match the student's question shape was tried first and
made a 3B model measurably worse. Told not to answer a what/when/how
question with a verdict, it started inventing hedges instead of declining --
"Whether seniors are friendly?" against the anti-ragging record went from a
correct NOT_FOUND to "Not all seniors are necessarily friendly; however..."
on every attempt, a claim CONTEXT does not make. Stripping the opener after
the fact costs the model nothing and cannot invent anything.

- app.llm refuses to serve a polar-shaped record to a non-polar question
  when the model could not restate it as an answer either -- a record that
  answers a yes/no question, offered raw to a student who asked "how many",
  is much more likely to be the wrong record than a wording problem.
- app.main drops a leading "Yes, "/"No, " from what the student is shown
  when the question was not a yes/no one. Only what is *displayed* changes:
  `raw_answer` still carries the stored text verbatim, so provenance,
  pair_id attribution and the review loop are untouched.
"""
from __future__ import annotations
import re

WORD_RE = re.compile(r"[a-z']+")

# A question opening with one of these asks for a value -- a name, a count,
# a time, a place -- and a bare "Yes" or "No" supplies none of them.
WH_OPENERS = {
    "what", "whats", "when", "where", "which", "who", "whom", "whose",
    "how", "why", "name", "list", "tell",
}

# ...and one opening with one of these asks for a verdict, where "Yes" or
# "No" is the answer and must be kept. "whether" belongs here: "whether
# stationery shop is available" is a yes/no question with the auxiliary
# dropped, which is ordinary fresher phrasing in this corpus.
POLAR_OPENERS = {
    "is", "are", "was", "were", "am", "be", "do", "does", "did", "can",
    "could", "will", "would", "shall", "should", "may", "might", "must",
    "has", "have", "had", "whether", "if", "any", "anyone", "isnt", "arent",
    "dont", "doesnt", "cant", "wont",
}

# Skipped before the opener is read, so "So can I..." and "Please is there..."
# are classified on the word that actually carries the question's shape.
LEAD_INS = {
    "please", "kindly", "hey", "hi", "hello", "so", "and", "but", "also",
    "actually", "just", "sir", "madam", "bro", "ok", "okay", "well",
}

def _opener(query: str) -> str | None:
    words = WORD_RE.findall(query.lower())
    while words and words[0] in LEAD_INS:
        words = words[1:]
    return words[0] if words else None

def is_polar_question(query: str) -> bool:
    """Whether `query` asks for a yes/no verdict rather than a value.

    Deliberately decided on the opener alone. A student's question is a
    single short line and its shape lives in its first word.
    """
    opener = _opener(query)
    return opener is not None and opener not in WH_OPENERS and opener in POLAR_OPENERS

def asks_for_value(query: str) -> bool:
    """Whether `query` demands a specific value -- a name, a count, a time,
    a place -- so that a yes/no answer cannot possibly satisfy it.

    Narrower than `not is_polar_question(...)`, and the two are used for
    deliberately different jobs. A large share of what students type is
    neither shape: "Seniors in SASTRA", "hostel gate closing time" -- a bare
    topic, where any relevant fact is a fine answer. Leading such a reply
    with "Yes" is still odd wording, so `strip_polar_prefix` drops the
    opener there; but that is cosmetic and reversible, whereas refusing to
    serve a record on shape grounds throws an answer away. Only an explicit
    what/when/where/which/how-many question is allowed to do the latter.
    """
    return _opener(query) in WH_OPENERS

# Only the comma form is treated as a strippable opener. The bare-sentence
# form -- "No. It is part of the tuition fee only." -- carries the entire
# verdict in that one word, and removing it would invert the answer; the
# comma form is always followed by a clause that restates the verdict in
# full ("No, day scholars are not allowed...").
LEADING_POLAR_RE = re.compile(r"^\s*(yes|no)\s*,\s*(?=\S)", re.IGNORECASE)

POLAR_ANSWER_RE = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)

# A "No, ..." opener is only dropped when the clause behind it still says no
# by itself. Belt and braces on top of the comma-form rule above: a stripped
# negative that lost its negation would be the one edit here that could
# change what a student is told, so it is checked directly rather than
# assumed.
NEGATIONS = {
    "not", "no", "never", "cannot", "cant", "nor", "none", "neither",
    "prohibited", "banned", "forbidden", "barred", "denied", "restricted",
    "without", "unless", "avoid", "prohibit", "prohibits",
}

def is_polar_answer(answer: str) -> bool:
    """Whether `answer` is written as the reply to a yes/no question."""
    return bool(POLAR_ANSWER_RE.match(answer or ""))

def strip_polar_prefix(answer: str, query: str) -> str:
    """Drop a leading "Yes, "/"No, " when the student did not ask a yes/no
    question. Returns `answer` unchanged in every other case."""
    if not answer or is_polar_question(query):
        return answer
    match = LEADING_POLAR_RE.match(answer)
    if not match:
        return answer
    rest = answer[match.end():]
    if match.group(1).lower() == "no":
        first_sentence = re.split(r"(?<=[.!?])\s", rest, maxsplit=1)[0]
        if not set(WORD_RE.findall(first_sentence.lower())) & NEGATIONS:
            return answer
    return rest[:1].upper() + rest[1:]
