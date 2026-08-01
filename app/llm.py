"""LLM answer synthesis over already-verified RAG matches.

Two entry points, both grounded the same way:

- `synthesize_answer` rephrases a single record app.main already accepted
  as correct (an exact question match, or matches[0] past TAU_HIGH). The
  LLM has no say in *whether* it's correct.
- `disambiguate_and_answer` is the one case where the LLM does get a say in
  *which* record answers the question: the human-clarify tier (below
  TAU_HIGH), where retrieval alone couldn't separate 2-3 topically close
  candidates. It picks exactly one candidate index, then hands that single
  record to `synthesize_answer` -- two calls, so the rephrase is grounded
  in the chosen record alone and never blends across candidates, which is
  what would break pair_id/source attribution, the thing this codebase
  protects most.

Grounding is always scoped to exactly one record -- the one already
accepted, or the one just picked. Never the wider retrieved pool: the
completeness checks below require every fact in CONTEXT to survive into the
answer, so extra records make that bar unsatisfiable and send nearly every
answer to the verbatim fallback (see `synthesize_answer`).

Neither function runs for mandatory-route responses (ragging, medical,
personal-record, staff-contradiction) -- those never reach retrieval at all.

Every path shares the same grounding contract: any failure, timeout,
explicit "not in context" signal, confused meta-commentary (talking about
the instructions instead of answering), unparseable/out-of-range choice, a
rephrasing whose numbers (fares, times, dates) don't exactly match the
chosen record's raw answer, or one that dropped a named fact (a place, a
brand) or a whole clause falls back to None, and the caller falls back to the deterministic
behaviour it had before this module existed -- the verbatim record, or the
plain clarify suggestion list. The LLM can only change presentation or pick
among human-reviewed candidates; it can never introduce a fact retrieval
didn't already vouch for.

Falling back to verbatim is the safety net, not the working path, and it
had quietly become the working path: nearly every answer a student saw was
raw corpus text. Two things now stand between a failed rephrase and that
fallback, neither of which relaxes the contract above by one check:

- `_grounded_rephrase` reports *why* it failed (see OK/NOT_RELEVANT/
  UNPHRASEABLE/UNAVAILABLE). A NOT_FOUND is the model saying the record
  doesn't answer the question -- information about the record, not the
  wording -- and `disambiguate_and_answer` now acts on it by trying the
  next candidate rather than serving a record its own rephrase just
  disowned.
- `_reword_only` gets one question-free pass at an already-accepted
  record's stored answer. Most failures above are artifacts of the
  *answering* task, so removing the question removes them; the same
  grounding checks still apply to the result.

Verbatim now means what it should: the LLM was unreachable, or it could
not restate a record without breaking one of the checks.

Observed in practice, several distinct failure modes, each handled on its
own terms rather than by one generic "looks wrong" check:

- A small model asked to "answer the question" often drops a secondary
  figure or named fact a record's later sentence contains (e.g. keeps a
  class-timing range but drops "6 hours/day"; keeps two CIA facts but drops
  "best of 2 from three"; keeps two of three recommended eateries but drops
  "Kannappa Hotel" entirely) even though it never invents anything --
  omission, not fabrication. Since a dropped fact can silently change the
  practical completeness of a policy or recommendation answer, that must
  fail the same grounding check as a fabricated one. Because it's a
  distinct, identifiable failure (the exact figure or name is known), it
  gets one retry naming it explicitly. An invented *number* (present in the
  reply but not the source) is caught and not retried -- asking the model
  to "fix" its own fabrication just invites a second, different one. An
  invented *name* is not specifically caught this way: reliably telling a
  fabricated name from a source name legitimately repositioned by the
  rephrase turned out not to be a solvable problem with cheap heuristics
  (see _names_drifted's docstring); that risk is covered by the
  meta-commentary and verbatim-copy checks instead, not by this one.
- A cautious model asked to "keep every fact exactly as given" often
  resolves that by not touching the wording at all, returning the source
  sentence(s) verbatim -- technically grounded, but not what "the LLM
  should send a personalised answer, not copy-paste the RAG's" means. That
  gets its own one-shot retry telling it explicitly not to copy, before the
  caller falls back to plain verbatim (never rejected outright: a correct
  verbatim answer is still better than no answer, it just isn't reported as
  LLM-phrased).
- A confused model sometimes narrates the instructions rather than
  following them (e.g. "Not found is the correct response for your
  question based on the information provided") -- neither the exact
  NOT_FOUND sentinel nor a real answer, but a reply that can still happen
  to repeat every source fact and figure while being useless to a student.
  No retry: a confused reply asked to try again has no reason to stop being
  confused.
- A friendly-toned model sometimes appends a sentence that states no fact at
  all -- "That's a small amount, so keep it handy!", "Do you need help
  planning your trip?" -- which passes every check above (nothing was
  dropped, altered, or copied) while still not being something CONTEXT
  said. Caught by a marker list (`FILLER_MARKERS`) plus an unsourced
  question-mark heuristic, with its own one-shot retry
  (`_retry_drop_commentary`).
- A whole clause carrying neither a digit nor a proper noun can be dropped
  ("Non-vegetarian food is strictly prohibited inside the campus and
  hostels" -- a prohibition, gone) or invented ("This is your best option
  for transportation as buses are usually available and affordable") with
  every number and name still intact, so nothing above sees it. Both are
  measured the same way, by how much of a sentence's substance appears in
  the other text (`_uncovered_sentences`): source-against-reply catches the
  omission and joins the missing-facts retry; reply-against-source catches
  the invention and joins the commentary retry. This is what actually holds
  the line on added commentary -- `FILLER_MARKERS` alone is a blocklist and
  was never going to be complete.

disambiguate_and_answer's three-way return value matters to callers: a
confirmed-but-unphraseable pick (`(chosen, None)`) still means the record
was correctly identified, so the caller serves it verbatim -- it is NOT the
same as `None`, which means no candidate was ever confirmed at all. The end
user is never shown a raw list of candidates to choose from either way: a
confirmed pick is answered (rephrased or verbatim), an unconfirmed one is
routed to a human.
"""
from __future__ import annotations
import re
import requests
from app.config import (LLM_ENABLED, LLM_MAX_CONTEXT_RECORDS, LLM_TIMEOUT_SECONDS,
                        LLM_VETO_OVERRIDE_SCORE, OLLAMA_HOST, OLLAMA_MODEL)

NOT_FOUND = "NOT_FOUND"
NUMBER_RE = re.compile(r"\d+")
INDEX_RE = re.compile(r"INDEX:\s*\[?(\d+|NONE)\]?", re.IGNORECASE)
# A whole reply that is nothing but the choice itself -- "1", "[2]", "2.",
# "NONE" -- with no INDEX: label. See _parse_disambiguation.
BARE_INDEX_RE = re.compile(r"^\W*(\d+|NONE)\W*$", re.IGNORECASE)

# A rephrase that keeps "best of 2" as "best of two" is not drift -- it's the
# same fact in words. Digit extraction alone treats that as a dropped number,
# so small counts get word-formed before comparison. Bounded to what campus
# facts actually use (counts, not compound numbers like "twenty-five"); dates
# and rupee/time figures in this corpus are already written as digits.
NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
    "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
}
NUMBER_WORD_RE = re.compile(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", re.IGNORECASE)

def _extract_numbers(text: str) -> list[str]:
    normalized = NUMBER_WORD_RE.sub(lambda m: NUMBER_WORDS[m.group(1).lower()], text)
    # "August 03" and "August 3" are the same date, so zero-padding is
    # stripped before comparison -- otherwise a rephrase that naturally
    # unpads a padded source date reads as having dropped the number.
    return [number.lstrip("0") or "0" for number in NUMBER_RE.findall(normalized)]

PERSONA_INSTRUCTIONS = (
    "Never output CONTEXT's sentences unchanged -- reword them and you may "
    "address the student directly instead of the third person CONTEXT may "
    "use. But the verdict of every fact must survive completely unchanged: "
    "if CONTEXT says something is or is not allowed, required, optional, "
    "banned, or permitted, your answer must state that same yes/no/"
    "required/optional plainly, with the same strength -- never soften, "
    "hedge, invert, or explain away a rule, and never add encouragement, "
    "opinion, or commentary that is not in CONTEXT. Every fact, number, "
    "date, name, rupee amount, time, or place in your answer must come "
    "from CONTEXT and mean exactly what it meant there -- do not invent, "
    "assume, drop, or alter any of them, even ones that seem secondary to "
    "the question. If CONTEXT is only one plain sentence, a plain one- or "
    "two-sentence answer is correct; do not pad it with invented framing. "
    "Do not add any sentence, clause, or closing remark that summarizes, "
    "evaluates, explains the significance of, or offers further help about "
    "a fact -- for example do not write things like 'that's a small "
    "amount', 'so basically...', 'which means...', 'let me know if...', or "
    "'do you need help with...'. Every sentence you write must itself state "
    "a fact from CONTEXT, nothing else."
)

SYSTEM_PROMPT = (
    "You are the SASTRA Freshers Assistant, helping a first-year student. "
    "Answer ONLY using the facts in CONTEXT below -- never your own "
    "knowledge. " + PERSONA_INSTRUCTIONS + " If CONTEXT does not answer the "
    "question, reply with exactly: " + NOT_FOUND
)

DISAMBIGUATION_SYSTEM_PROMPT = (
    "You are the SASTRA Freshers Assistant. CONTEXT below lists up to three "
    "candidates, numbered [1], [2], [3] -- each is a QUESTION this system has "
    "on file plus the ANSWER stored for it. They were retrieved as merely "
    "similar to the student's question, so they may be about a different "
    "event, group, or topic even when they share words with it (e.g. a "
    "question about documents for admission is NOT an answer to a question "
    "about documents for orientation, even though both mention documents). "
    "Compare the student's question to each candidate's QUESTION first --  "
    "that is the strongest signal of whether the topic actually matches, "
    "not shared keywords in the ANSWER. Decide which ONE candidate, if any, "
    "actually answers the student's question. If you are not confident any "
    "candidate is about the same topic, choose NONE rather than guess.\n\n"
    "Reply with exactly this one line and nothing else:\n"
    "INDEX: <the matching candidate's number, or NONE if none of them answer the question>"
)

def _build_context(answers: list[str]) -> str:
    return "\n".join(f"[{i}] {answer}" for i, answer in enumerate(answers, 1))

def _build_disambiguation_context(candidates: list[dict]) -> str:
    return "\n".join(
        f"[{i}] QUESTION ON FILE: {c['question']}\n    ANSWER: {c['record']['answer']}"
        for i, c in enumerate(candidates, 1)
    )

def _user_content(query: str, context: str, correction: str | None = None) -> str:
    content = f"CONTEXT:\n{context}\n\nSTUDENT QUESTION: {query}"
    return f"{content}\n\n{correction}" if correction else content

def _numbers_drifted(text: str, answers: list[str]) -> bool:
    # The failure mode that matters for a facts assistant isn't verbosity,
    # it's a fare/time/date silently changing in transit -- dropped, or a
    # new one invented (e.g. "around 16 rupees, so keep 20 handy"). The
    # rephrasing must use exactly the same set of numbers as the source
    # *answers* (word-forms normalized, see NUMBER_WORDS) -- checked against
    # the raw text, not the numbered [1]/[2]/[3] prompt labels, which would
    # otherwise leak false digits in.
    source_numbers = set(_extract_numbers("\n".join(answers)))
    return source_numbers != set(_extract_numbers(text))

def _missing_numbers(text: str, answers: list[str]) -> list[str]:
    """Source numbers (word-forms normalized) absent from `text`, in
    first-seen order. An empty result with `_numbers_drifted` true means
    `text` invented a number instead of dropping one -- see module
    docstring for why that's not retried."""
    present = set(_extract_numbers(text))
    seen, missing = set(), []
    for number in _extract_numbers("\n".join(answers)):
        if number not in present and number not in seen:
            seen.add(number)
            missing.append(number)
    return missing

# A whole clause can vanish from a rephrase without touching a single digit
# -- e.g. "...Kannappa Hotel is a great option" dropped entirely from a
# multi-place answer. Numbers alone don't catch that. Named entities (place
# and brand names) are the other cheap, checkable signal: real proper nouns.
#
# Everything below runs on the SOURCE side only -- it decides which words a
# record's author marked as names. The reply is never parsed this way; see
# _missing_names for why doing so rejected valid rephrasings.
#
# A hand-curated stopword list to exclude ordinary sentence-initial capitals
# ("Girls should...", "Parent orientation...") was tried and rejected: it's
# necessarily incomplete (any noun can lead a sentence: "Attendance...",
# "Classes...") and PERSONA_INSTRUCTIONS's own "address the student
# directly" guidance produces contractions ("You'll...") that need
# enumerating one by one. Position is a better, self-maintaining signal: a
# single capitalized word is a name candidate only when it is NOT the very
# first word of a sentence -- ordinary capitalization from sentence-casing
# doesn't happen anywhere else, but a real name can appear anywhere. The one
# exception is a *run* of 2+ consecutive capitalized words starting at
# position 0 (e.g. "SASTRA DEEMED UNIVERSITY..."): that's a strong enough
# signal on its own to count even there, and without it a record whose
# content is entirely its own name would have no checkable facts at all.
#
# Comparison is per-word, not per-phrase: two records may legitimately
# repeat the same name with different neighbours ("...of SASTRA is located
# at SASTRA DEEMED UNIVERSITY...") without that being a second, different
# fact -- grouping into phrases and comparing those as atomic units treated
# that natural repetition as drift.
NAME_RE = re.compile(r"[A-Z][a-zA-Z']+")

# Closed class of English function words (articles, determiners,
# quantifiers, conjunctions, common sentence-opening interjections) that can
# precede a real proper noun without being part of it -- "The SASTRA
# office...", "All SASTRA canteens...", "Yes, SASTRA has...". Unlike the
# open-ended "any noun can start a sentence" problem this deliberately does
# NOT try to solve (see below), this class is small and genuinely closed:
# nothing in it is ever itself a name.
SENTENCE_OPENERS = {
    "the", "a", "an", "this", "that", "these", "those", "all", "every",
    "some", "most", "both", "each", "no", "yes", "well", "also", "only",
    "since", "because", "if", "when", "while", "so", "and", "but", "it",
    "there", "here", "another", "such", "many", "few", "several", "any",
    "more", "other", "first", "second", "third", "next", "last",
}

def _name_candidates(sentence: str) -> list[str]:
    words = sentence.split()
    candidates = []
    for i, word in enumerate(words):
        match = NAME_RE.fullmatch(word.strip("\"'()[],:;.!?-"))
        if match:
            candidates.append((i, match.group(), word))
    # Drop a lone position-0 candidate (a plain sentence-initial capital);
    # keep it if it was part of a 2+-word run starting at position 0. A run
    # requires the words to be directly adjacent with no internal
    # punctuation break ("Yes, SASTRA has..." is index-adjacent (0, 1) but
    # the comma after "Yes" marks a sentence-opening interjection, not the
    # start of a multi-word name) AND the position-0 word itself must not be
    # a known function word ("The SASTRA office..." is index-adjacent with
    # no comma, but "The" is never part of the name that follows it).
    if candidates and candidates[0][0] == 0:
        run_length = 1
        for i in range(1, len(candidates)):
            prev_index, _, prev_raw = candidates[i - 1]
            index, _, _ = candidates[i]
            if index == prev_index + 1 and not re.search(r"[,:;]$", prev_raw):
                run_length += 1
            else:
                break
        if run_length == 1 or candidates[0][1].lower() in SENTENCE_OPENERS:
            candidates = candidates[1:]
    return [name for _, name, _ in candidates]

def _extract_names_in_order(text: str) -> list[str]:
    # Case-insensitive: "Sastra" in CONTEXT and "SASTRA" in a rephrase are
    # the same fact, not a drop-and-reinvent. Original casing is kept for
    # the ordered list (used in retry messages); comparisons elsewhere use
    # the lowercase form.
    seen, ordered = set(), []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        for name in _name_candidates(sentence):
            key = name.lower()
            if key not in seen:
                seen.add(key)
                ordered.append(name)
    return ordered

def _name_key(name: str) -> str:
    """Leading alphanumeric run of a name, lowercased -- so a possessive or
    punctuated form ("Krishna's") still matches the plain word a rephrase
    would use."""
    words = _normalized_words(name)
    return words[0] if words else name.lower()

def _missing_names(text: str, answers: list[str]) -> list[str]:
    # Asymmetric on purpose. The capitalization/position rules above encode
    # which words the *record author* marked as names, so they belong on the
    # source side only. Re-applying them to the rephrase -- which is what
    # this used to do -- required the reply to capitalize a name in the same
    # place the source did, and so counted ordinary re-casing as a dropped
    # fact: "Girls: Kurti with dupatta" rewritten as "girls wear a kurti
    # with a dupatta" keeps every fact, but lowercase "kurti" no longer
    # parsed as a name and the whole answer was rejected. That false
    # positive, not any real omission, is what forced the verbatim fallback
    # on most answers. Presence is therefore tested against every word of
    # the reply, however it happens to be cased or positioned.
    present = set(_normalized_words(text))
    return [name for name in _extract_names_in_order("\n".join(answers))
            if _name_key(name) not in present]

def _names_drifted(text: str, answers: list[str]) -> bool:
    # Deliberately one-directional (missing only, not "any extra name is
    # invented") -- unlike numbers, name detection can't reliably tell a
    # genuine repositioned repeat of a source name (fine) from a truly new,
    # fabricated one (not fine): sentence-position filtering that correctly
    # excludes ordinary capitals like "The" or "Parent" also means the same
    # word can legitimately appear "extra" in a reply for reasons that have
    # nothing to do with fabrication. Completeness (nothing dropped) is the
    # tractable, low-false-positive guarantee this can make; a fabricated
    # name that doesn't touch a number relies on the meta-commentary and
    # verbatim-copy checks, not this one.
    return bool(_missing_names(text, answers))

# Numbers and names are the two *token* signals, and between them they miss
# an entire class of drift: a clause carrying neither. Both directions of it
# were observed once the checks above stopped over-rejecting and answers
# actually started reaching students:
#
# - Omission. "Non-vegetarian food is strictly prohibited inside the campus
#   and hostels" vanished from a rephrase of the food-in-hostel record --
#   a prohibition, silently dropped, with not one digit or proper noun in it
#   to notice. Likewise "The requirement is applied course by course, not on
#   your overall average" from the attendance record.
# - Invention. "This is your best option for transportation as buses are
#   usually available and affordable. Remember to check schedules beforehand
#   so you don't miss your ride." -- appended to a one-line bus-fare record,
#   every number and name intact, none of it anything CONTEXT said.
#
# Both are the same measurement: how much of one sentence's substance
# appears in the other text. Run source->reply it catches omission; run
# reply->source it catches invention. Content words only, since the function
# words are shared by every sentence in English and would wash out the
# signal.
STOPWORDS = SENTENCE_OPENERS | {
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does",
    "did", "have", "has", "had", "can", "could", "will", "would", "shall",
    "should", "may", "might", "must", "of", "to", "in", "on", "at", "for",
    "from", "by", "with", "as", "into", "onto", "up", "down", "out", "off",
    "over", "under", "about", "than", "then", "or", "nor", "not", "you",
    "your", "yours", "they", "them", "their", "we", "our", "he", "she",
    "his", "her", "its", "i", "me", "my", "which", "who", "whom", "whose",
    "what", "where", "why", "how", "s", "t", "re", "ll", "ve", "d",
}

# Below this share of a sentence's content words appearing in the other
# text, the sentence is treated as unmatched. A genuine rephrase re-words
# heavily but keeps the nouns, and lands well above it; a sentence that was
# actually dropped or actually invented scores at or near zero, so the gap
# is wide and the exact value is not delicate.
SENTENCE_COVERAGE = 0.4

# Fewer content words than this and the ratio is too coarse to mean
# anything -- a two-word sentence is 0, 0.5 or 1. Short sentences are left
# to the number and name checks.
MIN_SENTENCE_CONTENT_WORDS = 4

def _content_words(text: str) -> set[str]:
    # Crude depluralization so "buses"/"bus" and "hostels"/"hostel" count as
    # the same word. Over-merging is harmless here: this measures overlap,
    # not identity, and a false *match* only ever makes the check more
    # permissive, never more likely to reject a good answer.
    words = set()
    for word in _normalized_words(text):
        if word in STOPWORDS:
            continue
        if len(word) > 3 and word.endswith("es"):
            word = word[:-2]
        elif len(word) > 3 and word.endswith("s"):
            word = word[:-1]
        words.add(word)
    return words

def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def _uncovered_sentences(source_text: str, against: str) -> list[str]:
    """Sentences of `source_text` whose content words are largely absent
    from `against`."""
    reference = _content_words(against)
    uncovered = []
    for sentence in _sentences(source_text):
        content = _content_words(sentence)
        if len(content) < MIN_SENTENCE_CONTENT_WORDS:
            continue
        if len(content & reference) / len(content) < SENTENCE_COVERAGE:
            uncovered.append(sentence)
    return uncovered

def _dropped_sentences(text: str, answers: list[str]) -> list[str]:
    return _uncovered_sentences("\n".join(answers), text)

def _invented_sentences(text: str, answers: list[str]) -> list[str]:
    return _uncovered_sentences(text, "\n".join(answers))

def _facts_drifted(text: str, answers: list[str]) -> bool:
    return (_numbers_drifted(text, answers) or _names_drifted(text, answers)
            or bool(_dropped_sentences(text, answers)))

def _missing_facts(text: str, answers: list[str]) -> list[str]:
    return _missing_numbers(text, answers) + _missing_names(text, answers)

# Observed in practice: a reply that's neither a clean answer nor the exact
# NOT_FOUND sentinel, but a sentence *about* the instructions -- e.g. "Not
# found is the correct response for your question based on the information
# provided." It can still happen to repeat every source fact and figure
# (so the checks above pass it clean) while being useless, confusing
# meta-commentary rather than an answer a student should ever see. None of
# these phrases have a legitimate reason to appear in a genuine campus-fact
# answer, so any match is treated as a hard failure, same as NOT_FOUND
# itself -- no retry, since a confused reply asked to try again has no
# reason to stop being confused.
META_COMMENTARY_MARKERS = (
    "context", "the information provided", "the correct response",
    "based on the provided", "as an ai", "i cannot", "i'm unable",
)

def _is_meta_commentary(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in META_COMMENTARY_MARKERS)

def _is_unusable(text: str | None) -> bool:
    return not text or text == NOT_FOUND or _is_meta_commentary(text)

# Why a failed rephrase needs a *reason* and not just None. The three ways a
# rephrase fails to come back mean completely different things about the
# record it was grounded in, and the old single None conflated them:
#
# - NOT_RELEVANT (the NOT_FOUND sentinel) is the model's verdict that this
#   record does not answer the question at all. That is information about
#   the *record*, not the wording -- and it was being thrown away. Observed:
#   for "What is the bus fare from SASTRA to Thanjavur?", the picker chose
#   the "bus to thanjavur from sastra" travel record over the literal
#   "What is the bus fare to Thanjavur?" record ranked above it; the
#   rephrase correctly answered NOT_FOUND (that record has no fare in it),
#   and the caller then served that same wrong record verbatim. The student
#   got a travel answer to a fare question, un-personalised. A NOT_RELEVANT
#   verdict must move the search on to the next candidate instead.
# - UNPHRASEABLE means the record is right but the wording kept failing a
#   grounding check. The record stands; only the presentation is missing,
#   which is what the reword pass below exists to recover.
# - UNAVAILABLE means Ollama never answered. Nothing further is worth
#   trying on this request.
OK = "ok"
NOT_RELEVANT = "not_relevant"
UNPHRASEABLE = "unphraseable"
UNAVAILABLE = "unavailable"

# UNPHRASEABLE means "this record answers the question, only the wording
# failed", and callers act on that: the disambiguation walk stops there and
# serves the record. So a reply must actually be an *attempt at restating
# this record* before its failure is allowed to confirm it, and plenty of
# failing replies aren't.
#
# Observed: asked "can I keep a pet dog in the hostel" against the record
# for "can I store non vegetarian food in the hostel", the model replied
# "NO". That is a refusal in the shape of an answer -- it never mentions
# food, it isn't the NOT_FOUND sentinel, and it fails the fact checks only
# because it dropped everything. Reading it as UNPHRASEABLE would have
# confirmed a food record for a pet question and served it verbatim. Same
# for prose hedges like "Exact timings cannot be provided as they differ
# among individual hostels" against the curfew record.
#
# The distinction that separates them from a genuine near-miss (a rephrase
# that kept the record but dropped one of its two figures) is how much of
# the reply is made of the record's own substance. A real restatement is
# almost entirely source content words; a refusal or hedge shares almost
# none. Measured with the same content-word machinery as everything else,
# and against the reply rather than the source -- the source side is what
# the failing grounding checks already cover.
MIN_ENGAGED_CONTENT_WORDS = 4

def _engaged_with_record(text: str, answers: list[str]) -> bool:
    """Whether `text` is recognisably an attempt to restate `answers`, as
    opposed to a refusal, hedge, or one-word reply that happens not to be
    the NOT_FOUND sentinel."""
    content = _content_words(text)
    # Too short to be a restatement of anything. "NO", "Yes", "Not sure" --
    # no record's stored answer is one or two content words long, so this
    # cannot be a near-miss rephrase of one.
    if len(content) < MIN_ENGAGED_CONTENT_WORDS:
        return False
    return len(content & _content_words("\n".join(answers))) / len(content) >= SENTENCE_COVERAGE

# A rephrase can pass every check above (facts intact, not a verbatim copy,
# no confused meta-commentary) and still add things CONTEXT never said --
# "Do you need help calculating...?", "Hey there!", "So basically..." -- a
# friendly-senior voice PERSONA_INSTRUCTIONS invites, taken past rephrasing
# into inventing encouragement/solicitation that isn't a fact but also isn't
# in the source. None of the number/name/verbatim checks catch *addition* of
# non-facts, only subtraction or mutation of real ones, so this needs its
# own check.
FILLER_MARKERS = (
    "do you need", "let me know", "feel free", "hope this helps",
    "any questions", "reach out if", "don't hesitate", "hey there",
    "so basically", "good luck", "great choice", "that's a small amount",
    "that's important", "that's worth", "which means", "so make sure",
    "keep in mind", "keep that in mind", "make sure you", "worth knowing",
    "handy to know", "good to know", "important to note", "important for",
    "important because", "this means", "make sure to",
)

def _has_added_commentary(text: str, answers: list[str]) -> bool:
    lowered = text.lower()
    source = "\n".join(answers).lower()
    # A marker the source itself uses -- "Make sure to carry the originals",
    # "...which means the fee is due" -- is a source fact being repeated, not
    # commentary being added. Only markers absent from the source count,
    # same reasoning as the question-mark heuristic below.
    if any(marker in lowered and marker not in source for marker in FILLER_MARKERS):
        return True
    # The marker list above is a blocklist and will never be complete -- it
    # did not contain "this is your best option for transportation" or
    # "remember to check schedules beforehand", both of which a rephrase of
    # a one-line bus-fare record invented wholesale. A reply sentence whose
    # substance simply isn't in CONTEXT is caught structurally instead of by
    # phrase-matching, which is what actually holds the line here.
    if _invented_sentences(text, answers):
        return True
    # A question mark aimed back at the student ("...around campus. Do you
    # need help calculating...?") is invented solicitation, not a fact --
    # unless the source itself already contains one.
    return "?" in text and not any("?" in answer for answer in answers)

WORD_RE = re.compile(r"[a-z0-9]+")

def _normalized_words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())

def _is_verbatim_copy(text: str, answers: list[str]) -> bool:
    # Catches both an exact copy of one candidate and a copy of the
    # concatenated multi-record context, modulo case/punctuation/whitespace
    # -- what "just copy-pasting the RAG's response" actually looks like.
    text_words = _normalized_words(text)
    if any(text_words == _normalized_words(answer) for answer in answers):
        return True
    return text_words == _normalized_words(" ".join(answers))

def _call_ollama(system_prompt: str, user_content: str) -> str | None:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        # 0.1 was so conservative the model often defaulted to copying
        # CONTEXT verbatim; 0.4 fixed that but bought embroidery along with
        # it (invented framing, added commentary) on short factual answers.
        # Measured personalization rate (llm_disambiguated vs verbatim
        # fallback across a spot-check query set, x3 each) after the
        # _name_candidates position-0 fix and the commentary retry were both
        # in place: 0.15 -> 19/27, 0.2 -> 16/27. Lower temperature now wins
        # on both axes -- less invented commentary to trigger the retry, and
        # (with the false-positive name rejections gone) no longer the
        # verbatim-copying trap it was before those fixes existed.
        "options": {"temperature": 0.15, "num_predict": 300},
    }
    try:
        response = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=LLM_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
    except (requests.RequestException, KeyError, ValueError):
        return None

def _retry_missing_facts(system_prompt: str, query: str, answers: list[str], first_attempt: str) -> str | None:
    missing = _missing_facts(first_attempt, answers)
    dropped = _dropped_sentences(first_attempt, answers)
    # Neither kind of omission found, yet the caller saw drift: the reply
    # invented a number rather than dropping one. Deliberately not retried
    # -- see the module docstring.
    if not missing and not dropped:
        return None
    left_out = []
    if missing:
        left_out.append("these exact figures or names: " + ", ".join(missing))
    if dropped:
        left_out.append("these statements: " + " ".join(f'"{s}"' for s in dropped))
    correction = (
        "Your previous answer left out " + " and ".join(left_out) + ". Rewrite "
        "the full answer to cover every one of them, with their meaning "
        "unchanged, along with everything else already correct. If "
        "CONTEXT still doesn't answer the question, reply with exactly: " + NOT_FOUND
    )
    retry_text = _call_ollama(system_prompt, _user_content(query, _build_context(answers), correction))
    if _is_unusable(retry_text) or _facts_drifted(retry_text, answers):
        return None
    return retry_text

def _retry_personalize(system_prompt: str, query: str, answers: list[str]) -> str | None:
    correction = (
        "Your previous answer just copied CONTEXT's sentences unchanged. "
        "Rewrite it from scratch in your own natural words, addressing the "
        "student directly, as a friendly senior would explain it out loud -- "
        "not a copy of the source wording. Keep every fact, figure, and name "
        "exactly as CONTEXT gives it, and don't drop any of them. If CONTEXT "
        "still doesn't answer the question, reply with exactly: " + NOT_FOUND
    )
    retry_text = _call_ollama(system_prompt, _user_content(query, _build_context(answers), correction))
    if _is_unusable(retry_text):
        return None
    if _facts_drifted(retry_text, answers) or _is_verbatim_copy(retry_text, answers):
        return None
    return retry_text

def _retry_drop_commentary(system_prompt: str, query: str, answers: list[str]) -> str | None:
    correction = (
        "Your previous answer added encouragement, opinions, offers of "
        "further help, or questions back to the student that are not in "
        "CONTEXT. Rewrite it as a plain, direct answer to the question -- "
        "state only what CONTEXT says, with nothing added before or after "
        "it. Keep every fact, figure, and name exactly as CONTEXT gives it. "
        "If CONTEXT still doesn't answer the question, reply with exactly: " + NOT_FOUND
    )
    retry_text = _call_ollama(system_prompt, _user_content(query, _build_context(answers), correction))
    if _is_unusable(retry_text):
        return None
    if _facts_drifted(retry_text, answers) or _is_verbatim_copy(retry_text, answers) or _has_added_commentary(retry_text, answers):
        return None
    return retry_text

# Last resort before the verbatim fallback, and the reason a student now
# sees LLM wording on answers that used to come out as raw corpus text.
#
# It is a strictly easier task than the one above: no question, no CONTEXT
# framing, no "answer the student" instruction -- just "say this same thing
# in different words". Most of the failure modes catalogued in the module
# docstring are artifacts of the *answering* task (the model reaches for the
# question, decides a source clause is irrelevant to it and drops it, or
# appends advice about it), and simply do not arise when there is no
# question in the prompt to reach for. That is why this recovers wording the
# full ladder could not: it is not another attempt at the same prompt.
#
# It cannot fix relevance, only presentation, so it runs ONLY on a record
# something else has already accepted -- the retrieval gate, or a
# disambiguation pick that came back UNPHRASEABLE rather than NOT_RELEVANT.
# Rewording a record that does not answer the question would just be a
# fluent wrong answer.
REWORD_SYSTEM_PROMPT = (
    "Rewrite the MESSAGE below in your own words, the way a friendly senior "
    "student would say it out loud to a first-year student. Output only the "
    "rewritten message and nothing else. Every fact, number, date, name, "
    "rupee amount, time and place in MESSAGE must appear in your rewrite, "
    "meaning exactly what it means there, and every yes/no, required, "
    "optional, allowed or prohibited verdict must stay exactly as strong as "
    "MESSAGE states it -- never soften, hedge, invert, or explain one away. "
    "Do not add, drop, summarize, evaluate, or ask anything, and do not "
    "leave a sentence out. Change the wording, nothing else."
)

def _reword_only(answers: list[str]) -> str | None:
    """One question-free rewording attempt of an already-accepted record.

    Single call by design. The full ladder above has already spent up to two
    calls on this record, and a reword that fails every grounding check is
    not a case where the model needs another nudge -- it is one where the
    verbatim record is the honest thing to show. The checks are the same
    ones the ladder uses, so nothing reaches a student that the ladder would
    have rejected."""
    text = _call_ollama(REWORD_SYSTEM_PROMPT, "MESSAGE:\n" + "\n".join(answers))
    if _is_unusable(text):
        return None
    if _facts_drifted(text, answers) or _is_verbatim_copy(text, answers) or _has_added_commentary(text, answers):
        return None
    return text

def _grounded_rephrase(query: str, answers: list[str]) -> tuple[str | None, str]:
    """The full retry ladder, reporting *why* it failed (see OK/NOT_RELEVANT/
    UNPHRASEABLE/UNAVAILABLE) so callers can tell "wrong record" apart from
    "right record, bad wording". Only NOT_FOUND is a verdict about the
    record; every other failure is about the wording."""
    first = _call_ollama(SYSTEM_PROMPT, _user_content(query, _build_context(answers)))
    if first is None:
        return None, UNAVAILABLE
    if first == NOT_FOUND:
        return None, NOT_RELEVANT
    # Judged on the model's *first* reply, before any retry: that is its
    # unprompted read of this record, and so the honest evidence of whether
    # it engaged with the record at all. A reply that isn't recognisably a
    # restatement of this record cannot be a grounded rephrase of it, so it
    # returns here rather than climbing the ladder -- there is nothing for
    # "you left out these figures" to repair in a reply that is a refusal,
    # and spending two more calls to discover that is latency charged to a
    # student who is going to be answered from a different record anyway.
    if not first or not _engaged_with_record(first, answers):
        return None, NOT_RELEVANT
    if _is_meta_commentary(first):
        return None, UNPHRASEABLE
    text = first
    if _facts_drifted(text, answers):
        text = _retry_missing_facts(SYSTEM_PROMPT, query, answers, text)
        if text is None:
            return None, UNPHRASEABLE
    if _is_verbatim_copy(text, answers):
        text = _retry_personalize(SYSTEM_PROMPT, query, answers)
        if text is None:
            return None, UNPHRASEABLE
    if _has_added_commentary(text, answers):
        text = _retry_drop_commentary(SYSTEM_PROMPT, query, answers)
        if text is None:
            return None, UNPHRASEABLE
    return text, OK

def synthesize_answer(query: str, match: dict | None, *, reword_on_failure: bool = True) -> str | None:
    """Return an LLM-phrased answer grounded in the single record `match`
    that the caller has already accepted as correct, or None to signal the
    caller should fall back to that record's verbatim answer. A reply
    that's just a copy of the source text is retried once, but never fails
    outright for being unparaphrased -- the caller's verbatim fallback
    already covers that case correctly, just without the "llm_synthesized"
    label this function's caller would otherwise attach to it.

    Grounding is deliberately scoped to that one record rather than the
    surrounding retrieved pool. Padding CONTEXT with the runners-up was
    tried, on the theory that more material yields a fuller answer; it did
    the opposite. The completeness check demands every figure and name in
    CONTEXT survive into the reply, so with three records in CONTEXT an
    answer to the student's actual question was scored as having "dropped"
    every fact belonging to the two records that weren't the answer -- an
    unsatisfiable bar that rejected almost every rephrase and fell back to
    verbatim. The runners-up are frequently unrelated anyway (a question
    about canteens retrieving ATM and stationery-shop records), so they
    were never material this function should have been drawing on.

    `reword_on_failure` is the "no student should be reading raw corpus
    text" guarantee: when the question-driven ladder gives up on a record
    the caller has already accepted, `_reword_only` gets one question-free
    pass at the stored answer, and only if that also fails does the caller
    fall back to verbatim. It is turned off by `disambiguate_and_answer`
    while that is still deciding *which* record answers the question --
    there, a failure is evidence to act on, not something to paper over."""
    if not LLM_ENABLED or not match:
        return None
    answers = [match["record"]["answer"]]
    text, reason = _grounded_rephrase(query, answers)
    if text is not None:
        return text
    # NOT_RELEVANT is deliberately not excluded here, unlike inside the
    # disambiguation loop. This function's callers (the exact-question match
    # and the past-TAU_HIGH gate) have already established that this record
    # answers the question by means stronger than a 3B model's opinion, so a
    # NOT_FOUND from it is the model being wrong about relevance -- not a
    # reason to withhold that record's own reworded text.
    if reason == UNAVAILABLE or not reword_on_failure:
        return None
    return _reword_only(answers)

def _parse_disambiguation(text: str) -> int | None:
    # Asked for a single "INDEX: n" line and nothing else, the model very
    # often drops the now-redundant label and answers with just "1" or
    # "[2]". That is a perfectly clear pick and must not be thrown away as
    # unparseable -- doing so abstained on most of this tier and routed
    # answerable questions to a human. The bare form is anchored to the
    # whole reply, so a stray digit inside a sentence of prose still counts
    # as no verdict rather than as a choice.
    stripped = text.strip()
    match = INDEX_RE.search(stripped) or BARE_INDEX_RE.match(stripped)
    if not match or match.group(1).upper() == "NONE":
        return None
    return int(match.group(1))

def disambiguate_and_answer(query: str, candidates: list[dict]) -> tuple[dict, str | None] | None:
    """Given the 2-3 human-clarify-tier candidates, ask the LLM to pick the
    ONE that actually answers the question, then rephrase only that record.

    Those are two calls, deliberately. They were one: a single prompt asking
    for an INDEX line and an ANSWER line together. Splitting them helped
    both halves. The pick becomes a plain classification with a one-line
    answer, which a 3B model does far more reliably than choosing and
    composing at once; and the rephrase becomes an ordinary
    `synthesize_answer` call, so it inherits the full retry ladder
    (missing facts, then verbatim-copy, then added commentary) instead of
    the abbreviated copy this function used to carry. That ladder is where
    most personalised answers are actually won -- first attempts fail a
    grounding check routinely and the retry recovers them -- so the merged
    version was giving up most of its answers to the verbatim fallback. The
    cost is one extra call on this tier only.

    The returned record is not always the picked one: a pick whose rephrase
    comes back NOT_FOUND is treated as disowned and the remaining candidates
    are tried in retrieval order (see the loop below). Whatever comes back
    is the record the answer is attributed to, so callers must read
    pair_id/source/confidence off it and never off `candidates[0]`.

    Three distinct outcomes, which callers must NOT collapse into one:
    - `(match, llm_answer)` -- a record was confirmed (it answered the
      question rather than returning NOT_FOUND) and its rephrase passed
      every grounding check. Caller shows the LLM text.
    - `(match, None)` -- a record was confirmed, but neither the question-
      driven ladder nor the question-free reword could restate it without
      breaking a grounding check. The record is still correct; only the
      wording isn't usable. Caller must show that record's stored answer
      verbatim, NOT fall back to a suggestion list -- the topic is resolved.
    - `None` -- no record was confirmed at all: explicit NONE verdict,
      unparseable or out-of-range index, every candidate answering
      NOT_FOUND, or the call failed outright.
      Nothing here should be shown as an answer, verbatim
      or otherwise. Callers must abstain (route to a human), never guess by
      falling back to matches[0] and never show a raw pick-list -- from the
      end user's side there is no such thing as "choose between topics"."""
    if not LLM_ENABLED or not candidates:
        return None
    capped = candidates[:LLM_MAX_CONTEXT_RECORDS]
    context = _build_disambiguation_context(capped)
    raw = _call_ollama(DISAMBIGUATION_SYSTEM_PROMPT, _user_content(query, context))
    if raw is None:
        return None
    index = _parse_disambiguation(raw)
    # A NONE verdict (or an unparseable/out-of-range one) used to abstain
    # outright. That was the single biggest source of "I don't have an answer
    # for that" on questions the corpus does answer, because of what the
    # picker is actually being asked: "which ONE candidate answers this".
    # For a broad question -- "What are the hostel rules for boys?" -- every
    # candidate answers a *part* (timings, curfew, outings, Wi-Fi) and none
    # answers the whole, so a model told to choose NONE rather than guess
    # correctly chooses NONE, and the student got nothing. Observed
    # alongside the near-identical "What is the hostel rules?" being answered
    # fine, purely because that phrasing happened to retrieve candidates one
    # of which the picker was willing to call a whole answer.
    #
    # A NONE is therefore no longer a verdict to act on, only a failure to
    # choose: fall through to the same per-candidate walk, which asks the
    # far easier question "does THIS record answer the question" one record
    # at a time and answers from the first that says yes. Genuinely
    # unanswerable questions still abstain -- every candidate returns
    # NOT_FOUND and the walk runs out (spot-checked against off-corpus
    # questions: pool location, US visas, exam seat numbers, yesterday's
    # cricket score, all still abstain).
    picked = capped[index - 1] if index is not None and 1 <= index <= len(capped) else None
    # The pick is tried first, then the rest in retrieval order -- because a
    # NOT_FOUND from the rephrase is a second, independent opinion on the
    # pick, and a much better grounded one: the picker sees three candidates
    # at once and answers a comparison, while this asks only "does THIS
    # record answer the question", which a 3B model is far better at.
    #
    # The pick going wrong is not hypothetical. For "What is the bus fare
    # from SASTRA to Thanjavur?" the picker chose the travel record ("bus to
    # thanjavur from sastra", 0.712) over the literal "What is the bus fare
    # to Thanjavur?" record ranked above it (0.992); the rephrase then
    # answered NOT_FOUND, correctly, because that record contains no fare.
    # That verdict used to be discarded and the travel record served
    # verbatim -- the student got the wrong record AND raw corpus text, the
    # two failures this module exists to prevent, from one bad pick.
    #
    # A NOT_RELEVANT verdict therefore moves on to the next candidate;
    # anything else settles here (only NOT_FOUND is a statement about the
    # record -- the rest are about the wording). Ordering the remainder by
    # retrieval rank rather than re-asking the picker keeps this cheap and
    # deterministic: in practice the first probe answers, and the walk costs
    # nothing at all unless a pick was actually wrong.
    order = [picked] + [c for c in capped if c is not picked] if picked else list(capped)
    for candidate in order:
        answers = [candidate["record"]["answer"]]
        text, reason = _grounded_rephrase(query, answers)
        if text is not None:
            return candidate, text
        if reason == UNAVAILABLE:
            return None
        if reason == UNPHRASEABLE:
            # This record does answer the question -- the model engaged with
            # it rather than returning NOT_FOUND -- so the topic is settled
            # and the walk stops. One question-free reword is the last
            # chance at LLM wording before the caller serves it verbatim.
            return candidate, _reword_only(answers)
    # Every candidate was rejected as not answering the question -- but that
    # is one 3B model's opinion, and on its own it is not allowed to throw
    # away strong retrieval evidence. When the best-scoring candidate clears
    # LLM_VETO_OVERRIDE_SCORE it is answered anyway, from the record
    # retrieval ranked best rather than the one the picker liked.
    #
    # This is not a relaxation of grounding: the record is human-reviewed,
    # the answer still comes from that one record, and the reword pass is
    # still held to every check. All that changes is who decides relevance
    # when the two disagree -- which, above this score, is retrieval, exactly
    # as it already is on the past-TAU_HIGH path.
    #
    # Below it, the model's refusal and weak retrieval agree, and that is a
    # real abstention: route to a human.
    best = max(capped, key=lambda c: c["score"])
    if best["score"] >= LLM_VETO_OVERRIDE_SCORE:
        return best, _reword_only([best["record"]["answer"]])
    return None
