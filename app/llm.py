"""LLM answer synthesis over already-verified RAG matches.

Two entry points, both grounded the same way:

- `synthesize_answer` rephrases a single record app.main already accepted
  as correct (an exact question match, or matches[0] past TAU_HIGH). The
  LLM has no say in *whether* it's correct.
- `disambiguate_and_answer` is the one case where the LLM does get a say in
  *which* record answers the question: the human-clarify tier (below
  TAU_HIGH), where retrieval alone couldn't separate 2-3 topically close
  candidates. It walks them in retrieval order, asking of each "does THIS
  record answer the question", and answers from the first that says yes.
  Each probe is grounded in that one record alone and never blends across
  candidates, which is what would break pair_id/source attribution, the
  thing this codebase protects most.

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
- A model answering one of the retry corrections below sometimes recites
  the correction's own list of left-out facts back instead of rewriting
  anything ("3, 2, Continuous, Internal, Assessments, Three, CIAs and these
  statements: ..."). Because the correction is assembled out of the source
  record's own figures and sentences, such a reply passes every check here
  by construction while being no answer at all, so it is rejected on the
  scaffolding phrases it repeats and on the shape of a bare list
  (`_echoes_correction`, `_is_bare_enumeration`). No retry: this is what a
  retry already produced.
- A source qualifier can come back re-punctuated as a sentence of its own --
  "...the problems you face. If possible." -- with every fact intact and a
  dangling fragment where the clause used to be (`_has_dangling_clause`).
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
import contextvars
import re
import time
from contextlib import contextmanager

import requests
from app.config import (LLM_ENABLED, LLM_LATENCY_BUDGET_SECONDS, LLM_MAX_CONTEXT_RECORDS,
                        LLM_TIMEOUT_SECONDS, LLM_VETO_OVERRIDE_SCORE, OLLAMA_HOST, OLLAMA_MODEL)
from app.phrasing import asks_for_value, is_polar_answer

NOT_FOUND = "NOT_FOUND"
NUMBER_RE = re.compile(r"\d+")

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

def _build_context(answers: list[str]) -> str:
    return "\n".join(f"[{i}] {answer}" for i, answer in enumerate(answers, 1))

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
    """Leading alphanumeric run of a name, lowercased and depluralized -- so
    a possessive, punctuated or plural form ("Krishna's", "CIAs") still
    matches the plain word a rephrase would use."""
    words = _normalized_words(name)
    return _stem(words[0]) if words else name.lower()

# A source "name" is only a *candidate* name -- the capitalization rules
# below cannot tell a real proper noun from a title-cased ordinary word, and
# this corpus is full of the latter: "CIA - Continuous Internal Assessments
# are the internal exams" contributes Continuous, Internal and Assessments.
# A rephrase that says "conducted continuously" or "three assessments" has
# dropped nothing, but exact-word matching scored all three as missing,
# which is what sent that record's rephrase into a retry it did not need and
# then to the verbatim fallback. Matching the stem as a prefix covers the
# inflections a rephrase legitimately reaches for (Continuous ->
# continuously, Assessments -> assessment) without loosening the check for
# the case it exists to catch: a genuinely dropped place or brand name
# ("Kannappa") shares no prefix with anything in the reply.
MIN_PREFIX_MATCH_LENGTH = 4

def _name_present(key: str, present: set[str]) -> bool:
    if key in present:
        return True
    return (len(key) >= MIN_PREFIX_MATCH_LENGTH
            and any(word.startswith(key) for word in present))

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
    present = {_stem(word) for word in _normalized_words(text)}
    return [name for name in _extract_names_in_order("\n".join(answers))
            if not _name_present(_name_key(name), present)]

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

def _stem(word: str) -> str:
    """Crude depluralization so "buses"/"bus", "hostels"/"hostel" and
    "CIAs"/"CIA" count as the same word. Over-merging is harmless
    everywhere this is used: they all measure overlap, not identity, and a
    false *match* only ever makes a check more permissive, never more likely
    to reject a good answer."""
    if len(word) > 3 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word

def _content_words(text: str) -> set[str]:
    return {_stem(word) for word in _normalized_words(text) if word not in STOPWORDS}

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

# A retry correction below names the facts the first attempt left out, and a
# small model sometimes "fixes" its answer by reciting that list back
# instead of rewriting anything. Observed, served to a student as the whole
# answer to "When are the CIA exams":
#
#   3, 2, Continuous, Internal, Assessments, Three, CIAs and these
#   statements: "The best of 2 from the three is considered as the final
#   internal marks".
#
# Every check in this module passes it, by construction rather than by
# accident: the correction was assembled out of the source record's own
# figures, names and sentences, so nothing is missing, nothing is invented,
# no number drifted, and it is not a verbatim copy of the record either. It
# is simply not an answer -- it is the scaffolding of the question that was
# put to the model.
#
# Caught two ways because neither is sufficient alone. The markers are the
# phrasing the correction wraps around its list, and catch an echo however
# it is punctuated; the shape check catches a list that was recited without
# the surrounding words. Both are scoped to replies to a correction -- a
# first attempt has no correction to echo, and the shape rule is too blunt
# to apply to answers in general.
CORRECTION_ECHO_MARKERS = (
    "these statements", "these exact figures", "previous answer", "left out",
    "rewrite the full answer", "everything else already correct",
    "with their meaning unchanged",
)

def _echoes_correction(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in CORRECTION_ECHO_MARKERS)

# Leading comma-separated fragments of one or two words each -- "3, 2,
# Continuous, Internal, Assessments, ...". No stored answer in this corpus
# opens that way, and no restatement of one has a reason to.
MIN_ENUMERATED_FRAGMENTS = 3

def _is_bare_enumeration(text: str) -> bool:
    sentences = _sentences(text)
    leading = 0
    for fragment in (sentences[0] if sentences else text).split(","):
        if len(fragment.split()) > 2:
            break
        leading += 1
    return leading >= MIN_ENUMERATED_FRAGMENTS

# A trailing subordinate clause promoted to a sentence of its own: the
# record "Yes, seniors will assist and guide you to solve the problems you
# face, if possible" came back rephrased as "...the problems you face. If
# possible." -- every fact intact, nothing invented, and a dangling fragment
# where the qualifier used to be. Restricted to a short sentence that opens
# with a subordinating word, which is what a split-off clause looks like and
# what a real sentence does not.
DANGLING_OPENERS = {
    "if", "when", "while", "although", "though", "because", "unless",
    "since", "whereas", "however", "therefore", "and", "but", "or", "so",
}
MAX_DANGLING_WORDS = 3

def _has_dangling_clause(text: str) -> bool:
    for sentence in _sentences(text)[1:]:
        words = _normalized_words(sentence)
        if 0 < len(words) <= MAX_DANGLING_WORDS and words[0] in DANGLING_OPENERS:
            return True
    return False

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

# One question's whole allowance of Ollama time. LLM_TIMEOUT_SECONDS caps a
# single call; this caps their sum, which is what a student actually waits
# and what nothing bounded before -- a clarify-tier question spends 4-6
# calls in series, so a per-call limit permits a wait several times its own
# size.
#
# A ContextVar rather than a parameter threaded through the ladder: every
# function between the endpoint and the call would have to carry it purely
# to pass it on, and `ask` is a sync endpoint FastAPI runs in a worker
# thread, so each request reads and writes its own copy.
_deadline: contextvars.ContextVar[float | None] = contextvars.ContextVar("llm_deadline", default=None)

@contextmanager
def latency_budget(seconds: float | None = None):
    """Bound the Ollama time spent inside this block. Callers do not handle
    the expiry: the functions below degrade to what they already do when the
    model is unreachable, which is to serve a retrieved record verbatim."""
    token = _deadline.set(time.monotonic() + (LLM_LATENCY_BUDGET_SECONDS if seconds is None else seconds))
    try:
        yield
    finally:
        _deadline.reset(token)

def _budget_remaining() -> float | None:
    """Seconds of Ollama time left, or None when no budget is in force (the
    terminal tester and the unit tests call in without one)."""
    deadline = _deadline.get()
    return None if deadline is None else deadline - time.monotonic()

def budget_spent() -> bool:
    remaining = _budget_remaining()
    return remaining is not None and remaining <= 0

def _call_ollama(system_prompt: str, user_content: str) -> str | None:
    # Refusing to start a call the budget cannot cover is the point: a
    # student waits for calls that were *started*, so the check belongs
    # here, before the request, not after it.
    remaining = _budget_remaining()
    if remaining is not None and remaining <= 0:
        return None
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
    # A call is also not allowed to run past the budget it started inside,
    # so the per-call timeout shrinks to whatever is left of it.
    timeout = LLM_TIMEOUT_SECONDS if remaining is None else min(LLM_TIMEOUT_SECONDS, remaining)
    try:
        response = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
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
    if _is_unusable(retry_text) or _echoes_correction(retry_text) or _is_bare_enumeration(retry_text):
        return None
    if _facts_drifted(retry_text, answers) or _has_dangling_clause(retry_text):
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
    if _is_unusable(retry_text) or _echoes_correction(retry_text):
        return None
    if _facts_drifted(retry_text, answers) or _is_verbatim_copy(retry_text, answers):
        return None
    if _has_dangling_clause(retry_text):
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
    if _is_unusable(retry_text) or _echoes_correction(retry_text):
        return None
    if _facts_drifted(retry_text, answers) or _is_verbatim_copy(retry_text, answers) or _has_added_commentary(retry_text, answers):
        return None
    if _has_dangling_clause(retry_text):
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
    if _has_dangling_clause(text):
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
    # A dangling clause is a wording defect, not a verdict on the record, so
    # it reports UNPHRASEABLE and lets the question-free reword have its
    # pass -- the fragment is an artifact of re-punctuating a source
    # qualifier while answering, and usually does not survive a rewrite that
    # has no question to answer.
    if _has_dangling_clause(text):
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

# Retrieval matches on topic and has no notion of what *kind* of question
# was asked, so a record whose stored answer is written as a reply to a
# yes/no question ("Yes, boys are allowed to go outside the hostels, but
# they must return before 9:30 PM") gets served to a student who asked for a
# value: "What are the hostels available for boys?", "How many hostels are
# there for boys?" -- both answered with that same sentence, which names no
# hostel and counts nothing, while the record that does answer both
# (AHALYA and ARUNDHATHI) sat in the same index.
#
# On its own a shape mismatch is weak evidence and must not veto anything:
# "What are the canteens available at campus" is answered perfectly well by
# a record that happens to open "Yes, SASTRA has multiple dining options.
# Popular spots are...". It only carries weight in combination with the
# model having declined to phrase that record as an answer at all -- at
# which point nothing is left vouching for relevance except lexical overlap
# with a record that answers a question the student did not ask.
#
# `asks_for_value`, not "isn't polar": a bare topic ("Seniors in SASTRA")
# demands no particular value, so a yes/no record can answer it and must not
# be discarded here. Tried the wider rule first and it abstained on exactly
# that query, which the corpus does answer.
def _shape_mismatch(query: str, record: dict) -> bool:
    return asks_for_value(query) and is_polar_answer(record.get("answer", ""))

def _out_of_time(query: str, capped: list[dict]) -> tuple[dict, None] | None:
    """What a student gets when the latency budget runs out mid-walk: the
    best retrieved record, served verbatim -- exactly what this service
    returned before any LLM work existed, and what it already falls back to
    whenever a rephrase fails a grounding check.

    Deliberately not an abstention. Retrieval put these candidates through
    the same TAU_LOW gate as always and they are human-reviewed records; the
    clock running out says nothing about whether they answer the question,
    so "I don't have an answer for that" would be a worse and less honest
    reply than the answer sitting in the index.

    Deliberately still shape-checked. A yes/no-shaped record served raw to a
    student who asked "how many" is the bug this module was just fixed for,
    and a timeout is no reason to reintroduce it -- so those are skipped and
    the best remaining record is used, or nothing if that leaves none."""
    for candidate in capped:
        if not _shape_mismatch(query, candidate["record"]):
            return candidate, None
    return None

def disambiguate_and_answer(query: str, candidates: list[dict]) -> tuple[dict, str | None] | None:
    """Given the 2-3 human-clarify-tier candidates, find the one that
    actually answers the question and rephrase only that record.

    This walks the candidates in retrieval order, asking of each "does THIS
    record answer the question" and answering from the first that says yes.
    It used to spend one call ahead of the walk on a separate picker -- a
    prompt showing all the candidates' questions on file at once and asking
    for the index of the matching one -- whose only effect was to move its
    choice to the front of this same walk.

    That call was removed for latency. Measured on qwen2.5:3b over a
    14-query set on the production VPS, it cost a whole extra sequential
    call on every clarify-tier question: p90 30.7s against 18.8s without it,
    3.2 calls per query against 2.3. Thirteen of the fourteen queries
    resolved to exactly the same record either way, because retrieval order
    and the picker's choice usually agree.

    The fourteenth is the known cost, and it is a real one. "When are the
    CIA exams" resolves to the record explaining what a CIA is rather than
    the one holding the exam dates: retrieval ranks those two the other way
    round, and comparing the student's question against each candidate's
    question on file -- which is what the picker did, and the only signal
    that separates "when" from "what" here -- is exactly what is now gone.
    So a question whose wording shares nothing lexically with the record
    that answers it ("when" against "dates") can land on a neighbouring
    record on the same topic. The answer is still a true, human-reviewed one
    about the thing asked about; it may answer a different question about
    it. Restoring the picker without paying for it serially was tried by
    overlapping the two calls concurrently, and made things much worse on
    this CPU-only box (see git history): two concurrent 3B calls each get
    half the cores, individual calls then exceed LLM_TIMEOUT_SECONDS, and
    answered dropped from 12/14 to 6/14.

    Whatever comes back is the record the answer is attributed to, so
    callers must read pair_id/source/confidence off it and never off
    `candidates[0]`.

    Three distinct outcomes, which callers must NOT collapse into one:
    - `(match, llm_answer)` -- a record was confirmed (it answered the
      question rather than returning NOT_FOUND) and its rephrase passed
      every grounding check. Caller shows the LLM text.
    - `(match, None)` -- a record was confirmed, but neither the question-
      driven ladder nor the question-free reword could restate it without
      breaking a grounding check. The record is still correct; only the
      wording isn't usable. Caller must show that record's stored answer
      verbatim, NOT fall back to a suggestion list -- the topic is resolved.
    - `None` -- no record was confirmed at all: every candidate answered
      NOT_FOUND, or the call failed outright.
      Nothing here should be shown as an answer, verbatim
      or otherwise. Callers must abstain (route to a human), never guess by
      falling back to matches[0] and never show a raw pick-list -- from the
      end user's side there is no such thing as "choose between topics"."""
    if not LLM_ENABLED or not candidates:
        return None
    capped = candidates[:LLM_MAX_CONTEXT_RECORDS]
    # A NOT_RELEVANT verdict moves on to the next candidate; anything else
    # settles here (only NOT_FOUND is a statement about the record -- the
    # rest are about the wording). Walking in retrieval rank keeps this
    # cheap and deterministic, and in practice the first probe answers, so
    # the rest of the walk usually costs nothing at all.
    #
    # Asking each record on its own is the whole point, and is why dropping
    # the picker costs so little: "does THIS record answer the question" is
    # a far easier question for a small model than "which of these three
    # does", and it is asked against the record's own answer text rather
    # than a list. A record the walk disowns is one this same ladder just
    # refused to phrase as an answer -- a better-grounded verdict than a
    # comparison made before any of them had been read closely.
    for candidate in capped:
        # Out of time: stop probing and serve what retrieval already found,
        # verbatim. See `_out_of_time` -- this is a fallback to the answer
        # the index holds, never an abstention.
        if budget_spent():
            return _out_of_time(query, capped)
        answers = [candidate["record"]["answer"]]
        text, reason = _grounded_rephrase(query, answers)
        if text is not None:
            return candidate, text
        if reason == UNAVAILABLE:
            # Told apart by the clock: a call the budget refused to start
            # looks identical to one Ollama never answered, but the first
            # means "we ran out of time to phrase a record we have" and the
            # second means "the model is gone". Only the latter abstains.
            return _out_of_time(query, capped) if budget_spent() else None
        if reason == UNPHRASEABLE:
            # This record does answer the question -- the model engaged with
            # it rather than returning NOT_FOUND -- so the topic is settled
            # and the walk stops. One question-free reword is the last
            # chance at LLM wording before the caller serves it verbatim.
            reworded = _reword_only(answers)
            # Unless what the caller would then serve is a yes/no answer,
            # raw, to a student who asked for a value. Engagement is the
            # only thing vouching for this record, and it is the weaker
            # signal of the two here: nothing that could be restated as an
            # answer to "how many" plus a stored answer shaped as a reply to
            # "are they allowed" means the walk has not found the record yet.
            if reworded is None and _shape_mismatch(query, candidate["record"]):
                continue
            return candidate, reworded
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
    #
    # Which record the override answers from matters as much as whether it
    # fires. Every candidate has refused the question by now, so the choice
    # rests entirely on retrieval score -- and a score is topical overlap,
    # which "hostels ... boys" earns just as easily from the outing-time
    # record as from the one that names the hostels. Those two really do
    # arrive a hair apart (0.609 and 0.597 in production), and taking the
    # maximum answered "What are the hostels available for boys?" with "Yes,
    # boys are allowed to go outside the hostels, but they must return
    # before 9:30 PM".
    #
    # A stored answer written as a reply to a yes/no question is the second
    # signal, independent of the score, that a record answers something the
    # student did not ask -- so those records are set aside here and the
    # override answers from the best-scoring record that remains. Setting
    # aside rather than abstaining is the point: the record that does answer
    # the question is usually still in the pool, a fraction of a point
    # behind, and abstaining throws it away along with the wrong one.
    #
    # The override's own confirmed cases are unaffected: "whether stationery
    # shop is available" is itself a yes/no question, so nothing is set
    # aside for it, and "hostel rules for boys" and "girls' outing rules"
    # resolve to records that do not open with a verdict. If every candidate
    # is set aside, or the best survivor is weak, this abstains exactly as
    # before.
    eligible = [c for c in capped if not _shape_mismatch(query, c["record"])]
    if eligible:
        best = max(eligible, key=lambda c: c["score"])
        if best["score"] >= LLM_VETO_OVERRIDE_SCORE:
            return best, _reword_only([best["record"]["answer"]])
    return None
