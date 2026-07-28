# PLAN.md — SASTRA Freshers Assistant (RAG)

Target surface: **Unify** (college app, Flutter + Firebase)
Constraint: **fully local / open-source / zero cost**
Language: **English only**
Failure mode: **abstain and route to a human, never guess**

---

## 1. The one rule everything else serves

A fresher acting on a wrong answer is worse off than a fresher who got no answer.

If the bot says outing permission is twice a semester when the 2026-27 rule is once a month, a student plans a trip home and gets refused at the gate. If it says classes start on the wrong date, someone books the wrong train. So the whole design is biased toward **abstention over coverage** — it is acceptable for the bot to answer only 70% of questions, provided the 70% it answers are right.

Concretely: **no free-form generation over retrieved text in v1.** The answers in the dataset are already human-written and verified. The bot's job is to *find the right one and return it*, not to rewrite it.

---

## 2. Why this is not a textbook RAG

Standard RAG splits long documents into chunks, embeds the chunks, retrieves the top chunks, and asks an LLM to compose an answer from them. Every one of those steps is a place hallucination can enter.

Our corpus is different. It is already a set of **question → verified answer pairs** (~435 pairs after the 2026-27 additions are appended). That means:

| Standard RAG | This project |
|---|---|
| Chunk documents by size | No chunking — one pair is one unit |
| Embed the document text | Embed the **question** side, return the **answer** side |
| LLM composes an answer from chunks | Return the stored answer verbatim |
| Hallucination risk: high | Hallucination risk: near zero by construction |
| Failure mode: plausible wrong prose | Failure mode: retrieving the *wrong pair* |

This shifts the entire engineering problem. We are not fighting hallucination, we are fighting **mis-retrieval**. That is a much easier problem to measure and fix, because there is always a single correct answer ID to compare against.

The retrieval task is technically "semantic question matching" — nearest-neighbour search over question phrasings — and the "RAG" label is really just describing the pipeline shape.

---

## 3. Architecture

```
 Flutter app (Unify)
        |
        |  POST /ask  { "query": "when do classes start" }
        v
 +---------------------------+
 |  FastAPI service (Python) |
 +---------------------------+
        |
        +--> [1] Normalize query (lowercase, strip, expand abbreviations)
        |
        +--> [2] Dense retrieval  ---> embed query -> cosine vs. index
        |
        +--> [3] Sparse retrieval ---> BM25 keyword match
        |
        +--> [4] Fuse both rankings (Reciprocal Rank Fusion)
        |
        +--> [5] Confidence gate
        |          |
        |          +-- score >= tau_high --> return stored answer verbatim
        |          +-- tau_low..tau_high  --> return top 3 as "did you mean?"
        |          +-- score <  tau_low   --> ABSTAIN + contact routing
        |
        +--> [6] Log query + outcome (for the improvement loop)
        v
   JSON response -> rendered in Flutter
```

**Index storage:** a single `.npz` file (numpy array of vectors) plus a `.json` of the pairs. At this corpus size no vector database is needed — 1,500 vectors of 384 dimensions is roughly 2.3 MB, and brute-force cosine similarity over that is sub-millisecond. Adding FAISS, Chroma or Pinecone here is cost and complexity for no gain. (Revisit only past ~50,000 pairs.)

---

## 4. Data layer

### 4.1 Current state

| File | Pairs | Style |
|---|---|---|
| `sastra_json.txt` | 319 | Short keyword-ish questions, some multi-phrasing |
| `sastra_1_.json` | 50 | Multi-paraphrase, `" ? "`-separated |
| `sastra_freshers_2026_27.json` | 66 | Multi-paraphrase, `" ? "`-separated |

### 4.2 Target schema

Migrate all three into one canonical format. The extra fields are not bureaucracy — each one prevents a specific way the bot can mislead someone.

```json
{
  "id": "hostel_biometric_attendance",
  "questions": [
    "How is hostel attendance taken?",
    "biometric attendance hostel timing",
    "roll call timings boys and girls"
  ],
  "answer": "Hostel attendance is registered biometrically ...",
  "category": "hostel",
  "source": "Hostel Rules and Regulations 2026-27 (SASTRA official)",
  "source_url": "https://www.sastra.edu/downloads/news/2026/june/Hostel_Rules_Regulations_2026-27.pdf",
  "effective_from": "2026-07-01",
  "valid_until": null,
  "status": "active",
  "last_verified": "2026-07-28",
  "contact_fallback": "deanswas@sastra.edu"
}
```

Field purposes:

- **`questions` as an array** — replaces the `" ? "` delimiter hack. Each phrasing gets embedded *separately*, which is what actually makes paraphrase matching work. Keeping them glued into one string averages the meanings together and degrades retrieval.
- **`status`** — `active` / `superseded`. Superseded pairs stay in the file for history but are **excluded from the index**.
- **`effective_from` / `valid_until`** — date-gating (see 4.4).
- **`source` / `source_url`** — shown to the student so they can verify. This is the single cheapest trust mechanism available.
- **`last_verified`** — drives the annual re-check (see 10).

### 4.3 Resolving the known contradictions

These existing pairs conflict with the 2026 official documents and must be marked `superseded` before indexing, or the bot will confidently serve obsolete rules:

| Existing pairs | Superseded by |
|---|---|
| Attendance below 75% / 80% (`#23`, `#78`, `#79`, `#196`) | Examination Regulations eff. 28 Jan 2026 (graded condonation ladder) |
| Iron box / kettle rules (`#65`, `#115`, `#203`, `#204`) | Hostel Rules 2026-27 (written permission from Dean SWAS) |
| Outing frequency (`#101`, `#104`, `#107`, `#205`) | Hostel Rules 2026-27 (first year: once a month) |
| Manual roll call (`#10`, `#13`) | Biometric attendance, 2026-27 |

**Do this before the first index build, not later.** A RAG that returns both the old and new rule is worse than one that returns neither, because the student has no way to tell which is current.

### 4.4 Date-sensitive content

Several new pairs contain hard dates ("classes commence 5 August 2026"). In August 2027 those become actively harmful.

Rule: any pair whose answer contains a specific calendar date gets a `valid_until`. Past that date the pair is either (a) auto-excluded from the index, or (b) served with a prepended banner: *"This was accurate for the 2026-27 academic year. Check the current academic calendar."* Option (b) is preferable — it still helps, without pretending to be current.

Implement as a check at index-build time, not at query time.

### 4.5 Quality rules for adding new pairs

- One fact per pair. Do not bundle five unrelated rules into one answer.
- Answer must be traceable to a named source. No pair from memory or from an undated forum post.
- Prefer official SASTRA documents. Community sources (Reddit, Quora) are usable only for *experiential* questions ("what is hostel food actually like") and must be labelled as student opinion, never as policy.
- Minimum 4-6 question phrasings per pair, written the way a nervous 17-year-old types — including lowercase, no punctuation, and abbreviations.

---

## 5. Retrieval

### 5.1 Embedding model (dense retrieval)

"Dense retrieval" means converting text to a vector of numbers so that similar meanings sit close together, then finding the closest stored question to the student's query.

Recommended: **`BAAI/bge-small-en-v1.5`** — English-only, 384 dimensions, ~130 MB, Apache-2.0, runs on CPU in ~10-30 ms per query. Fallback: `all-MiniLM-L6-v2` (smaller, slightly weaker).

Check the model card before wiring it up — some `bge` models expect a specific instruction prefix on the *query* side (not on the indexed side), and getting that wrong silently costs accuracy. Verify rather than trusting this document.

### 5.2 Keyword retrieval (BM25)

Dense embeddings are weak on rare tokens. Student queries are full of them: `CUB`, `SWI`, `CIA`, `SCOS`, `nSTORE`, `SWAS`, `TC`, `ABC ID`, `Ahalya`, `Kuruksastra`. An embedding model has barely seen these; a keyword index matches them exactly.

Use **BM25** (a classic keyword-ranking algorithm; `rank_bm25` is a small pure-Python library). Run it in parallel with dense retrieval.

### 5.3 Fusion

Combine the two ranked lists using **Reciprocal Rank Fusion** — score each pair by `sum(1 / (60 + rank_in_that_list))` across both lists. It needs no tuning and no score normalization, which is why it is the right default.

Skipping the hybrid step and going dense-only is the most common way this project would underperform. Build both from day one.

### 5.4 Abbreviation expansion

Maintain a small synonym map applied to the query before retrieval:

```
swi -> student web interface
cub -> city union bank
cia -> continuous internal assessment
swas -> student wellness activities and support
tc  -> transfer certificate
```

This is a dictionary, not a model. Twenty entries will measurably outperform any amount of embedding tuning here.

---

## 6. The confidence gate (the safety-critical part)

Two thresholds, calibrated on real data (see 8):

| Condition | Behaviour |
|---|---|
| `top_score >= tau_high` | Return the stored answer verbatim + source + `last_verified` date |
| `tau_low <= top_score < tau_high` | Return **top 3 questions as tappable suggestions** — do not answer |
| `top_score < tau_low` | Abstain |

The middle band matters more than it looks. Instead of guessing between three plausible matches, the app shows *"Did you mean one of these?"* and lets the student pick. This converts a potential wrong answer into a correct one, and it is trivial to build in Flutter as a list of chips.

**Abstention response template:**

> I don't have a verified answer for that. For hostel matters contact **deanswas@sastra.edu**; for transport, **transport@sastra.ac.in**; for admissions, **admissions@sastra.edu** or **+91 4362 264101**. You can also check www.sastra.edu.

Route the contact by predicted category where possible, and fall back to the general number.

**Never** let the bot answer these categories from retrieval, regardless of score — always route to a human:

- Anything about a specific student's marks, attendance count, fee balance, or disciplinary case
- Medical or mental-health distress (route to the campus hospital and the Tele-MANAS mental wellness line SASTRA links from its own site)
- Ragging complaints (route to the Anti-Ragging Committee, and say so immediately)
- Anything where the student says the bot's answer contradicts what a warden or faculty member told them — the human wins, always

---

## 7. Optional generation layer (v2 — do not build first)

A small local LLM (Qwen2.5-1.5B-Instruct or Llama-3.2-3B, via `llama.cpp`, quantized) can later be added for two things only:

1. **Multi-pair synthesis** — "what do I need to do before day one" touches six pairs.
2. **Tone** — softening a policy quote into a friendlier reply.

If added, the constraints are absolute: the prompt supplies only the retrieved answers; the model is instructed that it may **only** restate and combine them; and the output is rejected if it contains a date, number, email or rule not present in the retrieved text. That last check is a plain string/regex validator, not a judgement call — cheap to write, and it catches the failure mode that matters.

Ship v1 without this. Measure. Only add generation if the evaluation shows synthesis questions are a real fraction of traffic.

---

## 8. Evaluation (build this *before* tuning anything)

Without measurement, every tuning decision is a guess.

### 8.1 Golden set

Collect **150-200 real fresher questions**. Sources: last year's WhatsApp group backlog, the senior-junior mentoring channels, the CTF/club juniors, and your own memory of first year. Do not write them yourself in "clean" English — the value is in the messy phrasing.

Label each with: correct pair `id`, **or** `SHOULD_ABSTAIN` (deliberately include ~30% unanswerable ones — a bot that never abstains is not being tested).

### 8.2 Metrics

| Metric | What it means | Target |
|---|---|---|
| **Hit@1** | Correct pair ranked first | > 85% |
| **Hit@3** | Correct pair in top 3 | > 95% |
| **Wrong-answer rate** | Confidently returned the *wrong* pair | **< 2%** — the number that matters |
| **Abstention precision** | When it abstained, it genuinely had no answer | > 90% |
| **Abstention recall** | Of unanswerable questions, how many it correctly declined | > 80% |

Wrong-answer rate is the release gate. Hit@1 can be traded away to protect it.

### 8.3 Threshold calibration

Sweep `tau_high` and `tau_low` across the golden set and plot wrong-answer rate against coverage. Pick the highest coverage point that keeps wrong-answer rate under 2%. Record the chosen values and the date in the repo — they must be re-swept whenever the embedding model or the corpus changes materially.

---

## 9. Serving and Unify integration

### 9.1 API

```
POST /ask
  { "query": "when do classes start", "session_id": "..." }

200 ->
  { "status": "answered",
    "answer": "...", "source": "...", "source_url": "...",
    "last_verified": "2026-07-28", "confidence": 0.87, "pair_id": "..." }

  { "status": "clarify",
    "suggestions": [ {"pair_id": "...", "question": "..."}, ... ] }

  { "status": "abstained",
    "message": "...", "contacts": [ {...} ] }
```

Three distinct statuses, so the Flutter side can render three genuinely different UIs. Do not collapse them into one text blob.

### 9.2 Hosting (zero cost, always-on)

| Option | Verdict |
|---|---|
| **Oracle Cloud Always Free (ARM VM)** | Best fit — genuinely always-on, generous RAM, no cold starts |
| Hugging Face Spaces (free CPU) | Easiest to deploy; sleeps when idle, so first query is slow |
| Render / Fly.io free tier | Same cold-start problem |
| Firebase Cloud Functions | Poor fit — memory limits and cold starts make loading an embedding model painful |

Recommendation: prototype on Hugging Face Spaces, move to Oracle Always Free for the real launch. Keep Firebase for what it is good at — auth, Firestore for the pair data, analytics, and query logging.

Verify current free-tier terms yourself before committing; these change.

### 9.3 On-device alternative (v3, optional)

Ship a quantized ONNX embedding model plus the precomputed index inside the app. Roughly 30 MB model + 2.3 MB index. Gives zero server cost, zero latency, and offline operation, at the price of a harder Flutter integration and needing a Firestore-triggered index refresh. Worth considering once the corpus stabilizes — not before.

---

## 10. Maintenance loop

The corpus decays. Rules change every July, the calendar changes every year, and a stale bot is a misleading bot.

- **Every query is logged** with its outcome. Abstained queries are the roadmap — cluster them monthly and write pairs for the recurring ones.
- **Thumbs down** in the Flutter UI on every answer. A pair with repeated downvotes gets reviewed, not auto-deleted.
- **Annual July re-verification.** Before each intake, re-fetch the official documents (hostel rules, examination regulations, academic calendar, opening-day circular), diff against the corpus, and update `last_verified` on every pair touched. Pairs not re-verified for over 18 months get flagged in the admin view.
- **Named owner.** This needs one person accountable each year, or it rots. Hand it over deliberately to a junior before you graduate.

---

## 11. Build phases

| Phase | Deliverable | Gate to pass |
|---|---|---|
| **0** | Migrate all 435 pairs to the canonical schema; mark superseded pairs; add sources | Zero contradicting active pairs |
| **1** | Golden set of 150-200 labelled real questions | 30% labelled `SHOULD_ABSTAIN` |
| **2** | Index build script (embed + BM25) and offline retrieval, measured against the golden set | Hit@3 > 95% |
| **3** | Confidence gate + threshold calibration | Wrong-answer rate < 2% |
| **4** | FastAPI service + logging | p95 latency < 500 ms |
| **5** | Flutter integration: three response UIs + thumbs down | Working end to end in Unify |
| **6** | Pilot with ~20 seniors posing as freshers, before the August intake | No wrong answers on policy questions |
| **7** | Launch; monitor abstention clusters weekly for the first month | — |

Phases 0 and 1 are unglamorous and are the ones that determine whether this works. Resist starting at phase 4.

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| Stale rules served as current | `status` + `valid_until` + annual re-verification |
| Near-duplicate pairs compete and split the ranking | Deduplicate at phase 0; check for pairs with cosine > 0.95 |
| Over-confident matching on adjacent topics (girls' vs boys' hostel timings) | These pairs are semantically close and easy to confuse — put both in the golden set deliberately |
| Students treat the bot as authoritative on disciplinary outcomes | Persistent UI disclaimer + hard routing rules in section 6 |
| Corpus is 435 pairs but the question space is much larger | Accept low coverage at launch; grow from the abstention log |
| Project dies after you graduate | Named owner + documented rebuild script in the repo |

---

## 13. Open questions to resolve before phase 0

1. Who owns this after you? Is it a club project (Team 1nf1n1ty?) or personal?
2. Does the Unify team need to approve an external API call, and is there an existing backend you should sit behind?
3. Can you get official sign-off from the SWAS office? A bot quoting hostel policy without the university's awareness is a political risk, and their blessing also gives you a channel for corrections.
4. Where do the pairs live at runtime — bundled in the repo, or in Firestore so non-technical maintainers can edit them?
