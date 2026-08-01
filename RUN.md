# Run locally and connect UniFy

This guide runs the Freshers Assistant API on a development machine and shows
how UniFy should call it. The service returns verified stored text only; it
does not generate answers.

## 1. First-time setup

Run these commands from the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/migrate_corpus.py
HF_HUB_OFFLINE=1 .venv/bin/python scripts/build_index.py
```

`HF_HUB_OFFLINE=1` keeps the index build local after the embedding model has
already been downloaded. For the first build, omit that prefix if the BGE model
is not present in the local cache:

```bash
.venv/bin/python scripts/build_index.py
```

The default production build includes all trusted, non-superseded records.
Current output should say `301 records` and `850 phrasings`. Use
`--official-only` only when you explicitly want to exclude trusted legacy
records.

### Test every legacy answer locally

For exploratory testing only, you can include the legacy records that still
need source verification:

```bash
.venv/bin/python scripts/build_index.py --include-needs-review
.venv/bin/python scripts/chat.py
```

Every such answer is marked **TEST MODE ONLY — unverified** when shown. This
mode never includes records marked `superseded`, because they are known to be
outdated or contradictory. Before any UniFy deployment, rebuild normally
without the flag:

```bash
.venv/bin/python scripts/build_index.py
```

### Trusted legacy answers in production

The default build includes the project-owner-approved, non-superseded legacy
corpus. These records return `"verification_status": "trusted_legacy"` so the
client can retain their provenance label. Superseded records remain excluded.

### Deploy or update the server

`data/index/` is a generated directory and is deliberately not committed to
Git. After every code or corpus deployment, rebuild the index on the server
before restarting Uvicorn. The default command includes trusted legacy records
(including the non-vegetarian-food answer):

```bash
.venv/bin/python scripts/build_index.py
```

Do not use `--include-needs-review` on a deployed server; it is local testing
only.

## 2. Start the local LLM (optional but recommended)

`POST /ask` sends top RAG matches to a locally hosted Ollama model, which
either rephrases an already-confident match or, for a close-call clarify-tier
query, picks which of 2-3 candidates actually answers it and rephrases only
that one — strictly grounded in the stored text either way (see the README's
"LLM answer synthesis" section for the grounding guarantees). It never
supplies a fact of its own, and every fallback path works with no LLM
running at all — this step is optional.

```bash
ollama pull qwen2.5:3b
ollama serve   # usually already running as a background service after install
```

Verify it's reachable:

```bash
curl http://127.0.0.1:11434/api/tags
```

By default the API talks to `http://127.0.0.1:11434` and uses the
`qwen2.5:3b` model — override with the `OLLAMA_HOST` and `OLLAMA_MODEL`
environment variables, or set `LLM_ENABLED=0` to skip LLM synthesis and
always serve verbatim stored answers.

## 3. Start the API

For testing on the same computer:

```bash
.venv/bin/uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Check it in a second terminal:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"ok":true,"index_ready":true}
```

Interactive endpoint documentation is available at
`http://127.0.0.1:8000/docs` while the server is running.

## 4. Test an ask flow

### Easiest: interactive terminal chat

You do not need `curl` to try the RAG. Build the index once, then run:

```bash
.venv/bin/python scripts/chat.py
```

Type a question such as `Where can hostellers see the mess menu?`. You never
see a raw list of topics to pick from: below auto-answer confidence, the
local LLM is given the close candidates and either confirms one (answer
shown, rephrased or verbatim) or the query is routed to a human contact.
Type `exit` when finished.

This preserves the safety gate: the LLM never blends facts across candidates
and every reply is checked against the record it claims to be grounded in
before being shown. After calibration, sufficiently confident questions are
answered immediately without needing the LLM to pick between candidates.

### API test (optional)

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"Where can hostellers see the mess menu?","session_id":"local-test-1"}'
```

The response always has one of these statuses:

| Status | UniFy behaviour |
| --- | --- |
| `answered` | Render `answer`, plus source and last-verified date. Show the optional warning if present. |
| `abstained` | Render `message` and the human contact options. Do not replace this with a guessed answer. |

There is no `clarify` status in the live flow anymore: below auto-answer
confidence, the LLM itself is given the close candidates and must confirm
exactly one before anything is shown to the student. If it can't confirm
one, the query is `abstained`, not offered as a list to choose from.

An `answered` response also includes `"answer_mode"`, one of:

- `"llm_synthesized"` — an exact match or high-confidence record, rephrased
  by the local LLM, grounded strictly in that one record.
- `"llm_disambiguated"` — retrieval alone couldn't separate 2-3 close
  candidates; the LLM picked the one that actually answers the question and
  rephrased only that record. `pair_id`, `source`, and `raw_answer` reflect
  whichever record it picked.
- `"verbatim"` — the stored answer, unchanged. This is what you always get
  with no LLM running. It's also what you get when the LLM's rephrase fails
  the grounding check on the high-confidence path, or when the LLM
  confirmed a clarify-tier candidate but its rephrase never passed grounding
  (the candidate is still correct, only its wording wasn't usable).

`"raw_answer"` always carries the original stored text for audit, even when
`"answer"` was rephrased or LLM-picked.

Automatic `answered` results from plain retrieval confidence (not LLM
confirmation) are intentionally disabled until the project has a real
labelled golden set and calibrated thresholds. An exact normalised match to
one indexed question is returned directly, because it is a lookup of the
reviewed phrase rather than a fuzzy retrieval decision. Before calibration,
other relevant candidates return `answered` with `"llm_disambiguated"` (or
`"verbatim"` if the LLM confirmed a candidate but couldn't rephrase it), or
`abstained` when the LLM can't confirm any candidate answers the question.

`GET /pairs/{pair_id}` still exists for direct lookups by ID, but nothing in
the current `/ask` flow returns a suggestion list to call it from.

In test mode, an `answered` response includes
`"verification_status": "unverified_test_only"`. UniFy must display that
label and warning prominently. Only `"verified"` records may be presented as
official information in a real release.

Owner-approved legacy content uses `"verification_status": "trusted_legacy"`.
It is available for production display, but it must not be styled as an
officially sourced SASTRA policy answer.

To record a thumbs-up or thumbs-down after displaying an answer:

```bash
curl -X POST http://127.0.0.1:8000/feedback \
  -H 'Content-Type: application/json' \
  -d '{"rating":"up","pair_id":"unify_mess_menu","session_id":"local-test-1"}'
```

Queries and feedback are appended locally to `data/logs/queries.jsonl`.

## 5. Call it from UniFy (Flutter)

Create one API client with a configurable base URL. Use `127.0.0.1` only for a
desktop Flutter app running on the same machine. For an Android emulator use
`10.0.2.2`; for a physical phone, start the server on the computer's LAN IP and
put both devices on the same Wi-Fi network.

To expose the local server to a phone on your LAN:

```bash
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then configure UniFy with a URL such as `http://192.168.1.25:8000`. This is for
local development only. A production deployment must use HTTPS and a stable
server URL.

Example Flutter client (add the `http` package to UniFy if it is not already
present):

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

class FreshersAssistantApi {
  FreshersAssistantApi(this.baseUrl);
  final String baseUrl;

  Future<Map<String, dynamic>> ask(String query, String sessionId) async {
    final response = await http.post(
      Uri.parse('$baseUrl/ask'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'query': query, 'session_id': sessionId}),
    );
    if (response.statusCode != 200) throw Exception('Assistant unavailable');
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  // GET /pairs/{pair_id} still exists for a direct lookup by ID, but ask()
  // no longer returns a suggestion list to call it from -- nothing in the
  // standard flow needs this anymore.
  Future<Map<String, dynamic>> getPairById(String pairId) async {
    final response = await http.get(Uri.parse('$baseUrl/pairs/$pairId'));
    if (response.statusCode != 200) throw Exception('Verified answer unavailable');
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  Future<void> sendFeedback({required String rating, String? pairId, String? sessionId}) async {
    await http.post(
      Uri.parse('$baseUrl/feedback'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'rating': rating, 'pair_id': pairId, 'session_id': sessionId}),
    );
  }
}
```

Suggested UI handling:

```dart
final result = await api.ask(query, sessionId);
switch (result['status']) {
  case 'answered':
    // Show result['answer'] exactly as received; show source and last_verified.
    break;
  case 'abstained':
    // Show result['message'] and result['contacts']; offer Community Page too.
    break;
}
```

Do not send a register number, password, marks, attendance count, fee balance,
or other personal academic data to this endpoint. The RAG is for general,
verified information; personal cases are routed to a human.

## 6. Before testing automatic answers

Add consented, real fresher questions to `data/golden_set.json`, label the
correct pair or `SHOULD_ABSTAIN`, then run:

```bash
.venv/bin/python scripts/evaluate.py --sweep --write-calibration
```

The command writes `data/calibration.json` only when the wrong-answer release
gate is below 2%. Restart the API after that file is created. Do not manually
lower the thresholds just to make the bot answer more often.

See `data/GOLDEN_SET_GUIDE.md` and `data/golden_set_template.json` for the
required format and collection rules. The evaluator rejects a misspelled or
inactive expected pair ID.
