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

## 2. Start the API

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

## 3. Test an ask flow

### Easiest: interactive terminal chat

You do not need `curl` to try the RAG. Build the index once, then run:

```bash
.venv/bin/python scripts/chat.py
```

Type a question such as `Where can hostellers see the mess menu?`. If the
system finds a close match but has not yet been calibrated to auto-answer, it
shows numbered verified topics. Type `1` to select one and it prints the stored
answer, source, and verification date. Type `exit` when finished.

This preserves the safety gate: it never silently chooses between close topics.
After calibration, sufficiently confident questions are answered immediately.

### API test (optional)

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"Where can hostellers see the mess menu?","session_id":"local-test-1"}'
```

The response always has one of these statuses:

| Status | UniFy behaviour |
| --- | --- |
| `answered` | Render `answer` verbatim, plus source and last-verified date. Show the optional warning if present. |
| `clarify` | Render the supplied suggestion questions as tappable chips/list items. Do not show an answer yet. |
| `abstained` | Render `message` and the human contact options. Do not replace this with a guessed answer. |

Automatic `answered` results are intentionally disabled until the project has a
real labelled golden set and calibrated thresholds. An exact normalised match
to one indexed question is returned directly, because it is a lookup of the
reviewed phrase rather than a fuzzy retrieval decision. Before calibration,
other relevant candidates return `clarify`, or `abstained` when there is no
relevant candidate.

When a student taps a suggestion, use its `pair_id` exactly once:

```bash
curl http://127.0.0.1:8000/pairs/unify_mess_menu
```

In test mode, both answer and suggestion objects include
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

## 4. Call it from UniFy (Flutter)

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

  Future<Map<String, dynamic>> selectSuggestion(String pairId) async {
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
  case 'clarify':
    // Show result['suggestions']; a tap calls api.selectSuggestion(pair_id).
    break;
  case 'abstained':
    // Show result['message'] and result['contacts']; offer Community Page too.
    break;
}
```

Do not send a register number, password, marks, attendance count, fee balance,
or other personal academic data to this endpoint. The RAG is for general,
verified information; personal cases are routed to a human.

## 5. Before testing automatic answers

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
