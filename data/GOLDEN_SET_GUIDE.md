# Golden-set collection guide

`golden_set.json` is intentionally empty until real fresher wording is
collected with permission. Do not fill it with polished questions written by
the project team: the evaluation must represent how students actually ask.

Each row has exactly two fields:

```json
{
  "query": "mess menu today where",
  "expected_pair_id": "unify_mess_menu"
}
```

For a query that must not be answered from this RAG, use:

```json
{
  "query": "what is my current attendance percentage",
  "expected_pair_id": "SHOULD_ABSTAIN"
}
```

Rules:

- Remove names, register numbers, phone numbers and other personal data.
- Target 150–200 queries, with roughly 30% `SHOULD_ABSTAIN` rows.
- Label each answerable query with an ID from `data/index/records.json`, not a
  question string. The evaluator now rejects inactive or misspelled IDs.
- Include close distinctions deliberately: girls’ vs boys’ hostel matters,
  current vs old calendar information, and UniFy vs SWI questions.
- Keep the template as an example; copy only consented rows into
  `golden_set.json`.
