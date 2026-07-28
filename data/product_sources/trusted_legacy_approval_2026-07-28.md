# Trusted legacy corpus approval

On 2026-07-28, the UniFy project owner approved serving the non-superseded
legacy records in `data/corpus.json` in the UniFy production experience.

This approval means the records are project-owner-trusted content. It does not
convert them into official SASTRA policy sources. The RAG API marks these
records as `trusted_legacy`, and their source validation should be completed
when current official documents are available.

Records marked `superseded` remain excluded because they conflict with known
2026 official rules.
