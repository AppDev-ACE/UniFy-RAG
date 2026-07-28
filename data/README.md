# Corpus review

The raw file did not include official sources, effective dates, or a reliable
way to distinguish current policy from student experience. Consequently every
migrated record begins in `needs_review` except known obsolete records, which
are `superseded`.

To activate a record, fill `source`, `source_url`, `last_verified`, and set
`status` to `active`. SASTRA policy claims need official HTTPS SASTRA URLs.
UniFy product-capability claims can use a checked-in `local://` source that
names the product owner and verification date. Give annual calendar/opening-day
claims a `valid_until` date so they are labelled stale rather than silently
presented as current.
