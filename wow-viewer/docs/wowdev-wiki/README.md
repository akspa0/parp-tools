# wowdev.wiki Drafts

These files are draft upgrade notes for eventual contributions back to wowdev.wiki.

The goal is not to fork the spec into this repo. The goal is to keep small, copyable deltas that:

- start from the current public wowdev pages
- add only repo-backed findings or clearly labeled runtime observations
- preserve the wiki's existing structure and terminology where possible

Naming rule for these drafts:

- if a later wowdev page already has a stable name for the same data family, reuse that existing wiki term
- do not propose global renames just because the current tooling uses a convenient local alias
- when semantics are still open, keep the raw wiki field names and mention tooling aliases only as supporting notes

Confidence labels used in these drafts:

- Verified: backed by current shared parser behavior, focused validation, or native-client/Ghidra evidence already recorded in this repo
- Working interpretation: used by the shared stack today and supported by corpus evidence, but not yet proven as the final native semantic name
- Open: useful observation worth carrying forward, but still needs deeper native or corpus proof

Current drafts:

- Alpha page refresh: `alpha-draft.md`
- ADT/v18 cross-page follow-ups: `adt-v18-followups.md`
- PM4/PD4 page refresh: `pm4-pd4-draft.md`
