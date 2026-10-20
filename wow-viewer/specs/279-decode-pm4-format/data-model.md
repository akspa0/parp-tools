# Claim and Evidence Model — Spec 279

This is a documentation/research contract; it does not replace raw PM4 entry models.

## FormatElement

- Stable chunk signature + optional record offset/type/size.
- Current reader alias, explicitly not assumed to be semantic truth.
- Raw byte preservation and typed projection behavior.
- Linked current implementation and relevant measured claims.

## Claim

- Claim ID, statement, evidence category and scope (corpus/build/subset).
- Categories: **L** layout/code fact, **M** real-data measured relationship,
  **H** working hypothesis, **O** operator evidence, **R** refuted/limited interpretation,
  **U** unresolved meaning.
- Source report/command, input identity, denominator, controls and limitations.
- Status changes are recorded, not silently rewritten as historical certainty.

## Unknown

- Unknown ID and affected format elements.
- Competing interpretations and one next discriminating experiment.
- Required independent data/control and current blocker/proof owner.
- Last outcome: not tested / supported / refuted / inconclusive, with receipt link.

## ExperimentReceipt

- Question/hypotheses and instrument version/source.
- Files changed; exact commands and exit status.
- Same-byte source hashes, map inventory and diagnostics.
- Per-hypothesis real outputs, positive/wrong-target controls and missing-target accounting.
- Acceptance-criterion → evidence table and explicit proof gaps.

## ConsumerImplication

- Verified claim to consume, intended read-only interpretation or proposed behavior change.
- Owning service/spec task, baseline, regression checks and operator/runtime proof gate.
- Never upgrades an expectation, a navigation component or a sparse shape into asset identity.
