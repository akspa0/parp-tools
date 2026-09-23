# Spec reconciliation audit brief (2026-09-23)

Operator-directed pass: audit every non-archived Spec Kit spec under `wow-viewer/specs/` against the
**actual code**, so the open plans can be reconciled into a handful of new epic specs and the old
specs archived (moved, never deleted).

Each batch report in this folder was produced by a read-only audit sub-agent against this brief.

## Rules for auditors

- **Read-only.** Do not edit, move or create anything except your own batch report file.
- Checkboxes in `tasks.md` are **not evidence**. Many specs are known to be wrong in both directions
  (e.g. 238 CASC shows 0/41 checked yet CASC reads shipped in v0.6.0-alpha2). Verify claims against
  source: grep/glob for the named classes, CLI verbs, tests, and Python modules.
- Code roots: `wow-viewer/src/core/*`, `wow-viewer/src/viewer/WoWViewer/`, `wow-viewer/tools/*`,
  `wow-viewer/tests/*`, `wow-viewer/data-harvester/` (Python). Receipts live in each spec's `evidence/`.
- Code existing is not runtime proof. Separate "code present + tests" from "operator/runtime gate owed".
- Do **not** invent scope. "Open residue" must be scope the spec itself states (quote/paraphrase and
  cite the FR/US/task id).
- Keep reports tight: roughly 15-40 lines per spec.

## Per-spec report format

```
### <id> <title>
- Stated status: <what spec/tasks say>   | Tasks: <checked>/<total> (or "no tasks.md")
- Scope: <1-2 lines>
- Verified implemented: <bullets: capability -> file path / symbol / test / CLI verb>
- Partial: <bullets, what exists vs what is missing>
- Not implemented: <bullets with FR/US/T ids>
- Checkbox accuracy: <accurate | N checked-but-absent (ids) | N unchecked-but-present (ids)>
- Operator gates owed: <runtime/visual/client proofs still unwitnessed, or none>
- Open residue (spec-stated only): <bullets with ids — the ONLY things that should carry forward>
- Superseded by / overlaps: <spec ids>
- Disposition: ARCHIVE-COMPLETE | ARCHIVE-SUPERSEDED | ARCHIVE-COLD (research/abandoned, no live residue) | FOLD (residue carried into an epic) | KEEP-ACTIVE (live owner, too big/active to fold)
- Proposed epic theme: <one short theme name>
- Confidence: high | medium | low (+ why if not high)
```

End the batch report with a **Batch summary** table: `id | disposition | residue count | theme`.
