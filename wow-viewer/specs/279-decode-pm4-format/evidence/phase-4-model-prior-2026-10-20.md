# Phase 4 — Reference consolidation and opt-in model prior (2026-10-20)

## Question

Two operator-directed deliverables:

1. Rewrite `docs/research/pm4-format-reference.md` top to bottom as a single, concise document of
   truth — no appended findings, no refuted claims left standing, no stale sections.
2. Take the next bounded consumer step: feed the measured object-group → model join into the
   existing matcher as an **opt-in prior**, without changing the default scoring path.

## Method

### Reference rewrite

- Read the full 717-line document and every receipt it cited.
- Rebuilt the document around measured relationships first (§3), then the binary envelope and record
  layouts (§4–§5), coordinates (§6), consumer implications (§7), a single unknown register (§8), and
  parser limits (§9).
- Removed all superseded/refuted narrative (single-tile MPRR grid reading, early +0x04 edge-ID
  hypothesis, appended per-session notes). Refuted results are retained only where they are load-bearing
  negatives (§3.6 merged M2 mesh).
- Every claim carries an evidence letter: L (layout/code), M (measured), H (hypothesis), O (operator),
  R (refuted), U (unknown).

### Opt-in model prior

- `Pm4AssetMatchScorer.ScoreSegments` / `ScoreSegment` gained an optional
  `IReadOnlyDictionary<string, double>? modelPriors = null`.
- `ResolvePrior` matches an asset by exact path, then by filename, and clamps to 0–1.
- `EvaluateTypedCandidate` adds `priorBonus * 0.10` to the overall score, clamped at 1.0, and records
  `priorBonus` in the score breakdown plus a rationale line.
- With no prior map the bonus is exactly 0, so the default path is byte-for-byte unchanged.

## Result

- Reference is now ~475 lines, one document, one unknown register (U01–U17).
- Prior is opt-in and additive; it can only promote a candidate that the geometry scorer already
  considered, and only by at most 0.10.
- New test `ScoreSegment_ModelPriorBreaksTieOnlyWhenSupplied` proves:
  - identical imperfect geometry without a prior → `Ambiguous`, `priorBonus == 0`;
  - the same geometry with a prior on the second asset → `Matched`, second asset first,
    `priorBonus == 1`, and a strictly higher overall score than the baseline top candidate.

## Files changed

- `wow-viewer/docs/research/pm4-format-reference.md` — full rewrite.
- `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4AssetMatchScorer.cs` — opt-in prior.
- `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4AssetMatchScorerTests.cs` — prior test.

## Verification

```powershell
dotnet test wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug --filter "FullyQualifiedName~Pm4AssetMatchScorerTests"
```

Exit 0 — `Passed! - Failed: 0, Passed: 5, Skipped: 0, Total: 5`.

Full PM4 suite (same build, before the test-only adjustment):
`Failed: 1, Passed: 164, Skipped: 0, Total: 165` — the single failure was the first draft of the new
test, which asserted separation on a perfect (1.0) base score where the clamp absorbs the bonus. The
test was corrected to use imperfect geometry; the scorer was not changed.

## Criterion → evidence

| Criterion | Evidence |
| --- | --- |
| Reference is a single consolidated document | `docs/research/pm4-format-reference.md` §1–§10, one unknown register |
| No refuted claim remains as a live claim | §3.6 keeps the M2 negative explicitly labelled `M (negative)`; §8 lists only open items |
| Prior is opt-in | `modelPriors` defaults to `null`; `priorBonus` is 0 and the score is untouched |
| Prior changes ranking only when supplied | `ScoreSegment_ModelPriorBreaksTieOnlyWhenSupplied` |
| Default path unchanged | 164 pre-existing PM4 tests pass with the change in place |

## Limits

- The prior is a ranking nudge, not a proof of identity. It cannot rescue a candidate the geometry
  scorer rejected, and it cannot exceed +0.10.
- The prior map is not yet produced by a production caller; the object-group catalog and the ranked
  model list are the intended sources.
- No runtime/visual proof is claimed. The viewer Object Groups tab is unchanged by this phase.
