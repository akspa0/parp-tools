# Research Decisions and Open Questions — Spec 279

## Adopted measured baseline

The [Spec 270 structural receipt](../270-pm4-object-association-ledger-and-placement/evidence/structural-foundations-2026-10-20.md)
records the 616-file corpus fingerprint, exact surface/link windows, addressed-tile references,
geometry contacts and paired local-target controls. Reuse that evidence with its denominator and
source-build limitations. The [current reference](../../docs/research/pm4-format-reference.md)
is the reader-facing synthesis; this file contains planning decisions, not a second schema.

## Decisions

| Decision | Reason | Rejected alternative |
|---|---|---|
| Layout confidence and semantic confidence are separate | Parsing a word does not establish its meaning | Treating reader aliases as format names |
| Whole-map addressed context is mandatory | 15,773 measured links address another tile | Valid-looking source-file indices used as truth |
| Existing reports are adopted, not re-created as new proof | Preserve provenance and avoid duplicated “discoveries” | Repeating old corpus fractions without receipts |
| Begin with the current writeup | Operator explicitly requested it; prevents context drift | More probes with no coherent reference |
| Next test is raw link +0x04 | Legacy identity claim is unproved and affects M2 partition assumptions | Assuming all GroupObjectId values are asset instances |
| In-range tests are screening only | Small integers fit many unrelated arrays | Calling the best fitting domain decoded |
| Reference is corpus-scoped | Exact source build remains unidentified | Universal claims for all PM4/PD4 versions |
| Authored editor nodes remain a separate hypothesis | Operator screenshot evidence has not identified a stream | Renaming MPRL/MSCN to waypoints by appearance |
| Consumer integration follows semantic proof | Preserve useful WMO baseline and shared readers | Rewrite the viewer to make synthetic tooling agree |

## Phase 1 hypotheses (raw MSLK +0x04)

- **H1 reverse-link index:** in the addressed target tile, the indexed connection record points
  back to the source surface/tile; it should beat index+1/+17 and wrong-file controls.
- **H2 other index domain:** values fit a different stream, but this alone is not a semantic result;
  an independent structural/geometric relationship must be found before acceptance.
- **H3 recurring/group-like value:** values may repeat within or across surfaces; any ownership
  interpretation must later explain multiple copies/scales and independent authored identity.

The probe must be capable of detecting a constructed true reverse-link index and rejecting a
constructed in-range false value. A result rejecting H1 does not establish H2 or H3 automatically.

### Phase 1 observed outcome — 2026-10-20

[Whole-map report](evidence/raw-04-domain-2026-10-20.json), same 616-file fingerprint as the adopted
baseline: 1,273,335 link values; 32,484 distinct raw values; 539 zero values; no all-ones values.
All source-file values fit MSCN, MSLK **and** MSVI: range fit does not choose a domain.
H1 as a general reverse-link pointer is rejected: only 1,603 correct-window/back-reference hits
among 1,269,146 in-range addressed candidates, versus +1/+17 controls 1,092/833. Some partial
relationships may remain, but no universal meaning is assigned. H2/H3 remain unresolved.
Next concern: independent geometric/reference tests for MSCN/point-domain alternatives against
the addressed graph, preserving authored editor-node identity as a separate question.

## Sub-step 2A outcome — 2026-10-20

[Receipt](evidence/phase-2a-edge-ordinal-2026-10-20.md): direct +0x04 point-index alternatives show
no useful boundary relation under the tested indices/frames. Independent raw +0x01 test instead
identifies a zero-based **source-edge ordinal**: valid for all 1,273,335 connections, 674,013
shared boundaries vs shifted control 137. Auxiliary geometry lies in the selected edge's vertical
plane in 536,142 cases vs 3,120 controls; this is not proof of equal height/traversal semantics.
Source-object ownership and authored route-node layer remain unknown. No production alias was renamed.

## Wider unresolved areas

Headers/region metadata; surface class/plane/raw-word meanings; raw link subtype/+0x02/+0x04/high
address-half/+0x12 meanings; path-bearing-kind interpretation; MSCN point roles; MPRL field and
coordinate semantics; MPRR structure; auxiliary filename/state relationships; source-object
ownership; absence of neighboring files; build variation; editor node correspondence.

Every item is expanded with its next discriminating test in the reference's unknown register.
