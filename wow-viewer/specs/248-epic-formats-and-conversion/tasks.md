# Tasks — Epic 248 Formats, Readers, Writers & Conversion

Implementation tasks are generated **after** triage (`speckit-tasks` against the items marked
*Want*). Nothing here is checked without a §9.2 receipt.

## Phase 0 — Triage (operator)

- [ ] T001 Operator marks every F-xx item Want / Drop / Later in [TRIAGE.md](../TRIAGE.md).
- [ ] T002 Record decisions in this epic's spec.md (dated amendment); move Drop items to a
      "Dropped by operator" table with the date.
- [ ] T003 Generate phased implementation tasks for the Want items only; pick the approach per plan.md.

## Operator verification sweep (shipped code — proof only, no implementation)

- [ ] V001 Legacy M2/MDX render across 1.0.0–3.0.1 (235 SC-002/003/008) and alpha2 M2 texture-wrap fix.
- [ ] V002 Real-client liquid confirmation (205 T206/T304/T501/T502).
- [ ] V003 Load a DAT→LK export in a 3.3.5 client or Noggit; decide `--transpose` (237/247).
- [ ] V004 Compose a DAT Cartography layer on screen (247 US5).
- [ ] V005 CASC hash-match against an independent extractor (238 SC-001/002).
- [ ] V006 `LkAdtWriter` alpha2 chunk-completeness output loads in a client.
