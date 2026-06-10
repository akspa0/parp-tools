# Tasks: 052 PM4 Signature Matcher

## Phase 1: Signature Data Type (P1)

- [ ] T001 Add `Pm4ObjectSignature` struct in `WowViewer.Core` with all fields from FR-001
- [ ] T002 Add `Pm4SignatureType` enum (`DisjointDecoration`, `SimpleMesh`, `ConnectedWmo`, `ContiguousM2`)
- [ ] T003 Add unit tests for signature classification: ratio < 0.1 → disjoint, 0.1-0.3 → simple, 0.3-1.0 → connected, > 1.0 → contiguous

## Phase 2: Per-Object Signature API (P1)

- [ ] T004 Add `TryGetPm4ObjectSignature(key, out signature)` to WorldScene
- [ ] T005 Implement signature computation: count MSCN/MSPV points in object's bounds, compute volume, density, aspect ratios
- [ ] T006 Cache signature per object part; invalidate on overlay reload
- [ ] T007 Add `Pm4SignatureClassifier.Classify(signature)` static helper

## Phase 3: Signature Index Builder Tool (P1)

- [ ] T008 Add `pm4 build-index` command to `WowViewer.Tool.Inspect` (per FR-003)
- [ ] T009 Walk `world/wmo/**/*.wmo` and read MOHD/MOGI to estimate surface/vertex counts and bounds
- [ ] T010 Walk `world/m2/**/*.m2` and read equivalent M2 header data
- [ ] T011 Write index to `output/tmp/wmo_signature_index.json` with WMO + M2 entries
- [ ] T012 Add progress reporting for the index build
- [ ] T013 Test on the staged client at `I:\parp\parp-tools\output\tmp\wowarchive-clients\` — build completes in <5 min

## Phase 4: Signature Matcher Service (P1)

- [ ] T014 Add `Pm4SignatureMatcher` static class in viewer project
- [ ] T015 Implement `FindMatches(signature, index, topN)` with the 5-factor scoring formula from US4
- [ ] T016 Add `Pm4MatchCandidate` record (path, kind, score, score breakdown)
- [ ] T017 Add unit tests with synthetic signatures verifying the formula (1.0 for identical, monotonic for similar, low for different)

## Phase 5: UI Integration (P1)

- [ ] T018 Update `DrawPm4SelectionWorkbenchContent` to show the selected object's signature in the match panel header
- [ ] T019 Wire "Find Match" button to call ADT placement first, then signature search, then merge results
- [ ] T020 Add "Build Index" button in the PM4 workbench to invoke the index builder
- [ ] T021 Display top 5 candidates in a table with model path, kind (WMO/M2), score, score breakdown
- [ ] T022 Add "Confirm Match" per candidate — saves to `pm4_wmo_matches.json` via `Pm4WmoMatchStore`
- [ ] T023 Show "✓ saved" badge when the current object has a saved match
- [ ] T024 Show "From ADT" / "From Signature" labels on each candidate row

## Phase 6: Index Loading & Caching (P2)

- [ ] T025 Load index from `output/tmp/wmo_signature_index.json` on viewer startup
- [ ] T026 Add staleness check: if client root mtime > index mtime, suggest "Rebuild"
- [ ] T027 Add "Rebuild Index" button that re-runs the index builder
- [ ] T028 Cache loaded index in memory for fast repeated matcher calls

## Phase 7: Ground Truth Validation (P2)

- [ ] T029 Build ground truth set: 20+ PM4 objects with known WMO/M2 placements from a test tile
- [ ] T030 Run matcher against the ground truth, compute top-1 accuracy
- [ ] T031 If accuracy < 70%, tune the scoring weights and re-run
- [ ] T032 If accuracy >= 70%, declare spec "done" and add validation log to spec

## Phase 8: M2 Support (P2)

- [ ] T033 Add M2 path to signature index (already in FR-003 but verify the M2 reader works)
- [ ] T034 Display M2 candidates in the match panel with kind label
- [ ] T035 Confirm M2 matches save with the model kind

## Summary

**8 user stories, 7 functional requirements.**

**Core deliverable**: A user clicks "Find Match" on a selected PM4 object → sees top 5 WMO/M2 candidates ranked by signature similarity, with confidence levels and per-factor score breakdowns → confirms the right match → it's saved for future sessions.

**Key insight**: The PM4 signature (MSCN/MSPV counts + bounds + ratio) is unique per WMO/M2 model. A multi-factor similarity score across an index of all WMO/M2 signatures in a staged client gives a fast, reliable matcher that works even when ADT placement isn't available.

**Deferred** (out of scope):
- Learned signature embeddings (v1 is hand-crafted formula)
- M2 chunk-level extraction (too slow for indexing)
- Cross-client matching (each client needs its own index)
- Automated correction
- Per-instance WMO scaling
