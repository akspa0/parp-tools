# Tasks: M2 and WMO Doodad Submission Batching

## Phase 1 — Static M2 Batch Eligibility

- [x] T001 Make `M2Renderer` delegate legacy particle/ribbon fallback eligibility while preserving
      isolation for the native-runtime backend.
- [x] T002 Preserve the existing shared world `BeginBatch()` / `RenderInstance()` contract for
      static legacy-backed M2 instances.

## Phase 2 — Opaque WMO Doodad Grouping

- [x] T003 Group visible opaque WMO doodads by `IModelRenderer` and issue one batch setup per group.
- [x] T004 Preserve transparent distance order and particle/ribbon unbatched fallback behavior.
- [x] T005 Update the Spec 136 acceptance contract and memory-bank handoff with the proof boundary.

## Phase 3 — GPU Submission (Not Started)

- [ ] T006 Add a backend-specific batch key covering geometry/material/texture/state compatibility.
- [ ] T007 Prove visual parity and measure CPU, upload, driver-wait, and GPU stages on synthetic and
      user-run real-client scenes before promoting GPU instancing or multi-draw submission.
