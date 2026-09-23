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

## Phase 3 — GPU Submission

- [x] T006 Add a renderer-scoped GPU batch compatibility contract covering geometry/material/texture/
      state compatibility in `wow-viewer/src/viewer/WoWViewer/Rendering/IGpuInstancedModelRenderer.cs`.
- [x] T007 Upload per-placement model matrices/fade values and issue instanced opaque geoset draws
      in `wow-viewer/src/viewer/WoWViewer/Rendering/ModelRenderer.cs`; delegate legacy-backed M2
      support through `wow-viewer/src/viewer/WoWViewer/Rendering/M2Renderer.cs` and WMO placement
      batches through `wow-viewer/src/viewer/WoWViewer/Rendering/WmoRenderer.cs`.
- [ ] T008 Prove visual parity and measure CPU, upload, driver-wait, and GPU stages on synthetic and
      user-run real-client scenes before promoting GPU instancing or multi-draw submission.

## Phase 4 — Client I/O Contention Containment

- [x] T009 Move deferred WMO doodad model loading behind one scene-wide
      `WorldAssetManager` budget; remove placement-owned loading from `WmoRenderer.RenderWithTransform`.
- [x] T010 Reduce minimap client-data readers to one background worker while retaining bounded
      render-thread texture uploads.
- [ ] T011 Run the user-owned real-client comparison and record whether deferred asset CPU time,
      pending doodad depth, minimap progress, and frame rate recover; do not promote an FPS claim
      from the compile/build proof alone.
