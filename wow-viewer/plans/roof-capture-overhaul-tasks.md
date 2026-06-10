# Tasks: Roof Capture Overhaul

**Input**: [`wow-viewer/plans/roof-capture-overhaul-plan-2026-05-30.md`](wow-viewer/plans/roof-capture-overhaul-plan-2026-05-30.md)

**Branch**: `v0.5.0-clean` (hard-reverted to `630dee49`)

**Target builds**: `0_5_3_3368`, `3_0_1_8303`, `4_0_0_11927`

---

## Phase 1: Fix Empty Renders (Critical Blockers)

**Purpose**: Restore functional GPU rendering for all builds. Without this, nothing else works.

- [ ] T001 Fix FBO state restoration between renders in [`gillijimproject_refactor/src/MdxViewer/Catalog/ScreenshotRenderer.cs`](gillijimproject_refactor/src/MdxViewer/Catalog/ScreenshotRenderer.cs:26) — reset viewport, clear color+depth buffer, bind FBO at the start of every `RenderWmoRoofTopDown` and `RenderMdxRoofTopDown` call
- [ ] T002 [P] Add MDX/M2 file extension probing in [`gillijimproject_refactor/src/MdxViewer/Catalog/ScreenshotRenderer.cs`](gillijimproject_refactor/src/MdxViewer/Catalog/ScreenshotRenderer.cs:153) — when reading model bytes, if `.mdx` returns null try `.m2` and vice versa, so 3.x+ M2 files resolve correctly
- [ ] T003 [P] Wrap MdxRenderer.UpdateAnimation() in try/catch in roof capture paths at [`gillijimproject_refactor/src/MdxViewer/Catalog/ScreenshotRenderer.cs`](gillijimproject_refactor/src/MdxViewer/Catalog/ScreenshotRenderer.cs:162) — prevent corrupted animation tracks from crashing the render; static bind-pose fallback

**Checkpoint**: Verify by capturing 5 WMO assets from `duskwood` on `3_0_1_8303` — non-zero pixel output in `roof_topdown.png`

---

## Phase 2: Camera & Geometry Quality

**Purpose**: Ensure captures frame the model correctly and don't waste resources on invisible geometry.

- [ ] T004 Fix adaptive orthographic camera in [`gillijimproject_refactor/src/MdxViewer/Catalog/ScreenshotRenderer.cs`](gillijimproject_refactor/src/MdxViewer/Catalog/ScreenshotRenderer.cs:200) — use `max(spanX, spanY) * 1.15` orthographic size instead of perspective projection for roof capture; prevent clipping
- [ ] T005 Add underground WMO group culling in [`gillijimproject_refactor/src/MdxViewer/Catalog/ScreenshotRenderer.cs`](gillijimproject_refactor/src/MdxViewer/Catalog/ScreenshotRenderer.cs:460) — skip rendering WMO groups whose `BoundsMax.Z < wmo.BoundsMin.Z + 0.5f` (basement/underground geometry)
- [ ] T006 [P] Also fix the perspective multi-angle camera in [`gillijimproject_refactor/src/MdxViewer/Catalog/ScreenshotRenderer.cs`](gillijimproject_refactor/src/MdxViewer/Catalog/ScreenshotRenderer.cs:374) — add `Math.Max(aspectRatio, 1/aspectRatio)` to the max-distance formula so tall models aren't clipped in side-angle views
- [ ] T007 Fix M2 foliage-through-trunk rendering in roof capture in [`gillijimproject_refactor/src/MdxViewer/Catalog/ScreenshotRenderer.cs`](gillijimproject_refactor/src/MdxViewer/Catalog/ScreenshotRenderer.cs:396) — for the top-down orthographic roof pass only, render ALL M2 geometry as if it were opaque (disable transparent pass for roof capture) so tree leaves fully cover trunks underneath

**Checkpoint**: Capture a tall WMO (e.g. a tower) and verify it fills the frame without clipping.

---

## Phase 3: Direct-to-Zarr Output with Resume

**Purpose**: Eliminate intermediate PNG files on disk. Pack roof captures directly into Zarr stores with resume support.

- [ ] T007 Write Zarr append/resume helper in a new file [`gillijimproject_refactor/src/MdxViewer/Catalog/RoofCaptureZarrWriter.cs`](gillijimproject_refactor/src/MdxViewer/Catalog/ScreenshotRenderer.cs:1) — opens a Zarr store, writes `roof_rgb` (128×128×3 uint8 array), `roof_mask` (128×128 float32), `roof_confidence` (float32), `build_code` (int32), and a resume-state JSON tracking completed asset paths
- [ ] T008 Modify the roof capture pipeline in [`gillijimproject_refactor/src/MdxViewer/Catalog/ScreenshotRenderer.cs`](gillijimproject_refactor/src/MdxViewer/Catalog/ScreenshotRenderer.cs:627) to output directly to the Zarr writer instead of writing PNG/JPG files to disk; add `--capture-roof-zarr <path>` CLI option in [`gillijimproject_refactor/src/MdxViewer/ViewerApp_StartupAutomation.cs`](gillijimproject_refactor/src/MdxViewer/ViewerApp_StartupAutomation.cs:234)
- [ ] T009 Add resume logic in the roof capture batch loop in [`gillijimproject_refactor/src/MdxViewer/ViewerApp_CaptureAutomation.cs`](gillijimproject_refactor/src/MdxViewer/ViewerApp_CaptureAutomation.cs:806) — on startup, read existing Zarr resume-state; skip assets already captured; track per-asset success/failure with reason
- [ ] T010 [P] Write metadata parquet alongside Zarr in [`gillijimproject_refactor/src/MdxViewer/Catalog/RoofCaptureZarrWriter.cs`](gillijimproject_refactor/src/MdxViewer/Catalog/RoofCaptureZarrWriter.cs:1) — `asset_path`, `build`, `success`, `failure_reason`, `render_time_ms`

**Checkpoint**: Run pipeline on `0_5_3_3368` with 50 assets; kill mid-way; resume — verify it skips completed and only captures remaining.

---

## Phase 4: Validation & Full Harvest

**Purpose**: Run the full capture for all 3 target builds and validate outputs.

- [ ] T011 Run full roof capture on `0_5_3_3368` using `--capture-roof-zarr` with `--capture-roof-resolution 512`
- [ ] T012 Run full roof capture on `3_0_1_8303`
- [ ] T013 Run full roof capture on `4_0_0_11927`
- [ ] T014 Write validation sampler script in [`wow-viewer/data-harvester/scripts/validate_v18_object_roof_captures.py`](wow-viewer/data-harvester/scripts/validate_v18_signals.py:1) — reads each build's Zarr store, dumps sample `roof_rgb`/`roof_mask` overlays as PNG, and reports non-zero coverage stats per tile
- [ ] T015 Update [`wow-viewer/memory-bank/activeContext.md`](wow-viewer/memory-bank/activeContext.md:6) and [`wow-viewer/memory-bank/progress.md`](wow-viewer/memory-bank/progress.md:3) with new pipeline architecture and proof

---

## Dependencies

- **Phase 1** (Fixes 1-4): No dependencies. Can start immediately.
- **Phase 2** (Camera/Quality): Depends on Phase 1 (need non-empty renders first).
- **Phase 3** (Zarr output): Depends on Phase 1 (need functioning render pipeline first).
- **Phase 4** (Full harvest): Depends on Phases 1-3.

Parallel opportunities: T002 (MDX/M2 probing) and T003 (animation try/catch) can run in parallel since they touch different code paths in ScreenshotRenderer.cs.

## Implementation Strategy

### Must-Have (Phase 1 + Phase 2)

Stop after Phase 2 if we have functional 512×512 roof captures as PNG. The Zarr output (Phase 3) is a nice-to-have for disk efficiency but not required for validation.

### Recommended Incremental Delivery

1. Phase 1 fixes → test with 5 WMOs on 3_0_1_8303
2. Phase 2 quality fixes → get one good full-frame tower capture
3. If time permits, Phase 3 Zarr output with resume
4. Phase 4 validation + full harvest

### Suggested MVP Scope

**Phase 1 only**: FBO fix + MDX/M2 probing + animation try/catch. This alone should restore functional roof captures for all 3 builds.
