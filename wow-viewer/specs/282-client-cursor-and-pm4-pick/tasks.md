# Tasks: Real Client Cursor Model and Precise PM4 Picking

**Spec**: [`spec.md`](spec.md) · **Created**: 2026-10-21 · **Receipt**: [`evidence/receipt-2026-10-21.md`](evidence/receipt-2026-10-21.md)

## Phase 1 — PM4 triangle-level picking (FR-005, FR-006)

- [x] **T001** Replace the AABB-only PM4 pick with a two-stage pick: AABB as a cheap reject, then
      the nearest actual triangle under the ray. Receipt: `Pm4OverlayScene.Selection.cs`
      `TryPickPm4ObjectByRay`; `Pm4OverlayScene.cs` `RayTrianglesIntersect` / `RayTriangleIntersect`
      (Möller–Trumbore, epsilon 1e-6).
- [x] **T002** Keep the existing distance policy (`IsHoverPickDistanceAllowed`) so the operator's
      hover-range setting still applies. Receipt: same method, unchanged call site.
- [x] **T003** Verify the PM4 test suite still passes. Receipt: `WowViewer.Core.PM4.Tests`
      165/165 passed.

## Phase 2 — Client cursor model resolution (FR-001, FR-008)

- [x] **T004** Dispatch cursor candidates by extension instead of calling `MdxFile.Load` on every
      candidate. Receipt: `SceneCursorRenderer.EnsureCursorModelLoaded` →
      `TryLoadMdxCursor` / `TryLoadM2Cursor`.
- [x] **T005** Resolve M2 cursor skins (`Cursor.skin`, `Cursor00.skin`, `Cursor01.skin`,
      `Cursor02.skin`, plus lower-case variants) and build the runtime model through
      `WowViewerM2RuntimeBridge.BuildStaticRenderModel`; fall back to
      `BuildEra100StaticRenderModel` for Alpha-era inline-geometry M2. Receipt:
      `SceneCursorRenderer.TryLoadM2Cursor` + `EnumerateSkinCandidates`.
- [x] **T006** Report a missing or unreadable cursor model instead of silently falling back.
      Receipt: `SceneCursorRenderer.LastLoadError`, `ModelKind`, `LoadedCursorModelPath`,
      `HasAuthenticCursorModel`; surfaced in Options > Objects.
- [x] **T007** Re-point the cursor renderer at the active data source when a client is opened.
      Receipt: `ViewerApp.OnUpdate` calls `_sceneCursorRenderer?.SetDataSource(_dataSource, _texResolver)`;
      `SetDataSource` drops the cached model on a real change.

## Phase 3 — Render the model as the pointer (FR-002, FR-003, FR-004)

- [x] **T008** Render the resolved model at the world point under the pointer, oriented to the
      camera, with distance-proportional scaling. Receipt: `SceneCursorRenderer.Render` +
      `RenderCursorModel` (M2 and MDX paths).
- [x] **T009** Normalize only models authored far outside the ~1-unit cursor range, so an authentic
      model keeps its authored proportions. Receipt: `ResolveModelScaleNormalization`.
- [x] **T010** Load the model before touching GL state, so `M2Renderer` shader/buffer setup cannot
      clobber the cursor's depth range and blend state. Receipt: `Render` calls
      `EnsureCursorModelLoaded()` before `DepthRange`/`BlendFunc`.
- [x] **T011** Hide the hardware pointer only while the model cursor is drawn over the viewport.
      Receipt: `ViewerApp.OnRender` `hideHardwareCursor` gate (pre-existing, unchanged).
- [x] **T012** Keep the cursor out of scene picking. Receipt: the cursor is drawn in the overlay pass
      after ImGui and is not registered with `WorldScene`/`SceneHoverPickController`.

## Phase 4 — Options and documentation (FR-007)

- [x] **T013** Show the resolved cursor model (or the load error) in Options > Objects and add a
      `Reload cursor model` button. Receipt: `ViewerChromeService.Options.cs`
      `DrawOptionsObjectsContent`.
- [x] **T014** Register the new control in the UI control registry and regenerate the reference.
      Receipt: `eng/ui/ui-controls.json` entry `reload-cursor-model`;
      `docs/WoWViewer/UI-REFERENCE.md` regenerated; `Verify-UiReference.ps1` PASSED
      (`inventory_rows=128 distinct_labels=119 registry_entries=119`).

## Phase 5 — Verification

- [x] **T015** Solution build. Receipt: `dotnet build WowViewer.slnx -c Debug` → 0 errors.
- [x] **T016** UI documentation tests. Receipt: `UiControlDocumentationTests` 3/3 passed.
- [ ] **T017** Operator visual proof: cursor model follows the pointer over terrain, WMO and M2
      surfaces; 2D arrow hidden only over the viewport. **Operator-owned** — not claimed from a
      build.
- [ ] **T018** Operator functional proof: PM4 click selects the surface under the pointer in a spot
      check (SC-002). **Operator-owned** — not claimed from a build.

## Known gaps

- The M2 cursor path is exercised only by compilation. No real client cursor asset has been loaded
  in a run yet, so the skin-name candidates and the authored scale are unproven against real data.
- `M2Renderer` animation is not driven for the cursor; the model holds its first frame. The spec
  allows this ("play the idle sequence, or hold the first frame if none exists"), but the idle
  sequence is not played.
