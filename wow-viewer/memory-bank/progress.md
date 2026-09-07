# Progress — wow-viewer

Last updated: 2026-09-06

## 2026-09-07 — Wireframe/selection fixes, export freeze fix, Spec 231 UI overhaul planned

- Wireframe root cause fixed across terrain and models: all wireframe passes drew textured
  lines identical to the fill beneath them (invisible on terrain; alpha-cutout orange
  fragments + silhouette-only lines on objects). Flat-color passes now in
  [TerrainRenderer.cs](../src/viewer/WoWViewer/Terrain/TerrainRenderer.cs) (semi-transparent
  white), [M2Renderer.cs](../src/viewer/WoWViewer/Rendering/M2Renderer.cs),
  [ModelRenderer.cs](../src/viewer/WoWViewer/Rendering/ModelRenderer.cs) via a color-override
  parameter on [IModelRenderer](../src/viewer/WoWViewer/Rendering/IModelRenderer.cs).
- Selection highlight is now the model's red wireframe instead of a bounding box
  ([WorldScene.cs](../src/viewer/Terrain/WorldScene.cs) selection block; box fallback only
  when the model is not streamed; WMO-doodad placement markers keep pins/axes).
- Editor toolbar: `Anim` toggle (world doodad animations default ON now) + hovered-WMO
  doodad-set combo; left sidebar layer/overlay wall collapsed by default; Imports & Exports
  groups collapsed by default (first de-congestion pass; structural fix is Spec 231).
- PM4 OBJ export freeze fixed: moved off the render thread with status + re-entrancy guard
  ([ExportPm4ObjectsObjSet](../src/viewer/WoWViewer/ViewerApp_Pm4Utilities.cs)).
- **Spec 231 authored per operator directive** (speckit; implementation deferred to a fresh
  session): [231-editor-archaeology-ui-overhaul/](../specs/231-editor-archaeology-ui-overhaul/spec.md)
  with plan (4-page Editor IA, Archaeology de-hosting, dedupe D1–D6, Spec 228 page-class
  pattern, phases P0–P5) and gated tasks. Registered in STATUS.md + UI epic.
- Receipts: `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj` 0 errors at each
  step. Visual/interactive acceptance operator-owned. See also the 2026-09-06 v0.5.3-rc1
  entry below for the version-plumbing + changelog work committed together.

## 2026-09-06 — Version plumbing fix + v0.5.3-rc1 bump

- Fixed the About-box/version drift: `eng/Version.props` was never imported (bare relative
  `Exists()` in the repo-level `Directory.Build.props` resolves against the *project* directory),
  and a shadowing `Directory.Build.props` in `src/viewer/WoWViewer/` blocked the repo-level file.
  Both now chain correctly; the viewer csproj's hardcoded `0.5.2.2` block was removed.
- Viewer title/About now derive from the assembly `InformationalVersion` (SDK `+<git-commit>`
  metadata trimmed for display) instead of a hardcoded const
  ([ViewerApp.cs](../src/viewer/WoWViewer/ViewerApp.cs)).
- Version bumped to `0.5.3-rc1` (assembly `0.5.3.0`) per operator direction; release notes
  rewritten as a real changelog of the ~160 commits since `v0.5.2.1` (5.0.1 era support, PM4
  semantics campaign, Rosetta, editor platform, converter fixes) at
  [docs/releases/v0.5.3-rc1.md](../docs/releases/v0.5.3-rc1.md). Receipt: `dotnet build` of both
  `WoWViewer.csproj` and `WoWViewer.CrossPlatform.csproj` 0 errors; both targets'
  `ParpToolsWoWViewer.dll` report `ProductVersion 0.5.3-rc1+<sha>`, `FileVersion 0.5.3.0`.
  The CrossPlatform csproj also hardcoded `0.5.2.2` (the binary the operator was running) and was
  de-hardcoded in the same pass. Tag push (`v0.5.3-rc1`) and GitHub Actions release run are
  operator-owned.

## 2026-09-06 — Context and documentation reduction pass

- Created [docs/README.md](../docs/README.md) as the canonical documentation router; legacy
  `DOCUMENTATION-STATUS.md` and `PLANS-OVERVIEW.md` are now redirects.
- Preserved, rather than deleted, high-confidence historical material under `docs/archive/`:
  the 49-file 2026 game-viewer plan pack, the consumed M2 investigation packet, the stale
  2026-08-01 spec audit, and the intact legacy MdxViewer tarball.
- Moved only clearly superseded specs 080, 145, and 195 to
  `specs/archived/superseded/` with successor pointers. They are not asserted complete.
- Added [Spec 228](../specs/228-source-decomposition/plan.md) planning artifacts. The first source
  extraction remains blocked by the Spec 227 T004 UI-authority gate.
- The full Spec 224 receipt audit is still open; this pass recorded its archive and routing work
  without checking its cleanup tasks or silently closing Gate 1.

## 2026-09-06 — Current implementation handoff

- Spec 227 T001/T002 source documentation is receipted. The next task is the operator-owned T003
  screenshot/input matrix, then T004 gate.
- Spec 223's fog/WMO/capture acceptance retest remains separately operator-owned.

Earlier same-day narrative was preserved in
[memory-bank/archive](archive/README.md), not discarded.
