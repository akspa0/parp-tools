# P1 bounded file-operation repair — 2026-10-21

**Code delivered; P1 runtime gate remains open.** Focused viewer build passes. Seven
reporting checks and source-structure checks pass. No client, export job, or real-terrain
import was run. Parent must first integrate the content APIs in the [P3 handoff](p3-top-menu.md).
No task checkbox, status registry, memory bank, host bridge, reader, renderer, or god-class
was edited. No commits. Unrelated worktree changes were preserved.

## Changed code

- [Main menu](../../../src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs): shared
  Open File picker; WDT open preserves the loader's actual result rather than unconditionally
  claiming success; PM4 and loose-folder callbacks report exceptions and selected paths.
- [Content/deferred adapters](../../../src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs):
  destination entry points and compatibility request consumption.
- [Terrain operations](../../../src/viewer/WoWViewer/Workbench/Services/FileOperations/TerrainFileOperationsService.cs):
  10 export actions, 3 session-only import actions, shared path picker, exception reporting
  **inside** callbacks, nonempty-output verification, positive applied/written count required.
- [Model/world operations](../../../src/viewer/WoWViewer/Workbench/Services/FileOperations/ModelWorldFileOperationsService.cs):
  shared save/folder picker, model GLB/WMO collision/world tile scopes; unsupported formats and
  null M2 conversion now fail rather than falling through to a success message.
- [Reporter](../../../src/viewer/WoWViewer/Workbench/Services/FileOperations/FileOperationReporter.cs),
  [verification script](../../../src/viewer/WoWViewer/Workbench/Services/FileOperations/Verify-FileOperations.ps1).
- [Old Data I/O dashboard](../../../src/viewer/WoWViewer/Workbench/Services/EditorPanels/EditorPanelsService.cs:1071):
  duplicate export/import controls removed; obsolete bridge temporarily gives destination guidance.
- [Terrain page](../../../src/viewer/WoWViewer/Workbench/Pages/EditorTerrainToolsPage.cs:14) and
  [Converters page](../../../src/viewer/WoWViewer/Workbench/Pages/EditorConvertersPage.cs:14): optional
  content delegates; parent constructor/host wiring is intentionally outside this assignment.

## Apply contract and limits

The new orchestration uses existing helpers, without changing their format interpretation:

| Import | Existing apply path | Persistence / evidence |
|---|---|---|
| Alpha atlas | DecodeAlphaShadowArrayFromAtlas → ReplaceTileAlphaShadowArray | Live renderer preview only; not an ADT edit or save. Real-client proof pending. |
| 257 heightmap + JSON | DecodeL16 → ApplyHeightmap257ToChunks → ReplaceTileChunksAndRebuild | Session chunks/rebuild only. Real-client proof pending. |
| MCCV PNG | ApplyTileImageToChunks → ReplaceTileChunksAndRebuild | Session chunks/rebuild only. Real-client proof pending. |

Import folders accept the exporter naming convention: `tile_X_Y_alpha.png`,
`tile_X_Y_height_257.png` (matching per-tile JSON or `heightmap_257_map.json`), and
`tile_X_Y_mccv.png`. This excludes unrelated PNGs/chunk masks instead of applying an
arbitrary image to a tile. Renamed single-file exports must be restored to that convention
before folder import. Scope is current/loaded/whole map, captured when opening the picker.
An unloaded matching target, missing/invalid height metadata, no matches, or callback exception
is a failure, not “Imported 0”. Map identity is checked again when a selection returns.

Map-normalized height export rejects a folder containing conflicting per-tile metadata;
it does not silently reuse stale normalization or delete old files. Exports default to the
existing managed output root; another location is selected explicitly by the operator.
Failure reports warn that partial output/application may remain; operations are not transactional.
Cancel never reports success. Whole-map operations remain operator-owned interactive jobs.

Existing DAT exporters remain unchanged and exposed in world content. Their own picker,
write/manifest/status paths are retained. The nearby-DAT action now requires a data source,
catches pre-export exceptions, and treats a zero-tile result as failure. DAT format correctness,
converter execution, PM4 export, synthesized minimap, and external host-owned dialogs are not
certified by this bounded repair.

## Verification receipts

### Focused build

```powershell
dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug --no-restore -v quiet
```

Final exit **0**, **317 warnings / 0 errors**, elapsed **00:00:06.76**. The first two
implementation builds exited 1 (image-name ambiguity, then missing adapter namespace); both
were fixed in scope. The third and final builds passed. Compiler excerpts and all four
statuses are retained in [p1-focused-build.md](p1-focused-build.md).

### Reporting and source checks

```powershell
pwsh -NoProfile -File wow-viewer/src/viewer/WoWViewer/Workbench/Services/FileOperations/Verify-FileOperations.ps1
```

Exit **0**; repeated on final code:

```text
STRUCTURE_PASS roots=File,View,Help chrome_owners=0 menu_exports_imports=0 terrain_exports=10 terrain_imports=3
REPORTER_PASS checks=7 real_file=I:\parp\parp-tools\wow-viewer\specs\280-ui-audit-refactor\evidence\p1-reporter-artifact.md
```

The seven checks cover positive completion, full-path status, zero/negative work, a thrown
write exception, missing output, and empty output. The [real tiny file](p1-reporter-artifact.md)
tests **only the reporter**; it is not a terrain/model artifact or runtime proof. The script
compiles this reporter in memory and never instantiates the viewer or loads client data.
No C# unit-test suite was rerun. The parent's baseline logs already existed; their seven Core
test failures are not reclassified or fixed here.

## Criterion → real evidence / open gate

| Assigned criterion | Evidence | Status |
|---|---|---|
| Open File no longer requires typed modal | Shared picker call in source; old modal absent in structural check | Code verified; actual open pending parent/operator |
| Every surviving terrain export writes and reports | 10 actions route to existing image encoders and verified output; build + reporter checks | Real terrain exports pending |
| Imports read and apply or are removed | 3 explicit proven apply chains above; zero work/error is failure | Real import application pending |
| GLB actions do not falsely report a write | Unsupported/null conversion throws; output must exist/nonempty | Loadable GLB proof pending |
| Explicit failure and selected path | Seven checks, real reporter file; callback-level catch in source | Reporter proven, actual failures in viewer pending |
| No format/renderer/god-class changes | Scoped source diff; focused build | Source-only proof |

Operator-originated starting state remains: **File mostly works; Export unreliable; Import
non-functional; Open File required a typed path**. This assignment provides no new operator
runtime result with which to label individual actions “works”.

Next bounded step: parent integrates [P3 APIs](p3-top-menu.md), then operator exports each of
the 10 terrain variants plus model/world/DAT outputs, and imports the 3 supported folder types
against a recorded real build/map. Confirm actual files/loadability, visible apply, reported
paths, no-match failure, invalid metadata failure, picker cancel, and model/map changes while
the picker is open. Do not close P1 from compilation or reporter checks alone.
