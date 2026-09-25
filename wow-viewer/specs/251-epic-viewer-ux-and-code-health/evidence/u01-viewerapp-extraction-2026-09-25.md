# Receipt — U-01 ViewerApp extraction campaign (2026-09-25)

Epic 251 item U-01. Covers **U01-T006** (E3: menu bar + converter dialogs), **U01-T010**, **U01-T011** and
**U01-T012** (operator direction 2026-09-25, spec.md amendment). It claims **no runtime behaviour**: the
viewer has no unit tests over `ViewerApp`, so build, audit and tests below prove only that the code
compiles, that moved bodies are unchanged, and that nothing else regressed. Operator smoke is
**U01-T007** / **U01-T013**.

## Operator direction

*"continue, so that our ViewerApp is as concise as possible, and not a giant headache to edit later on.
That's been a hindering factor for the project for a long time, so making the core of the viewer's code
a bit smaller is a welcome change so we can build it out a bit more sane."*

## Result

| Measure | Before (`ffad810`) | After (`HEAD`) |
|---|---|---|
| `ViewerApp` partial class, all files | 42,973 lines in 29 files | **2,512 lines in 6 files** |
| `ViewerApp.cs` | 16,746 | **1,725** |
| Owned services / static helpers created | — | 51 service classes + 3 static helper classes (`TerrainChunkMath`, `SceneViewportMath`, `ImGuiListLayout`) |
| Largest new file | — | 1,897 lines (`TerrainWeakSignalRestoreService.cs`); every file < 2,000 |

What remains in `ViewerApp` (the app shell):

| File | Lines | Contents |
|---|---|---|
| `ViewerApp.cs` | 1,725 | shared app-state fields, nested enums/DTOs, `ViewerApp()` composition root, window lifecycle (`Run`, `OnLoad`, `OnUpdate`, `OnRender`, `DrawUI`, keyboard/mouse input, `Dispose`) |
| `ViewerApp_Host.cs` | 492 | explicit `IViewerAppHost` implementation (414 one-line forwarding members; plumbing only) |
| `ViewerApp_Editor.cs` | 147 | editor composition (`EnsureEditorPages`, `EnsureEditorHost` hand `this` to the editor pages/adapters), `IEditorPageHost` forwarding, nested editor adapters |
| `ViewerKeyBindings.cs` | 84 | active keyboard context |
| `ViewerApp_PhaseLayers.cs` | 37 | shared map-sort selector |
| `ViewerApp_Sidebars.cs` | 27 | three utility-page fields |

Services (`Workbench/Services/**`, namespace `WoWViewer`; files per class include per-source partials and
`*.Host.cs` bridge files):

| Folder | Class | Lines | Files |
|---|---|---|---|
| `Workbench/Services/Archaeology` | `ArchaeologyPanelService` | 2,445 | 6 |
| `Workbench/Services/AreaContext` | `AreaContextService` | 257 | 1 |
| `Workbench/Services/Audio` | `AudioPanelService` | 270 | 2 |
| `Workbench/Services/CameraPaths` | `CameraPathsService` | 1,327 | 3 |
| `Workbench/Services/CaptureAutomation` | `CaptureAutomationService` | 2,363 | 4 |
| `Workbench/Services/CascAhdr` | `CascAhdrSourceService` | 596 | 2 |
| `Workbench/Services/Chrome` | `ViewerChromeService` | 956 | 4 |
| `Workbench/Services/ChunkEdit` | `ChunkEditService` | 925 | 1 |
| `Workbench/Services/ChunkEdit` | `TerrainChunkMath` | 428 | 1 |
| `Workbench/Services/ClientDialogs` | `ClientDialogsService` | 673 | 2 |
| `Workbench/Services/Converters` | `ConverterDialogsService` | 1,136 | 1 |
| `Workbench/Services/DataSourceSession` | `DataSourceSessionService` | 724 | 1 |
| `Workbench/Services/DatasetCatalog` | `DatasetCatalogService` | 216 | 2 |
| `Workbench/Services/DatasetExport` | `DatasetExportDialogsService` | 1,033 | 1 |
| `Workbench/Services/EditorPanels` | `EditorPanelsService` | 1,364 | 2 |
| `Workbench/Services` | `IViewerAppHost` (interface) | 499 | 1 |
| `Workbench/Services/InspectorPayloads` | `InspectorPayloadsService` | 538 | 2 |
| `Workbench/Services/Investigation` | `InvestigationService` | 1,111 | 3 |
| `Workbench/Services/Lighting` | `LightingPanelService` | 350 | 2 |
| `Workbench/Services/LogViewer` | `LogViewerService` | 184 | 2 |
| `Workbench/Services/MainMenu` | `MainMenuBarService` | 1,338 | 4 |
| `Workbench/Services/MinimapStatus` | `MinimapAndStatusService` | 927 | 2 |
| `Workbench/Services/MlTraining` | `MlTrainingService` | 947 | 2 |
| `Workbench/Services/ModelInspector` | `ModelInspectorPanelService` | 1,042 | 4 |
| `Workbench/Services/ModelLoading` | `StandaloneModelLoaderService` | 1,787 | 2 |
| `Workbench/Services/Navigator` | `NavigatorPanelService` | 706 | 3 |
| `Workbench/Services/PlacementEdit` | `PlacementEditService` | 794 | 1 |
| `Workbench/Services/Pm4Workbench` | `Pm4WorkbenchService` | 4,825 | 7 |
| `Workbench/Services/ProjectOutput` | `ProjectOutputService` | 195 | 1 |
| `Workbench/Services/RenderQuality` | `RenderQualityService` | 187 | 2 |
| `Workbench/Services/Scene` | `SceneHoverAndPickService` | 1,182 | 3 |
| `Workbench/Services/Scene` | `SceneViewportMath` | 129 | 1 |
| `Workbench/Services/Settings` | `SettingsWindowService` | 383 | 2 |
| `Workbench/Services/Settings` | `ViewerSettingsService` | 721 | 1 |
| `Workbench/Services/ShellLayout` | `ShellLayoutService` | 913 | 1 |
| `Workbench/Services/SqlSpawnStreaming` | `SqlSpawnStreamingService` | 428 | 1 |
| `Workbench/Services/StartupAutomation` | `StartupAutomationService` | 607 | 2 |
| `Workbench/Services/Stratigraphy` | `StratigraphyService` | 248 | 1 |
| `Workbench/Services/SynthesizedMinimap` | `SynthesizedMinimapExportService` | 606 | 2 |
| `Workbench/Services/TaxiAreaPoi` | `TaxiAndAreaPoiSelectionService` | 653 | 1 |
| `Workbench/Services/TaxiAreaPoi` | `TaxiPanelService` | 639 | 4 |
| `Workbench/Services/TerrainAnalysis` | `TerrainAnalysisModels.cs` (terrain-analysis model types) | 89 | 1 |
| `Workbench/Services/TerrainAnalysis` | `TerrainAnalysisService` | 823 | 2 |
| `Workbench/Services/TerrainControls` | `TerrainControlsPanelService` | 960 | 3 |
| `Workbench/Services/TerrainInspection` | `TerrainInspectionPanelService` | 521 | 2 |
| `Workbench/Services/TerrainQuery` | `TerrainQueryService` | 582 | 3 |
| `Workbench/Services/TerrainTileIo` | `TerrainTileIoService` | 972 | 1 |
| `Workbench/Services/TerrainWeakSignal` | `TerrainWeakSignalRestoreService` | 1,897 | 1 |
| `Workbench/Services/Themes` | `ThemesService` | 298 | 2 |
| `Workbench/Services/WdlPreview` | `WdlPreviewService` | 481 | 2 |
| `Workbench/Services/WmoGroups` | `WmoGroupsPanelService` | 291 | 2 |
| `Workbench/Services/WorkbenchPanels` | `WorkbenchPanelsService` | 1,486 | 2 |
| `Workbench/Services/Workspaces` | `WorkspacesService` | 536 | 2 |
| `Workbench/Services/WorldLoading` | `WorldLoaderService` | 821 | 3 |
| `Workbench/Services/WorldObjects` | `WorldObjectsPanelService` | 942 | 2 |
| `UI` | `ImGuiListLayout` | 51 | 1 |

## Method (same rules as E1)

1. **Selection** — a Roslyn member map of all `ViewerApp` partial files plus a closure: members used only by
   a cluster move with it (including the fields that are its own state). Members already referenced by an
   earlier service stay unless seeded explicitly.
2. **Verbatim move** — member text (with its leading comments) is copied unchanged into
   `internal sealed class <Feature>Service`. Members from each source file land in their own partial file
   carrying **that file's original `using` directives**, so every name resolves exactly as before; adding
   usings to moved code was avoided on purpose (a new namespace could silently offer a better extension
   method).
3. **Host contract** — the only way back to app state is `IViewerAppHost`
   (`Workbench/Services/IViewerAppHost.cs`), implemented explicitly in `ViewerApp_Host.cs`. Mutable fields
   are `ref`-returning properties, so moved ImGui code still passes them by `ref`. Each service re-declares
   what it uses as private bridge members with the old names (in `<Service>.Host.cs` when the bridged types
   come from several files).
4. **Delegation** — `ViewerApp` keeps one field per service, built in the `ViewerApp()` constructor (the
   composition root). Remaining references were rewritten `X` → `_service.X` (or `Service.X` for statics)
   by a syntax-aware pass that skips member-access names, initializer targets, named arguments and anything
   shadowed by a local or parameter.
5. **Visibility only** — `private` → `internal` where the split needs cross-class access. No body, signature
   or ordering change.
6. **Hard stops** — a member using bare `this` or an unqualified `GetType()`/`ToString()`/… blocks the move
   unless reviewed (`EnsureEditorPages`/`EnsureEditorHost` stay in `ViewerApp` for this reason); a new class
   name that already exists anywhere, or equals a host member name, aborts the step.
7. **Per step** — extract → visibility fixes → line-multiset audit → full-solution build; the step is
   committed only on `0 Error(s)`.

Manual edits outside the mechanical rules (all receiver-only):

- Two nested editor adapters call moved members through their `ViewerApp _app` reference:
  `_app.MaterializeReconciliationOutput(...)` → `_app._archaeologyPanel.MaterializeReconciliationOutput(...)`,
  `_app.GetCurrentCaptureMapName()` → `_app._captureAutomation.GetCurrentCaptureMapName()` (×2).
- `SqlPopulationService` was renamed `SqlSpawnStreamingService` before commit because its name read the same
  as the host property for the existing `SqlWorldPopulationService` field.
- Two file-level doc comments of deleted partial files were carried into the new files as notes.

## Per-step ledger (`ViewerApp.cs` size after each commit)

| Commit | `ViewerApp.cs` | Step |
|---|---|---|
| `71c1dfe` | 14,777 | converter and dataset-export dialogs out of ViewerApp |
| `213f5bb` | 12,992 | terrain weak-signal restore out of ViewerApp |
| `2910328` | 12,117 | terrain tile import/export out of ViewerApp |
| `09d3b5b` | 11,741 | pure terrain-chunk math into TerrainChunkMath |
| `1c82172` | 11,014 | terrain chunk tools out of ViewerApp |
| `7ced764` | 10,298 | placement editing out of ViewerApp |
| `b858915` | 10,031 | SQL spawn streaming out of ViewerApp |
| `e00ff54` | 9,954 | scene viewport math into SceneViewportMath |
| `4948818` | 9,392 | taxi and area-POI selection out of ViewerApp |
| `4f3f722` | 8,936 | scene hover, pick and click selection out of ViewerApp |
| `508bb8c` | 8,145 | shell dock layout out of ViewerApp |
| `fd8f02e` | 7,929 | WDL map preview out of ViewerApp |
| `f119a8c` | 7,752 | area context out of ViewerApp |
| `a70b0e1` | 6,190 | standalone model loading out of ViewerApp |
| `72bae63` | 5,590 | world loading out of ViewerApp |
| `3198eb4` | 4,982 | data-source session out of ViewerApp |
| `e22fe24` | 4,756 | terrain queries and editor overlays out of ViewerApp |
| `4facf86` | 4,582 | stratigraphy analysis out of ViewerApp |
| `241486e` | 4,465 | project output paths out of ViewerApp |
| `c071f7a` | 3,805 | world objects panel out of ViewerApp |
| `ca22013` | 3,266 | settings persistence out of ViewerApp |
| `64b4434` | 3,263 | CASC/AHDR sources and client dialogs out of ViewerApp |
| `c1644c6` | 2,251 | main menu bar out of ViewerApp (E3) |
| `aeecd8e` | 2,234 | PM4 workbench UI out of ViewerApp |
| `a5ada6a` | 2,235 | taxi panel out of ViewerApp |
| `25dad42` | 2,182 | archaeology panels out of ViewerApp |
| `e9937d9` | 2,145 | model inspector panel out of ViewerApp |
| `95d62a2` | 2,133 | terrain controls panel out of ViewerApp |
| `3154906` | 2,044 | navigator panel out of ViewerApp |
| `a4fa72b` | 2,046 | viewer chrome out of ViewerApp |
| `a3a5e12` | 2,048 | lighting panel (whole ViewerApp_Lighting.cs) out of ViewerApp |
| `bc72a9b` | 2,050 | audio panel (whole ViewerApp_Audio.cs) out of ViewerApp |
| `2581672` | 2,052 | UI themes (whole ViewerApp_Themes.cs) out of ViewerApp |
| `7ae755d` | 2,055 | terrain inspection panel (whole ViewerApp_TerrainInspection.cs) out of ViewerApp |
| `8eb91ca` | 2,057 | visual investigation (whole ViewerApp_Investigation.cs) out of ViewerApp |
| `c465243` | 2,061 | inspector payloads (whole ViewerApp_InspectorPayloads.cs) out of ViewerApp |
| `3ebd415` | 2,072 | workbench and inspector panels out of ViewerApp |
| `45ac89e` | 2,074 | log viewer window out of ViewerApp |
| `fd0e329` | 2,076 | render quality settings out of ViewerApp |
| `38b2642` | 2,078 | dataset catalog panel out of ViewerApp |
| `da39773` | 2,080 | standalone WMO group overlays out of ViewerApp |
| `0d63458` | 2,085 | settings window out of ViewerApp |
| `589d28d` | 2,070 | synthesized minimap export out of ViewerApp |
| `9998254` | 2,072 | ML training panel out of ViewerApp |
| `fc28b69` | 2,059 | terrain analysis (currently unreferenced UI) out of ViewerApp |
| `1560b70` | 2,060 | minimap window and status content out of ViewerApp |
| `6cd1586` | 2,063 | startup automation out of ViewerApp |
| `42a4038` | 2,068 | camera paths out of ViewerApp |
| `1ee61f6` | 2,072 | capture automation out of ViewerApp |
| `27daf58` | 2,076 | workspaces out of ViewerApp |
| `7c42bec` | 2,080 | editor panels out of ViewerApp |
| `79fbafa` | 1,725 | consolidate the IViewerAppHost implementation |
| `ca0aa36` | 1,725 | ImGui list layout helpers into ImGuiListLayout |
| `1c827c5` | 1,725 | tidy generated service headers |

## Verification

Environment: Linux container, .NET SDK 10.0.112, `-p:EnableWindowsTargeting=true`.

| Check | Result |
|---|---|
| `dotnet build WowViewer.slnx -c Debug -p:EnableWindowsTargeting=true` after every step | exit 0, `0 Error(s)` (a step that did not build was never committed; two such steps were fixed and squashed before push) |
| End-to-end line-multiset audit, `ffad810` → `HEAD` (all `ViewerApp` partials vs partials + every new file, undoing only receiver qualification and `private`/`internal`) | **29 original lines absent, all boilerplate**: the class declaration (gains `, IViewerAppHost`), 23 `public partial class ViewerApp` headers of deleted files, 5 file-level doc comments of deleted files. Every added line is scaffolding (headers, bridge/interface members, composition root, file-origin comments). |
| Viewer compiler warnings, `ffad810` vs `HEAD` (clean worktree, `--no-incremental`, messages normalised for the new owning class) | 520 → 540. **+24 `CS0618`**: uses of the `[Obsolete]` `ShellPanelId` enum are now reported because they sit outside its declaring type (the moved shell/workbench code); the uses themselves are unchanged. **−4 `CS0169`**: the compiler stopped reporting two unreferenced private fields in `Rendering/ModelRenderer.cs` (`MdxRenderer._uFlipTexU`, `_flipTextureUForCurrentDraw`); bisected to `4f3f722`; the file is identical and both fields are unreferenced at both commits — a reporting difference only. `CS0105` duplicate-using noise copied from `ViewerApp.cs`'s header was removed from the generated files (`1c827c5`). |
| `dotnet test WowViewer.slnx -c Debug -p:EnableWindowsTargeting=true` | exit 1 with the **identical 26 pre-existing failures** as the session baseline (missing `test_data/`, Windows-path expectations). Totals: Core.Tests 1,578 passed / 18 failed / 1 skipped; Core.PM4.Tests 95 / 8; Core.Editor.Tests 91 / 0; Core.Curation.Tests 39 / 0. |

## Criterion → evidence

| Task | Criterion | Evidence |
|---|---|---|
| U01-T006 (E3) | `DrawMenuBar`, converter dialogs, world-objects panel out of `ViewerApp.cs` | `MainMenuBarService` (`c1644c6`), `ConverterDialogsService` + `DatasetExportDialogsService` (`71c1dfe`), `WorldObjectsPanelService` (`c071f7a`) |
| U01-T010 | tooling, host, composition root; converter + dataset-export dialogs | `IViewerAppHost`, `ViewerApp()` composition root (`71c1dfe`) |
| U01-T011 | remaining `ViewerApp.cs` clusters | ledger `213f5bb` … `c1644c6`; `ViewerApp.cs` 14,777 → 2,251 |
| U01-T012 | over-budget and feature partial files | ledger `aeecd8e` … `ca0aa36`; `ViewerApp_Sidebars` 5,765 → 27, `ViewerApp_Pm4Utilities` 4,178 → deleted, `ViewerApp_CaptureAutomation` 2,244 → deleted; 24 partial files removed (23 deleted; `ViewerApp_TerrainAnalysis.cs`'s model types moved to `Workbench/Services/TerrainAnalysis/TerrainAnalysisModels.cs`), 1 added (`ViewerApp_Host.cs`) |
| All | behaviour-preserving | audit above; builds; unchanged test failure set |
| All | runtime | **not claimed** — U01-T007 / U01-T013 operator smoke |

## Findings for the operator (no action taken)

- `TerrainAnalysisService.DrawTerrainAnalysisContent` (formerly `ViewerApp_TerrainAnalysis.cs`) is not
  referenced anywhere: the terrain-analysis UI is currently unreachable. Moved verbatim; delete or wire up is
  an operator decision.
- The shell-panel system (`ShellPanelId`) is marked obsolete ("will be removed in 070") and is still used by
  `ShellLayoutService`, `WorkbenchPanelsService` and the menu; the new `CS0618` warnings list the sites.
- `ViewerApp.cs`'s own header has duplicate `using` lines (pre-existing, left untouched).

## Operator smoke (U01-T007 / U01-T013, PowerShell 7)

```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
dotnet run --project I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug
```

Exercise each moved surface once: every main-menu item; map/WMO converter, VLM export, MK harvest, terrain
texture transfer, synthesized minimap export and client-selection dialogs; open an MPQ/CASC/AHDR source and a
map (default spawn, WDL preview); load standalone M2/MDX/WMO models; hover/click selection, taxi and area-POI
picking; navigator, world objects, model inspector, terrain controls, archaeology, PM4 workbench, editor and
workbench/inspector panels; toolbar, bottom bar, splitters and dock layout (incl. layout restore after
restart, i.e. settings load/save); chunk clipboard, tile import/export, weak-signal restore, stratigraphy;
camera paths, capture queue, video recording, taxi-ride camera; lighting, audio, themes, render quality, log
viewer, ML training, startup automation (`--` command-line captures).
