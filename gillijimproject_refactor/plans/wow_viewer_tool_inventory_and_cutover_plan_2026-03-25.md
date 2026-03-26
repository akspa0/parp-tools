# wow-viewer Tool Inventory And Cutover Plan

This document inventories the legacy tool sprawl in parp-tools and makes explicit cutover decisions for the production wow-viewer repo.

It refines the repo-shape work in plans/v0_5_0_wow_viewer_bootstrap_and_migration_draft_2026-03-25.md by deciding which historical tools survive as product surfaces, which become shared services only, and which stay behind as archaeology.

## First Output

### 1. Tool families that clearly deserve first-class survival in wow-viewer

- Interactive viewer shell: the current MdxViewer experience should survive as WowViewer.App, including core world browsing, inspection, minimap, spawn/runtime inspection, and selected embedded tool workflows.
- Headless conversion pipeline: the current terrain, WMO, model, and texture conversion surface should survive as one canonical Tool.Converter executable backed by shared services, not as multiple competing CLIs.
- Headless inspection and forensics pipeline: WDT, DBC, MDX-L, and selected PM4/WL audit surfaces should survive as one Tool.Inspect executable over shared services.
- PM4 workspace and library: PM4 deserves first-class survival immediately as a shared library and viewer workspace. The active MdxViewer PM4 behavior is the runtime reference implementation, and Pm4Research should be ported as the evolving PM4 library family because PM4 semantics are still under active research.

### 2. Biggest duplicated tools that should be merged or killed

- WoWMapConverter.Cli, AlphaLkToAlphaStandalone, and overlapping WoWRollback terrain or WMO commands should collapse into one Tool.Converter surface.
- AlphaWdtAnalyzer.Cli and AlphaWdtInspector should collapse into one WDT inspection command group in Tool.Inspect.
- DBCTool should die in favor of DBCTool.V2 behavior only.
- parpToolbox, PM4Tool, and overlapping older PM4 executables should not survive as separate product identities. The PM4 family should center on the current MdxViewer behavior plus a ported Pm4Research-based library, with other PM4 codebases treated as supporting evidence.
- Viewer modal converter dialogs should stop owning distinct conversion logic. They should become thin clients over the same shared services used by Tool.Converter.

### 3. Old executables that look like archaeology, not production assets

- ADTPrefabTool
- DBCTool
- AlphaWdtInspector once its unique diagnostics are merged
- AlphaLkToAlphaStandalone once its current behavior is ported into Tool.Converter
- WoWRollback.Gui, WoWRollback.Viewer, WoWRollback.ViewerModule, WoWDataPlot, and the old WMOv14 subtrees
- parpToolbox and PM4Tool as whole product identities; salvage code and evidence, not the permanent app split
- WlAnalyzer as a hardcoded liquid test harness
- old PM4-oriented helper executables unless a specific capability is deliberately promoted into the new PM4 library or inspect surface

### 4. First migration wave to schedule

1. Build WowViewer.Core.IO plus Tool.Converter by merging WoWMapConverter with the still-useful conversion seams from WoWRollback and AlphaLkToAlphaStandalone.
2. Bootstrap WowViewer.App with the core surviving panels from MdxViewer and re-home the converter dialogs as service-backed workflows instead of app-local conversion logic.
3. Build Tool.Inspect from the WDT, DBC, MDX-L, selected PM4 inspection verbs, and selected liquid inspection surfaces, with PM4 backed by the new Core.PM4 library family rather than a later winner-take-all rewrite.

## Tool Inventory Table

| Current tool or family | Current role in parp-tools | Classification bucket | wow-viewer destination | Shared-core requirements | Cutover decision |
| --- | --- | --- | --- | --- | --- |
| MdxViewer built-in panels and utilities | Main interactive viewer, inspectors, minimap, PM4 overlay, terrain and model workflows | fold into the main viewer as a panel or workflow | src/viewer/WowViewer.App | Core.Runtime scene services, Core.IO readers and writers, shared jobs and reporting | Survive as the main app shell. Do not split its existing tool workflows into separate permanent apps unless they need headless operation. |
| WoWMapConverter.Cli | Unified terrain, WMO, MDX, PM4 export, VLM, and repair CLI | keep as standalone CLI over shared services | tools/converter/WowViewer.Tool.Converter | Core.IO terrain, WMO, model, liquid, and export services; Core.Runtime build and data-source services; Tools.Shared job plumbing | This should be the canonical conversion surface after absorbing the unique non-duplicated WoWRollback and AlphaLkToAlpha logic. |
| AlphaWdtAnalyzer.Cli | Alpha WDT and ADT audit, diff, sanity, and export CLI | keep as standalone CLI over shared services | tools/inspect/WowViewer.Tool.Inspect with wdt verbs | Core.IO WDT and ADT readers and diff services; Tools.Shared reporting | Keep the family, but merge it with AlphaWdtInspector into one inspect surface. |
| AlphaWdtInspector | Parallel Alpha WDT and ADT inspection CLI with heavily overlapping verbs | discard as superseded duplication | none; mine any unique diagnostics into Tool.Inspect | Same WDT and ADT inspection services as AlphaWdtAnalyzer.Cli | Do not port this executable identity forward. Keep it only as reference while extracting any unique analysis behavior. |
| DBCTool | Older DBC compare and dump CLI | discard as superseded duplication | none | None beyond reference to older scripts and assumptions | Treat as deprecated. Preserve only for archaeology until V2 fully covers any missing niche behavior. |
| DBCTool.V2 | Current DBC compare and dump CLI with better config flow | keep as standalone CLI over shared services | tools/inspect/WowViewer.Tool.Inspect with dbc verbs | Core.Runtime DBC schema and build services; Tools.Shared reporting | This is the DBC survivor. Fold its behavior into Tool.Inspect rather than keeping a separate branded app. |
| BlpResizer | Narrow texture downscale and BLP conversion utility | keep as standalone CLI over shared services | tools/converter/WowViewer.Tool.Converter with texture verbs | Core.IO texture and image services with SereniaBLPLib wrapper | Keep headless support. Also expose thin viewer workflow hooks later if useful, but do not make a dedicated GUI app. |
| AlphaLkToAlphaStandalone | Focused LK to Alpha terrain conversion CLI | discard as superseded duplication | none; fold into Tool.Converter terrain verbs | Core.IO terrain and placement conversion services | The behavior matters, the executable does not. Merge its proven path into Tool.Converter. |
| Pm4Research.Core and Pm4Research.Cli | PM4 hypothesis scanning, audit, decoding research, and CLI reports around still-unknown PM4 semantics | keep as standalone CLI over shared services | src/core/WowViewer.Core.PM4 plus tools/inspect/WowViewer.Tool.Inspect pm4 verbs | Core.PM4 decode, linkage, transform, and report services; Tools.Shared reporting | Port this family into wow-viewer as the evolving PM4 library seam, but keep the current MdxViewer PM4 behavior as the runtime reference to preserve what already works. |
| MDX-L_Tool | Alpha 0.5.3 MDX archaeology, inspection, and conversion CLI | keep as standalone CLI over shared services | tools/inspect/WowViewer.Tool.Inspect with mdx-l verbs; optional viewer workflow hooks later | Core.IO early-model readers and exporters; Core.Runtime listfile and texture lookup; image services | Preserve as a headless archaeology surface and optionally wire selected commands into the viewer later. |
| WlAnalyzer | Hardcoded liquid conversion and validation harness | keep as research/reference only in parp-tools | none; move any durable liquid conversion code into Core.IO | Core.IO liquid readers and writers if logic proves reusable | This should not ship as a production executable. Keep the harness as evidence only. |
| parpToolbox | Large PM4, WMO, correlation, and analysis suite with many overlapping CLIs | keep as research/reference only in parp-tools | none as an app identity; mine validated PM4 correlation or export logic into Core.PM4 or Tool.Inspect | Supporting PM4 correlation, export, and report evidence | Valuable source material, but not a product surface to port as-is. |
| PM4Tool | Broad PM4 processing suite with overlapping analysis and export tools | keep as research/reference only in parp-tools | none as an app identity; mine validated geometry and transform logic into Core.PM4 | Supporting PM4 geometry, transform, and validation evidence | Preserve evidence and reusable algorithms, but do not keep this as a separate product identity. |
| ADTPrefabTool | Poorly documented Alpha WDT or ADT prefab experiment | keep as research/reference only in parp-tools | none | None until its purpose is re-established | Treat as archaeology unless a concrete surviving feature is rediscovered. |
| WoWRollback.Cli | Historical Alpha to LK rollback and packing CLI with major overlap against WoWMapConverter | discard as superseded duplication | none; fold unique still-useful verbs into Tool.Converter and Tool.Inspect | Core.IO terrain, WMO, packing, DBC, and report services; Tools.Shared job harness | The code still matters, but the final repo should not keep both WoWRollback.Cli and Tool.Converter. |
| WoWRollback.PM4Module | PM4 to ADT restoration pipeline and PM4 OBJ inspection helpers | keep as standalone CLI over shared services | tools/converter/WowViewer.Tool.Converter with pm4-restore verbs; selected viewer workflow hooks later | Core.PM4 geometry and matching services, Core.IO ADT writer services, Tools.Shared job plumbing | Preserve this pipeline as a consumer of the new PM4 library family rather than as its own source of PM4 truth. |
| WoWRollback.AdtConverter and WoWRollback.LkToAlphaModule | Specialized terrain conversion modules duplicated elsewhere | discard as superseded duplication | none; merge behavior into Tool.Converter services | Core.IO terrain and placement conversion services | Keep their algorithms, not their executable or module boundaries. |
| WoWRollback.AnalysisModule and WoWRollback.DbcModule | Terrain and DBC analysis helpers | fold into the main viewer as a panel or workflow | Mostly Tool.Inspect command groups; selective viewer diagnostic hooks | Core.Runtime DBC services, Core.IO terrain diagnostics, Tools.Shared reporting | Rebuild only the useful reports inside Tool.Inspect and thin viewer diagnostics. Do not preserve the old module split. |
| WoWRollback.MinimapModule | Minimap extraction or analysis helper | keep as standalone CLI over shared services | tools/catalog/WowViewer.Tool.Catalog with minimap verbs | Core.Runtime data-source and listfile services, texture or image services | Useful as a batch surface if screenshot or minimap catalog work remains important. |
| WoWRollback.Orchestrator | Batch runner over other rollback modules | keep as standalone CLI over shared services | tools-shared/WowViewer.Tools.Shared job runner plus Tool.Converter batch verbs | Tools.Shared batch execution, logging, manifest, and report services | Preserve the concept, not the executable name. The new repo should have one shared job harness. |
| WoWRollback.Gui, WoWRollback.Viewer, and WoWRollback.ViewerModule | Older GUI or viewer surfaces parallel to MdxViewer | keep as research/reference only in parp-tools | none | None | Do not port parallel GUI shells. MdxViewer already won the interactive surface. |
| WoWRollback.TestHarness and WoWRollback.Verifier | Internal harness and verification utilities | keep as research/reference only in parp-tools | none as tools; move any durable checks into tests under src/core or src/viewer | Tests and validation scaffolding only | Useful for reference but not shipping executables. |
| WoWRollback.WMOv14 subtree, WoWDataPlot, and old_sources | Legacy conversion, plotting, and archived code | keep as research/reference only in parp-tools | none | None | These are archaeology and evidence, not candidate production surfaces. |

## Destination Mapping By Surviving Tool Family

| Surviving family | wow-viewer surface | Source roots to mine first | Notes |
| --- | --- | --- | --- |
| Interactive viewer | src/viewer/WowViewer.App | src/MdxViewer plus selected runtime and data-source seams | This remains the single interactive product shell. |
| Conversion pipeline | tools/converter/WowViewer.Tool.Converter | WoWMapConverter.Cli, WoWMapConverter.Core, AlphaLkToAlphaStandalone, WoWRollback conversion modules | One canonical headless converter is enough. Viewer dialogs become thin clients over the same services. |
| Inspection and archaeology pipeline | tools/inspect/WowViewer.Tool.Inspect | AlphaWdtAnalyzer.Cli, DBCTool.V2, MDX-L_Tool, selected WlAnalyzer reports, selected WoWRollback analysis helpers | This is where most legacy diagnostic CLIs should converge. |
| Asset or minimap catalog work | tools/catalog/WowViewer.Tool.Catalog | Selected MdxViewer asset catalog capture paths, WoWRollback.MinimapModule | This is optional early, but it is the right home if batch catalog or screenshot generation remains important. |
| PM4 workspace and library | src/core/WowViewer.Core.PM4 plus viewer PM4 workspace, inspect pm4 verbs, and converter pm4-restore verbs | Current MdxViewer PM4 overlay and runtime behavior first, then Pm4Research.Core and Pm4Research.Cli, then selected WoWRollback.PM4Module logic, then supporting evidence from parpToolbox and PM4Tool | The runtime reference is MdxViewer. Pm4Research becomes the new PM4 library family because PM4 semantics remain under active research. |

## Shared-Service Requirements By Tool Family

| Tool family | Required shared services in wow-viewer | Primary source roots |
| --- | --- | --- |
| Terrain and map conversion | ADT and WDT readers and writers, placement and liquid contracts, split ADT support, repair and diff services | WoWMapConverter.Core, MdxViewer terrain runtime, WoWRollback terrain modules, AlphaWDTAnalysisTool |
| Model and WMO conversion | MDX and M2 readers and exporters, WMO version conversion, texture lookup and export, early-model profile routing | WoWMapConverter.Core, MDX-L_Tool, MdxViewer runtime and format profile code, WoWRollback WMO helpers |
| DBC, listfile, and build metadata | Build catalog, listfile resolution, DBC schema and query services | MdxViewer data-source and DBC paths, DBCTool.V2, WoWRollback.DbcModule |
| Texture and image work | BLP read or write, PNG export, atlas or resize helpers | BlpResizer, MdxViewer export paths, SereniaBLPLib integration |
| PM4 geometry and correlation | Core.PM4 decode, transform, grouping, correlation, export, research, and validation services | Current MdxViewer PM4 overlay and runtime code, Pm4Research.Core, Pm4Research.Cli, WoWRollback.PM4Module, selected evidence from PM4Tool and parpToolbox |
| Headless jobs and reporting | Shared option parsing, manifests, tee logging, batch orchestration, durable report writers | WoWRollback.Orchestrator, WoWMapConverter.Cli batch paths, parpToolbox reporting patterns |
| Runtime viewer services | Scene loading, SQL integration, DBC and listfile runtime access, asset catalog hooks, tool-state persistence | MdxViewer DataSources, Population, ViewerApp partials |

## MdxViewer Panel And Workflow Cutover Decisions

| Current panel or workflow | Decision | wow-viewer destination | Why |
| --- | --- | --- | --- |
| Navigator | survive and rebuild | WowViewer.App main dock | This is core browsing UX, not a separate tool. |
| Inspector | survive and rebuild | WowViewer.App main dock | This is core scene inspection UX. |
| Minimap | survive and rebuild | WowViewer.App panel | It is core world navigation, not optional tool sprawl. |
| Asset catalog browser | survive and rebuild | WowViewer.App browser now; selected batch capture later in Tool.Catalog | The interactive browser belongs in the app, but batch catalog jobs do not. |
| Log Viewer | merge | WowViewer.App Diagnostics dock | It should not remain a one-off window once diagnostics are regrouped. |
| Perf Window | merge | WowViewer.App Diagnostics dock | Same reason as Log Viewer. |
| Render Quality panel | merge | WowViewer.App Render and Diagnostics dock | Keep the controls, not the separate identity. |
| WDL Preview | survive but rebuild as a workflow | WowViewer.App open or preview workflow | Useful before world load, not as a permanent panel. |
| Chunk Clipboard | survive but defer | WowViewer.App Terrain Editing workspace | Valuable, but only after terrain editing services are re-homed cleanly. |
| PM4 Alignment | merge | WowViewer.App PM4 workspace over Core.PM4 | It should merge with PM4 correlation, selection, and research-backed diagnostics rather than stand alone forever. |
| PM4 or WMO Correlation | merge | WowViewer.App PM4 workspace plus Tool.Inspect pm4 reports | One PM4 workspace is cleaner than many PM4 windows. |
| Map Converter dialog | survive as a workflow | WowViewer.App tool workflow over Tool.Converter services | The UI can stay, but conversion logic must move out of the app. |
| WMO Converter dialog | survive as a workflow | WowViewer.App tool workflow over Tool.Converter services | Same pattern as Map Converter. |
| VLM export dialog | survive as a workflow | WowViewer.App job workflow over Tool.Converter services | Keep the job surface, not viewer-owned conversion code. |
| Terrain Texture Transfer dialog | survive as a workflow | WowViewer.App tool workflow over Tool.Converter services | Useful interactive wrapper over shared terrain transfer services. |
| Chunk visibility controls | survive | WowViewer.App Inspector | Core inspection surface. |
| SQL spawn loader and actor controls | survive and rebuild | WowViewer.App Inspector over Core.Runtime SQL services | This is product-facing viewer behavior. |
| Taxi route inspector | survive and rebuild | WowViewer.App Inspector | Same reason as SQL spawn loader. |
| WDT liquid inspector | survive and rebuild | WowViewer.App Inspector | Same reason as the other world-inspection tools. |

## Archaeology-Only Leftovers

- DBCTool
- AlphaWdtInspector after merger
- AlphaLkToAlphaStandalone after merger
- ADTPrefabTool unless its purpose is rediscovered with concrete code value
- parpToolbox and PM4Tool as full product identities
- WlAnalyzer as a hardcoded validation harness
- WoWRollback.Gui, WoWRollback.Viewer, WoWRollback.ViewerModule
- WoWRollback.TestHarness and WoWRollback.Verifier as tools; move only durable checks into tests
- WoWRollback.WMOv14 old sources, WoWDataPlot, and archived subtrees

## Legacy Executables That Should Not Be Ported Forward As Executables

- DBCTool
- AlphaWdtInspector
- AlphaLkToAlphaStandalone
- WoWRollback.Cli as a permanent brand or separate converter surface
- WoWRollback.Gui
- WoWRollback.Viewer
- WoWRollback.ViewerModule
- parpToolbox as a permanent product split
- PM4Tool as a permanent product split
- ADTPrefabTool

## First Migration Wave

### Wave 1: Conversion Core and CLI

- Build WowViewer.Core, WowViewer.Core.IO, WowViewer.Core.PM4, and WowViewer.Tools.Shared skeletons.
- Port WoWMapConverter.Core first, then merge in the still-useful AlphaLkToAlpha and WoWRollback terrain and WMO conversion seams.
- Port Pm4Research.Core into Core.PM4 while preserving current MdxViewer PM4 behavior as the runtime reference contract.
- Publish one Tool.Converter executable with terrain, WMO, model, texture, pm4-restore, and batch verbs.

### Wave 2: Viewer Shell And Surviving Panels

- Bootstrap WowViewer.App with the core dock layout from MdxViewer.
- Move Navigator, Inspector, Minimap, runtime loading, SQL hooks, the PM4 workspace, and selected inspection workflows onto shared runtime services.
- Rebuild converter dialogs as thin job launchers over shared converter services.

### Wave 3: Inspection CLI

- Build Tool.Inspect from AlphaWDT, DBC, MDX-L, PM4, and selected liquid or terrain inspection workflows.
- Collapse AlphaWdtAnalyzer.Cli, DBCTool.V2, and selected WoWRollback analysis helpers into one inspect app.
- Re-home Pm4Research.Cli report families onto Core.PM4-backed verbs while keeping unstable research seams clearly labeled as experimental.

## Major Cutover Risks

- PM4 semantic drift: PM4 semantics are still under active research, so the new repo must preserve the current MdxViewer behavior as its runtime reference while allowing Core.PM4 to keep evolving through Pm4Research-backed work. Treating every legacy PM4 executable as equal would freeze disagreement into the new repo.
- Converter duplication: WoWMapConverter and WoWRollback both own terrain and WMO conversion logic. If both are migrated as-is, wow-viewer will inherit the same ambiguity it is supposed to eliminate.
- Viewer over-coupling: current MdxViewer dialogs and tool panels often reach directly into app state. If shared services are not extracted first, the new repo will just rename the old coupling.
- Validation gap: many tools are build-validated or research-validated, but not fully signed off against broad real datasets. The cutover must keep the project memory rule that real-data validation matters more than synthetic or archival tests.
- Archaeology creep: the repo contains many historically useful executables. If they are all given permanent homes in wow-viewer, the new repo will repeat the same sprawl under different names.

## Bottom Line

- wow-viewer should ship one interactive app, one converter CLI, one inspect CLI, and optionally one catalog CLI.
- PM4 should survive immediately as a shared library plus viewer workspace, with MdxViewer as the behavioral reference and Pm4Research as the evolving library seam.
- Most historical executables should not survive as executables. Their algorithms or evidence should be mined into shared services, then left behind in parp-tools.

## PM4 Correction

- Current MdxViewer behavior is the de facto PM4 reference implementation for runtime behavior.
- Pm4Research should be ported into wow-viewer as the PM4 library family because PM4 semantics are still being discovered.
- parpToolbox, PM4Tool, and other PM4-heavy tools remain supporting evidence, not primary PM4 truth.


## Validation Status

- planning and documentation only
- no code changes, builds, or runtime validation were performed for this slice