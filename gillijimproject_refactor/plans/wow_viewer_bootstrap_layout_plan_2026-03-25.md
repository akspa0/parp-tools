# wow-viewer Bootstrap Layout Plan

This document turns the repo-shape prompt into a concrete bootstrap layout for the future wow-viewer repo.

It assumes the user correction from Mar 25, 2026:

- the active MdxViewer PM4 behavior is the de facto runtime reference implementation
- Pm4Research should be ported as the future PM4 library family because PM4 semantics remain under active research
- parp-tools remains the R&D and archaeology repo, not the long-term production home

## First Output

### 1. Minimum repo tree to create on day one

```text
wow-viewer/
  .github/
  docs/
  eng/
  scripts/
  libs/
  src/
    viewer/
      WowViewer.App/
    core/
      WowViewer.Core/
      WowViewer.Core.IO/
      WowViewer.Core.Runtime/
      WowViewer.Core.PM4/
    tools-shared/
      WowViewer.Tools.Shared/
  tools/
    converter/
      WowViewer.Tool.Converter/
    inspect/
      WowViewer.Tool.Inspect/
    catalog/
      WowViewer.Tool.Catalog/
  research/
    pm4/
  testdata/
```

### 2. First-party projects that must exist immediately

- WowViewer.App
- WowViewer.Core
- WowViewer.Core.IO
- WowViewer.Core.Runtime
- WowViewer.Core.PM4
- WowViewer.Tools.Shared
- WowViewer.Tool.Converter
- WowViewer.Tool.Inspect

Tool.Catalog can exist on day one as an empty placeholder or be added immediately if asset-catalog automation is still considered a first-wave requirement.

### 3. Upstream repos that should be auto-cloned immediately

- wowdev/wow-listfile
- wowdev/WoWDBDefs
- wowdev/DBCD
- ModernWoWTools/Warcraft.NET
- Marlamin/WoWTools.Minimaps
- WoW-Tools/SereniaBLPLib

Alpha-Core should remain a tracked dependency target, but whether it is auto-cloned on day one depends on whether the new repo wants to own local SQL tooling immediately or only preserve integration contracts.

### 4. Old repo parts that should not be copied forward blindly

- the old root-folder sprawl itself
- every separate CLI identity from parp-tools
- archived WMOv14 or PM4 helper executables
- WoWRollback GUI and viewer shells
- parpToolbox and PM4Tool as product identities
- historical library layering such as carrying gillijimproject-csharp forward unchanged as a permanent dependency

## Proposed Top-Level Repo Tree

```text
wow-viewer/
  .github/
    workflows/
  docs/
    architecture/
    migration/
    validation/
  eng/
    Directory.Build.props
    Directory.Packages.props
    Version.props
  scripts/
    bootstrap.ps1
    bootstrap.sh
    sync-libs.ps1
    sync-libs.sh
    validate-real-data.ps1
  libs/
    wowdev/WoWDBDefs/
    wowdev/wow-listfile/
    wowdev/DBCD/
    ModernWoWTools/Warcraft.NET/
    Marlamin/WoWTools.Minimaps/
    WoW-Tools/SereniaBLPLib/
    alpha-core/
  src/
    viewer/
      WowViewer.App/
      WowViewer.App.Tests/
    core/
      WowViewer.Core/
      WowViewer.Core.Tests/
      WowViewer.Core.IO/
      WowViewer.Core.IO.Tests/
      WowViewer.Core.Runtime/
      WowViewer.Core.Runtime.Tests/
      WowViewer.Core.PM4/
      WowViewer.Core.PM4.Tests/
    tools-shared/
      WowViewer.Tools.Shared/
      WowViewer.Tools.Shared.Tests/
  tools/
    converter/
      WowViewer.Tool.Converter/
    inspect/
      WowViewer.Tool.Inspect/
    catalog/
      WowViewer.Tool.Catalog/
  research/
    pm4/
      notes/
      reports/
      promoted-from-parp-tools/
  testdata/
    fixtures/
    real-data-notes/
```

## Proposed Project and Solution Tree

### Shipping app and tool solution grouping

- WowViewer.App
  - UI shell, renderer integration, view state orchestration, interactive workflows
- WowViewer.Tool.Converter
  - headless conversion, transfer, repair, batch, and pm4-restore entry points
- WowViewer.Tool.Inspect
  - headless WDT, DBC, MDX-L, PM4, and validation report entry points
- WowViewer.Tool.Catalog
  - optional batch screenshot, minimap, and catalog export entry points

### Shared first-party library grouping

- WowViewer.Core
  - FourCC, binary primitives, build catalogs, common domain models
- WowViewer.Core.IO
  - owned format readers, writers, and conversion contracts
- WowViewer.Core.Runtime
  - data-source, DBC, listfile, SQL, and scene-facing runtime services
- WowViewer.Core.PM4
  - PM4 decode, transform, grouping, research-backed diagnostics, correlation, and export contracts
- WowViewer.Tools.Shared
  - option parsing, operation envelopes, progress, cancellation, manifests, tee logging, report writers

### Research-only grouping

- research/pm4
  - notes, promoted reports, and explicitly non-shipping experiments

## Bootstrap Dependency Matrix

| Dependency | Role in wow-viewer | Baseline or optional | Bootstrap action |
| --- | --- | --- | --- |
| wow-listfile | listfile and path resolution | baseline | auto-clone |
| WoWDBDefs | DBD schemas | baseline | auto-clone |
| DBCD | DBC reader or schema support | baseline | auto-clone |
| Warcraft.NET | upstream format support and reference | baseline | auto-clone |
| WoWTools.Minimaps | minimap extraction support | baseline | auto-clone |
| SereniaBLPLib | BLP image work | baseline | auto-clone |
| Alpha-Core | SQL integration and reference data | baseline if SQL integration is first-wave, otherwise tracked optional | clone or configure path |
| ADTMeta | later evaluation or reference | optional | do not auto-clone initially |
| wow.tools.local | later evaluation or reference | optional | do not auto-clone initially |
| wow.export | later evaluation or reference | optional | do not auto-clone initially |
| MapUpconverter | later evaluation or reference | optional | do not auto-clone initially |

## scripts/bootstrap Responsibilities

### bootstrap scripts must do

- clone or update baseline upstream repos under libs/
- restore NuGet dependencies
- validate the expected folder layout
- generate any local settings templates needed for real-data validation
- surface a clear report when an optional dependency is missing instead of silently failing

### bootstrap scripts must not do

- copy archaeology projects from parp-tools into the new repo
- treat research leftovers as production dependencies by default
- require hand-edited tribal setup steps just to open the solution

## What Stays In parp-tools

- archived or obsolete GUI shells and legacy CLIs
- broad reverse-engineering notes and one-off experiments
- old WMOv14 or PM4 support trees that are evidence only
- PM4Tool and parpToolbox as product identities
- ADTPrefabTool and similar archaeology
- any research that has not been promoted into a tested first-party library contract yet

## First Bootstrap Milestone

The first milestone that proves the repo shape is real is:

1. clone wow-viewer on a clean machine
2. run bootstrap.ps1 or bootstrap.sh successfully
3. open the solution and build WowViewer.Core, WowViewer.Core.IO, WowViewer.Core.PM4, and WowViewer.Tool.Converter
4. load one real map or tile set through the new library surface without the app owning format truth

That milestone is enough to prove the new repo has real boundaries instead of just renamed folders.

## Risks If The Tree Is Done Badly

- if PM4 is not given a first-class Core.PM4 home, its research and runtime contracts will drift again across viewer code, CLIs, and archaeology tools
- if too many one-off tools are copied forward, the new repo will just preserve the same sprawl under different names
- if first-party code lands in libs/, it will become harder to tell what the team actually owns
- if the viewer app still hosts parsing or conversion logic directly, the repo split will be cosmetic only
- if research folders become hidden production dependencies, the new repo will lose clarity immediately

## Bottom Line

- day one should create the main app, the core libraries, the PM4 library, the tools-shared layer, and the converter plus inspect tools
- the PM4 library must exist immediately because PM4 is already a first-class concern and its semantics are still evolving
- parp-tools stays the archaeology repo; wow-viewer gets only the deliberate, re-owned production surfaces

## Validation Status

- planning plus initial workspace scaffold
- a first-pass `wow-viewer/` skeleton now exists under the current workspace with the planned core, tool, and app projects
- `dotnet build .\WowViewer.slnx -c Debug` passed on Mar 25, 2026 for that scaffold
- this is still only a skeleton; no real code-port, runtime validation, or upstream bootstrap cloning has happened yet