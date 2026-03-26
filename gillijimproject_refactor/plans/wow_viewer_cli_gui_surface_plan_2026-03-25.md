# wow-viewer CLI GUI Surface Plan

This document defines how wow-viewer should expose the same capability through both CLI and GUI surfaces without duplicating format logic.

It also applies the PM4 correction from Mar 25, 2026:

- MdxViewer is the runtime reference implementation for PM4 behavior
- Core.PM4 should be ported from Pm4Research while staying aligned with current MdxViewer behavior

## First Output

### 1. Tool families that most need dual-surface support

- terrain and map conversion
- WMO and model conversion
- PM4 inspection, correlation, and restore workflows
- WDT and DBC inspection
- texture and minimap export or catalog jobs

### 2. Shared services they should sit on

- WowViewer.Core.IO for file readers, writers, and conversion contracts
- WowViewer.Core.Runtime for data-source, build, DBC, listfile, and SQL access
- WowViewer.Core.PM4 for PM4 decode, grouping, transforms, correlation, and research-backed reports
- WowViewer.Tools.Shared for operation envelopes, progress, cancellation, manifests, and report writers

### 3. Capabilities that should stay CLI-only or GUI-only for now

- CLI-only for now
  - PM4 hypothesis scans and unstable research reports
  - batch catalog, screenshot, and large export jobs
  - wide WDT or DBC diff jobs intended for automation
- GUI-only for now
  - interactive world navigation and live scene inspection
  - camera-driven PM4 alignment, selection, and preview overlays
  - SQL spawn visualization and other scene-state-heavy runtime features

### 4. First end-to-end workflow to implement both ways

- terrain texture transfer or map conversion is the first dual-surface workflow to build because it already exists in the viewer and has a clear headless counterpart in WoWMapConverter

## Service-Layer Breakdown

### Operation boundary

Every substantial tool capability should be represented as a service operation with:

- request object
- validation result
- progress callbacks
- cancellation token
- result object with durable report paths and machine-readable metadata

The same operation should be invokable from:

- Tool.Converter
- Tool.Inspect
- Tool.Catalog when relevant
- WowViewer.App workflow wrappers

### Core service families

| Service family | Responsibility | Primary consumers |
| --- | --- | --- |
| TerrainConversionService | ADT and WDT conversion, repair, texture transfer, validation | Tool.Converter, WowViewer.App |
| WmoModelConversionService | WMO and model conversion and export | Tool.Converter, WowViewer.App, Tool.Inspect |
| DbcInspectionService | DBC query, compare, dump, and build-aware lookup | Tool.Inspect, WowViewer.App diagnostics |
| WdtInspectionService | WDT and ADT sanity, diff, and summary reports | Tool.Inspect, WowViewer.App preflight workflows |
| Pm4WorkspaceService | PM4 decode, grouping, transforms, correlation, validation, and report generation | WowViewer.App, Tool.Inspect, Tool.Converter pm4-restore |
| TextureImageService | BLP image work, resizing, atlas, preview exports | Tool.Converter, Tool.Catalog, WowViewer.App |
| CatalogCaptureService | batch asset or minimap capture and metadata export | Tool.Catalog, WowViewer.App |

## CLI Command Tree

```text
wowviewer-converter
  terrain
    convert
    convert-lk-to-alpha
    texture-transfer
    repair
  wmo
    convert
    convert-to-alpha
    info
  model
    convert-mdx
    convert-m2-to-mdx
  texture
    resize
    export
  pm4
    restore-adt
  batch
    run

wowviewer-inspect
  wdt
    summary
    sanity
    diff
    dump
  dbc
    compare
    dump
    query
  mdx-l
    info
    convert
    batch
  pm4
    inspect
    validate
    export-json
    correlation-report
    scan-hypotheses
    scan-linkage
  liquid
    analyze

wowviewer-catalog
  assets
    capture
    export-metadata
  minimap
    export
```

## GUI Panel Tree

```text
WowViewer.App
  Navigator dock
  Inspector dock
  Diagnostics dock
    logs
    perf
    render quality
  World tools
    minimap
    WDL preview
  Conversion workflows
    map converter
    WMO converter
    terrain texture transfer
    VLM export
  PM4 workspace
    alignment
    correlation
    selection and metadata
    report export hooks
  Runtime inspection
    SQL actors
    taxi tools
    liquid and chunk inspectors
```

## Dual-Surface Mapping By Tool Family

| Tool family | GUI surface | CLI surface | Notes |
| --- | --- | --- | --- |
| Terrain conversion | converter dialogs and terrain editing workflows | converter terrain verbs | Same service path, different host UX |
| WMO and model conversion | converter dialogs and preview workflows | converter wmo and model verbs | Same service path, different host UX |
| WDT and DBC inspection | preflight or diagnostics views | inspect wdt and dbc verbs | Most heavy reports stay CLI-first |
| PM4 workspace | dedicated PM4 workspace in the app | inspect pm4 verbs and converter pm4-restore verbs | MdxViewer behavior remains the runtime reference |
| MDX-L archaeology | optional preview or inspector hooks | inspect mdx-l verbs | Better as CLI-first, GUI-selective |
| Texture and minimap export | preview, catalog, and asset workflows | converter or catalog verbs | Some batch workflows stay headless-only |

## Cancellation, Progress, And Reporting Design

### Shared contract

- all long-running operations emit typed progress events
- all operations accept cancellation
- all operations produce a durable result folder or report path when they generate artifacts
- CLI should render progress to console and write reports to disk
- GUI should render the same progress in modal job panes, logs, or background job lists

### Operation result shape

- status
- summary text
- warnings
- produced files
- machine-readable manifest path when applicable

## First Dual-Surface Workflow To Implement

Terrain texture transfer is the best first dual-surface workflow because:

- the viewer already has a seeded dialog flow
- the service boundary is clear and file-based
- it does not depend on the full PM4 or renderer migration
- it proves the new shared-operation pattern without inventing new behavior

Map conversion is the second dual-surface workflow immediately after that.

## Duplication Risks Still Remaining

- if PM4 report logic lives partly in the viewer and partly in CLI code, PM4 will drift again immediately
- if dialog code keeps doing file parsing directly, the new app will still own tool logic instead of calling shared services
- if CLI tools wrap hidden app behavior instead of service calls, automation and validation will remain fragile
- if unstable research verbs are not labeled clearly, users will treat exploratory PM4 output as canonical behavior

## Bottom Line

- the new repo should expose one shared service layer and multiple hosts, not multiple implementations
- the viewer should stay strong where interactivity matters, but heavy inspection and batch work should remain easy to run headlessly
- PM4 must be dual-surface from the start: interactive workspace in the app, inspect verbs in CLI, restore verbs in converter, all on top of Core.PM4

## Validation Status

- planning and documentation only
- no code changes, builds, or runtime validation were performed for this slice