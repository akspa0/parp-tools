# Feature Specification: MdxViewer Migration to wow-viewer

**Feature Branch**: `033-mdxviewer-migration`

**Created**: 2026-05-30

**Status**: Draft

**Input**: MdxViewer currently lives in `gillijimproject_refactor/src/MdxViewer/` and references 4 legacy projects that don't exist in wow-viewer. The goal is to move MdxViewer into `wow-viewer/` so that wow-viewer becomes a self-contained, repo-extractable codebase with its own viewer application. The existing `WowViewer.App` is the current thin host; MdxViewer is the full-featured renderer that needs to transition.

## Problem Statement

wow-viewer cannot be extracted as its own repository because the actual working viewer application (MdxViewer) lives outside it. The current structure forces every wow-viewer development session to cross repo boundaries:

1. MdxViewer references `WowViewer.Core.IO`, `WowViewer.Core.Runtime`, and `WowViewer.Core.PM4` via `..\..\..\wow-viewer\` — already crossing the boundary
2. MdxViewer depends on `WoWMapConverter.Core` (5 namespaces, ~12 files) — not in wow-viewer
3. MdxViewer depends on `MDX-L_Tool` (1 namespace, ~14 files) — not in wow-viewer
4. MdxViewer depends on `DBCD`/`DBCD.IO` (2 namespaces, ~8 files) — not in wow-viewer
5. MdxViewer uses a different copy of `SereniaBLPLib` from wow-viewer's `libs/` directory
6. The existing `WowViewer.App` is a thin host that doesn't have the full MdxViewer rendering capability

## Dependency Audit

### Legacy Dependencies to Decouple

| Legacy Project | Namespaces Used | Files Using It | Migration Strategy |
|---|---|---|---|
| `WoWMapConverter.Core` | `Converters`, `VLM`, `Formats.LichKing`, `Formats.Liquids`, `Diagnostics` | ~12 | Replace with `WowViewer.Core.IO` equivalents where they exist; for missing converters/formats, port the specific classes needed into `WowViewer.Core.IO` |
| `MDX-L_Tool` | `MdxLTool.Formats.Mdx` | ~14 | Port MDX format reader into `WowViewer.Core.IO/Mdx/` (already has partial Mdx readers) |
| `DBCD` / `DBCD.IO` | `DBCD`, `DBCD.Providers` | ~8 | Port DBC reader into `WowViewer.Core.IO/Dbc/` or vendor the library under `wow-viewer/libs/` |
| `SereniaBLPLib` (old copy) | `SereniaBLPLib` | ~7 | Switch to wow-viewer's existing copy at `libs/WoW-Tools/SereniaBLPLib/` |

### Already-In-wow-viewer References

MdxViewer already references these wow-viewer projects (crossing the repo boundary):

- `WowViewer.Core.IO` — file format readers
- `WowViewer.Core.Runtime` — M2 runtime, world visibility
- `WowViewer.Core.PM4` — PM4 library

These references become internal after migration.

### WowViewer.App vs MdxViewer Capability Gap

| Capability | MdxViewer | WowViewer.App | Gap |
|---|---|---|---|
| Terrain rendering | Full (TerrainRenderer, StandardTerrainAdapter, AlphaTerrainAdapter) | Minimal (WorldGpuPreviewRenderer) | **Critical** |
| WMO rendering | Full (WmoRenderer, interior/exterior, liquid) | Minimal (WmoGpuPreviewRenderer) | **Critical** |
| MDX/M2 rendering | Full (ModelRenderer, M2Renderer, particles, ribbons) | Minimal (M2GpuPreviewRenderer, MdxGpuPreviewRenderer) | **Critical** |
| Sky rendering | SkyDomeRenderer + skybox backdrop | SkyRenderer (hardcoded) | **Critical** |
| World session | WorldScene + WorldAssetManager | WowViewerWorldSessionBootstrapper | **Critical** |
| DBC services | LightService, AreaTableService, MapDiscoveryService | None | **Moderate** |
| LIT loading | LitLoader | None | **Moderate** |
| Capture/export | ScreenshotRenderer, AssetExporter, GlbExporter, MapGlbExporter | ValidationCaptureSession | **Moderate** |
| ImGui UI | Full sidebar system | Minimal | **Moderate** |
| Minimap rendering | MinimapRenderer | WorldMinimapRenderer (partial) | **Low** |

## Scope

### In Scope

- Moving MdxViewer source files from `gillijimproject_refactor/src/MdxViewer/` into `wow-viewer/src/viewer/MdxViewer/`
- Decoupling from `WoWMapConverter.Core` — replace imports with `WowViewer.Core.IO` equivalents or port missing classes
- Decoupling from `MDX-L_Tool` — port MDX format reader into `WowViewer.Core.IO/Mdx/`
- Decoupling from `DBCD`/`DBCD.IO` — vendor the library under `wow-viewer/libs/` or port into `WowViewer.Core.IO/Dbc/`
- Switching SereniaBLPLib reference to wow-viewer's existing copy
- Updating `WowViewer.slnx` to include the migrated MdxViewer project
- Updating all `using` statements and project references
- Ensuring the migrated MdxViewer builds and runs from within wow-viewer

### Out of Scope

- Merging MdxViewer and WowViewer.App into a single application (that's a later step)
- Rewriting MdxViewer rendering to use Core.Runtime render pipelines (spec 032 handles that)
- Removing MdxViewer from gillijimproject_refactor (it can stay as a stub/redirect initially)
- Porting WoWMapConverter.Core in its entirety (only port what MdxViewer actually uses)

## Phases

### Phase 1 — Vendor External Libraries (No Code Changes in MdxViewer)

Copy `DBCD`/`DBCD.IO` and `MDX-L_Tool` into `wow-viewer/libs/` so MdxViewer can reference them without crossing the repo boundary. Switch SereniaBLPLib reference to wow-viewer's copy.

### Phase 2 — Move MdxViewer Source Files

Physically move the MdxViewer project directory into `wow-viewer/src/viewer/MdxViewer/`. Update .csproj to reference wow-viewer-internal copies of everything. Add to WowViewer.slnx.

### Phase 3 — Decouple WoWMapConverter.Core Imports

Replace `WoWMapConverter.Core` usages with `WowViewer.Core.IO` equivalents. For classes that don't have equivalents yet, port the specific converter/format classes into WowViewer.Core.IO. This is the hardest phase.

### Phase 4 — Build and Run Verification

Ensure the migrated MdxViewer builds from within wow-viewer and produces a working viewer. Validate against staged client data.

## Success Criteria

- `dotnet build wow-viewer/WowViewer.slnx` includes MdxViewer and succeeds
- MdxViewer can be run from within the wow-viewer directory structure
- No source file in `wow-viewer/` references a path outside `wow-viewer/` (except game client paths)
- The gillijimproject_refactor copy of MdxViewer can be safely archived or stubbed
