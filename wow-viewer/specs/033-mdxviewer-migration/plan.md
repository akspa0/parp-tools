# Implementation Plan: MdxViewer Migration to wow-viewer

**Branch**: `033-mdxviewer-migration` | **Date**: 2026-05-30 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `wow-viewer/specs/033-mdxviewer-migration/spec.md`

## Summary

Move the MdxViewer application from `gillijimproject_refactor/src/MdxViewer/` into `wow-viewer/src/viewer/MdxViewer/` so that wow-viewer becomes a self-contained, repo-extractable codebase. MdxViewer currently depends on four legacy projects: `WoWMapConverter.Core`, `MDX-L_Tool`, `DBCD`, and `SereniaBLPLib`. After migration, DBCD and SereniaBLPLib already exist in wow-viewer/libs/ and only MDX-L_Tool needs vendoring. All WoWMapConverter.Core usages are ported to WowViewer.Core or WowViewer.Core.IO equivalents. The existing `WowViewer.App` remains alongside MdxViewer.

## Project Structure

### Existing Libraries (already in wow-viewer — no action needed)

| Library | Location | Status |
|---------|----------|--------|
| DBCD | `wow-viewer/libs/wowdev/DBCD/` | Already present — MdxViewer will reference this |
| SereniaBLPLib | `wow-viewer/libs/WoW-Tools/SereniaBLPLib/` | Already present — MdxViewer will reference this |
| SharpGLTF.Toolkit | NuGet package | Already referenced in MdxViewer.csproj |

### Libraries to Vendor

| Library | Source | Destination |
|---------|--------|-------------|
| MDX-L_Tool | `gillijimproject_refactor/src/MDX-L_Tool/` | `wow-viewer/libs/WoW-Tools/MDX-L_Tool/` |

### Migration Map

**Before migration:**
```text
gillijimproject_refactor/src/
├── MdxViewer/               # Source — TO BE MOVED
│   └── MdxViewer.csproj     # References gillijimproject_refactor paths (BROKEN after move)
├── MDX-L_Tool/              # TO BE VENDORED
├── WoWMapConverter/         # Some classes TO BE PORTED
│   └── WoWMapConverter.Core/
└── lib/
    ├── SereniaBLPLib/       # Already in wow-viewer — DO NOT USE
    └── wow.tools.local/
        └── DBCD/            # Already in wow-viewer (wowdev/) — DO NOT USE

wow-viewer/
├── libs/
│   ├── wowdev/
│   │   └── DBCD/            # CORRECT DBCD location
│   ├── WoW-Tools/
│   │   ├── SereniaBLPLib/   # CORRECT SereniaBLPLib location
│   │   └── [MDX-L_Tool missing]  # <-- needs vendor
│   └── SharpGLTF.Toolkit/   # NuGet — already OK
└── src/viewer/WowViewer.App/ # Existing thin host

wow-viewer/WowViewer.slnx    # Does NOT include MdxViewer yet
```

**After migration:**
```text
wow-viewer/
├── libs/
│   ├── wowdev/DBCD/         # Existing — used by WowViewer.Core.IO
│   └── WoW-Tools/
│       ├── SereniaBLPLib/   # Existing — used by renderer
│       └── MDX-L_Tool/      # NEWLY VENDORED
├── src/
│   ├── core/
│   │   ├── WowViewer.Core/
│   │   ├── WowViewer.Core.IO/
│   │   ├── WowViewer.Core.Runtime/
│   │   ├── WowViewer.Core.Renderer/
│   │   └── WowViewer.Core.PM4/
│   └── viewer/
│       ├── WowViewer.App/   # Existing thin host
│       └── MdxViewer/       # NEWLY MOVED — full WinForms viewer
│           ├── MdxViewer.csproj
│           ├── Rendering/
│           ├── Terrain/
│           ├── M2/
│           ├── World/
│           └── ViewerApp*.cs
└── WowViewer.slnx           # UPDATED — includes MdxViewer
```

## Implementation Phases

### Phase 1 — Verify Existing Libraries and Vendor MDX-L_Tool

**Goal**: Confirm DBCD and SereniaBLPLib are already in the right wow-viewer/libs/ location. Vendor MDX-L_Tool. Update MdxViewer.csproj to reference the correct library paths.

**Dependencies**: None.

**Approach**:
1. Verify `wow-viewer/libs/wowdev/DBCD/DBCD/DBCD.csproj` exists and matches the version MdxViewer needs
2. Verify `wow-viewer/libs/WoW-Tools/SereniaBLPLib/SereniaBLPLib/SereniaBLPLib.csproj` exists
3. Copy `gillijimproject_refactor/src/MDX-L_Tool/` to `wow-viewer/libs/WoW-Tools/MDX-L_Tool/`
4. Update MdxViewer.csproj to use wow-viewer paths for all libraries
5. Verify MdxViewer builds in its original gillijimproject_refactor location with updated paths

**Steps** (max 8):
1. Verify `wow-viewer/libs/wowdev/DBCD/DBCD/DBCD.csproj` and `wow-viewer/libs/wowdev/DBCD/DBCD.IO/DBCD.IO.csproj` exist
2. Verify `wow-viewer/libs/WoW-Tools/SereniaBLPLib/SereniaBLPLib/SereniaBLPLib.csproj` exists
3. Copy `gillijimproject_refactor/src/MDX-L_Tool/` to `wow-viewer/libs/WoW-Tools/MDX-L_Tool/`
4. Add `wow-viewer/libs/WoW-Tools/MDX-L_Tool/MDX-L_Tool.csproj` to `wow-viewer/WowViewer.slnx`
5. Read `gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj` and update library paths:
   - Change DBCD: `..\..\lib\wow.tools.local\DBCD\DBCD\DBCD.csproj` → `..\..\..\wow-viewer\libs\wowdev\DBCD\DBCD\DBCD.csproj`
   - Change DBCD.IO: same pattern
   - Change SereniaBLPLib: `..\..\lib\SereniaBLPLib\...` → `..\..\..\wow-viewer\libs\WoW-Tools\SereniaBLPLib\SereniaBLPLib\SereniaBLPLib.csproj`
   - Change MDX-L_Tool: `..\MDX-L_Tool\MDX-L_Tool.csproj` → `..\..\..\wow-viewer\libs\WoW-Tools\MDX-L_Tool\MDX-L_Tool.csproj`
6. Verify MdxViewer.csproj now has zero references to gillijimproject_refactor paths
7. Run `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` — verify build succeeds with new paths

---

### Phase 2 — Move MdxViewer Source Files

**Goal**: Physically move MdxViewer project into `wow-viewer/src/viewer/MdxViewer/`. Update project references. Add to solution.

**Dependencies**: Phase 1 (all library references inside wow-viewer/libs/).

**Approach**:
1. Copy `gillijimproject_refactor/src/MdxViewer/` to `wow-viewer/src/viewer/MdxViewer/` (preserving directory structure)
2. Update MdxViewer.csproj project references — all should already point to wow-viewer paths from Phase 1, but paths must be adjusted since the file moved
3. Add to WowViewer.slnx
4. Build and verify

**Steps** (max 8):
1. Copy `gillijimproject_refactor/src/MdxViewer/` to `wow-viewer/src/viewer/MdxViewer/` preserving timestamps
2. In the copied `wow-viewer/src/viewer/MdxViewer/MdxViewer.csproj`, update relative paths for WowViewer.Core.IO/Runtime/PM4 project references (paths will need `..\..\..\` prefix adjustment since the file moved deeper)
3. Verify DBCD, DBCD.IO, SereniaBLPLib, MDX-L_Tool paths still point to `wow-viewer/libs/`
4. Add `wow-viewer/src/viewer/MdxViewer/MdxViewer.csproj` to `wow-viewer/WowViewer.slnx`
5. Add `wow-viewer/src/viewer/MdxViewer/MdxViewer.CrossPlatform.csproj` to `wow-viewer/WowViewer.slnx` if it exists
6. Run `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
7. Fix any compilation errors from the move
8. Verify MdxViewer launches: `dotnet run --project i:/parp/parp-tools/wow-viewer/src/viewer/MdxViewer/MdxViewer.csproj -c Debug`

---

### Phase 3 — Decouple WoWMapConverter.Core Imports

**Goal**: Replace all `WoWMapConverter.Core` usages in MdxViewer with WowViewer.Core or WowViewer.Core.IO equivalents. Make MdxViewer fully independent of gillijimproject_refactor.

**Dependencies**: Phase 2 (MdxViewer files are in wow-viewer but reference gillijimproject_refactor projects).

**Approach**:
1. Audit all `WoWMapConverter.Core.*` usages in MdxViewer
2. For each usage, either use existing WowViewer.Core.IO equivalent or port the specific class
3. Update all `using` statements and remove the WoWMapConverter.Core project reference

**Steps** (max 10):
1. Grep `wow-viewer/src/viewer/MdxViewer/` for `using WoWMapConverter` — list all namespaces and classes
2. Determine for each: existing WowViewer.Core.IO equivalent, or needs porting
3. Port `WoWMapConverter.Core.Converters` (AssetExporter, GlbExporter, MapGlbExporter) to `WowViewer.Core.IO/Converters/`
4. Port VLM reader to `WowViewer.Core.IO/Maps/VlmReader.cs`
5. Check if `StandardTerrainAdapter` can use `WowViewer.Core.IO`'s LkAdtReader instead of WoWMapConverter.Core.Formats.LichKing
6. Check if liquid classes can use `WowViewer.Core.IO`'s AdtLiquidReader/AdtMclqReader
7. Port Diagnostics utilities to `WowViewer.Core/Diagnostics/`
8. Update all MdxViewer `using` statements to use new namespaces
9. Remove `WoWMapConverter.Core.csproj` reference from MdxViewer.csproj
10. Build and fix all compilation errors

---

### Phase 4 — Build and Run Verification

**Goal**: Ensure migrated MdxViewer builds from WowViewer.slnx and runs against staged client data.

**Dependencies**: Phases 1–3 complete.

**Steps** (max 6):
1. Full `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
2. Launch MdxViewer and load a staged client map (`I:\parp\parp-tools\output\tmp\wowarchive-clients\`)
3. Verify terrain, WMO, sky dome render without errors
4. Grep for any remaining `gillijimproject_refactor`, `WoWMapConverter`, `H:\CLIENTS` references
5. Stub `gillijimproject_refactor/src/MdxViewer/` with README pointing to new location
6. Update memory bank (`gillijimproject_refactor/memory-bank/progress.md`) with migration status

## Complexity Tracking

| Decision | Why | Simpler Alternative Rejected |
|----------|-----|---------------------------|
| MDX-L_Tool vendoring | MDX format parsing is complex and has internal dependencies on ImageSharp and SereniaBLPLib. Vendoring preserves working code. | Rewrite MDX reader in WowViewer.Core.IO — rejected: would take weeks and introduce bugs. |
| Converter porting | MdxViewer's AssetExporter/GlbExporter are small utility classes that call existing wow-viewer readers internally. Porting is straightforward. | Leave them referencing WoWMapConverter — rejected: violates repo-independence Rule 4. |
| VLM porting | VLM is terrain vertex lighting data format. Small reader class, easily ported. | Leave referencing WoWMapConverter — rejected: violates Rule 4. |