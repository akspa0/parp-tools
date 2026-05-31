# Task Breakdown: MdxViewer Migration to wow-viewer

**Branch**: `033-mdxviewer-migration` | **Date**: 2026-05-30 | **Spec/Plan**: `spec.md` / `plan.md`

Each phase must complete and validate before the next phase begins.

---

## Phase 1 — Verify Existing Libraries and Vendor MDX-L_Tool

**Dependencies**: None.

**Owner**: Phase 1 (self-contained, no blockers)

---

### Phase 1.1 — Verify existing DBCD and SereniaBLPLib

- [ ] **T1.1.1**: Verify `wow-viewer/libs/wowdev/DBCD/DBCD/DBCD.csproj` exists
- [ ] **T1.1.2**: Verify `wow-viewer/libs/wowdev/DBCD/DBCD.IO/DBCD.IO.csproj` exists
- [ ] **T1.1.3**: Verify `wow-viewer/libs/WoW-Tools/SereniaBLPLib/SereniaBLPLib/SereniaBLPLib.csproj` exists
- [ ] **T1.1.4**: Verify SharpGLTF.Toolkit is already a NuGet PackageReference in MdxViewer.csproj (no action needed)

---

### Phase 1.2 — Vendor MDX-L_Tool

- [ ] **T1.2.1**: Copy `gillijimproject_refactor/src/MDX-L_Tool/` to `wow-viewer/libs/WoW-Tools/MDX-L_Tool/` preserving timestamps
- [ ] **T1.2.2**: Verify `wow-viewer/libs/WoW-Tools/MDX-L_Tool/MDX-L_Tool.csproj` exists
- [ ] **T1.2.3**: Add `wow-viewer/libs/WoW-Tools/MDX-L_Tool/MDX-L_Tool.csproj` to `wow-viewer/WowViewer.slnx`
- [ ] **T1.2.4**: Verify MDX-L_Tool has no references to WoWMapConverter projects (it should be standalone)

---

### Phase 1.3 — Update MdxViewer.csproj library paths

- [ ] **T1.3.1**: Read `gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj`
- [ ] **T1.3.2**: Change DBCD path from `..\..\lib\wow.tools.local\DBCD\DBCD\DBCD.csproj` to `..\..\..\wow-viewer\libs\wowdev\DBCD\DBCD\DBCD.csproj`
- [ ] **T1.3.3**: Change DBCD.IO path similarly
- [ ] **T1.3.4**: Change SereniaBLPLib path from `..\..\lib\SereniaBLPLib\...` to `..\..\..\wow-viewer\libs\WoW-Tools\SereniaBLPLib\SereniaBLPLib\SereniaBLPLib.csproj`
- [ ] **T1.3.5**: Change MDX-L_Tool path from `..\MDX-L_Tool\MDX-L_Tool.csproj` to `..\..\..\wow-viewer\libs\WoW-Tools\MDX-L_Tool\MDX-L_Tool.csproj`
- [ ] **T1.3.6**: Verify all WowViewer.Core.IO/Runtime/PM4 project references point to `..\..\..\wow-viewer\src\core\...` (these should already be correct)

---

### Phase 1.4 — Build MdxViewer in original location with updated paths

- [ ] **T1.4.1**: Run `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- [ ] **T1.4.2**: Verify build succeeds with updated library paths pointing into wow-viewer/libs/
- [ ] **T1.4.3**: If build fails, diagnose and fix path or dependency issues in Phase 1

**Phase 1 Exit Criteria**: MdxViewer.csproj has zero references to `gillijimproject_refactor` library paths. All libraries point to `wow-viewer/libs/` or NuGet. Build succeeds in original location.

---

## Phase 2 — Move MdxViewer Source Files

**Dependencies**: Phase 1 (all library references point into wow-viewer/libs/).

---

### Phase 2.1 — Copy MdxViewer into wow-viewer

- [ ] **T2.1.1**: Copy `gillijimproject_refactor/src/MdxViewer/` to `wow-viewer/src/viewer/MdxViewer/` preserving file timestamps
- [ ] **T2.1.2**: Verify ~300 files copied including all subdirectories (Rendering/, Terrain/, M2/, World/, UI/, Diagnostics/, Export/, Shaders/)
- [ ] **T2.1.3**: Verify `ViewerApp.cs` and `ViewerApp.Designer.cs` are present
- [ ] **T2.1.4**: Verify shader files (.glsl, .vert, .frag, .shader) copied if present

---

### Phase 2.2 — Update project references in copied MdxViewer.csproj

- [ ] **T2.2.1**: Read `wow-viewer/src/viewer/MdxViewer/MdxViewer.csproj` — find all ProjectReference elements
- [ ] **T2.2.2**: All paths in the copied file still say `..\..\..\wow-viewer\...` which is now wrong since MdxViewer is INSIDE wow-viewer — update to `..\..\..\` relative paths
- [ ] **T2.2.3**: Verify DBCD, DBCD.IO, SereniaBLPLib, MDX-L_Tool paths point to `wow-viewer/libs/`
- [ ] **T2.2.4**: Verify WowViewer.Core.IO/Runtime/PM4 project reference paths are correct for the new location
- [ ] **T2.2.5**: Verify `PackageReference` elements for Silk.NET and ImageSharp are unchanged (NuGet packages, not affected by move)

---

### Phase 2.3 — Add MdxViewer to WowViewer.slnx

- [ ] **T2.3.1**: Read `wow-viewer/WowViewer.slnx` — understand its SDK-style solution format
- [ ] **T2.3.2**: Add `wow-viewer/src/viewer/MdxViewer/MdxViewer.csproj` to `/src/viewer/` folder in WowViewer.slnx
- [ ] **T2.3.3**: Add `wow-viewer/src/viewer/MdxViewer/MdxViewer.CrossPlatform.csproj` if it exists and is needed
- [ ] **T2.3.4**: Verify solution file is valid XML with correct project GUIDs

---

### Phase 2.4 — Build wow-viewer solution including MdxViewer

- [ ] **T2.4.1**: Run `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
- [ ] **T2.4.2**: Fix compilation errors (likely namespace conflicts or path issues after move)
- [ ] **T2.4.3**: Verify MdxViewer builds without errors

---

### Phase 2.5 — Launch MdxViewer from wow-viewer location

- [ ] **T2.5.1**: `dotnet run --project i:/parp/parp-tools/wow-viewer/src/viewer/MdxViewer/MdxViewer.csproj -c Debug`
- [ ] **T2.5.2**: Verify MdxViewer launches without crashing (shows main window)
- [ ] **T2.5.3**: Load a staged client map — verify no file-not-found errors on startup

---

### Phase 2.6 — Validate rendering against staged client

- [ ] **T2.6.1**: Open `development/Development_0_0` using staged client at `I:\parp\parp-tools\output\tmp\wowarchive-clients\`
- [ ] **T2.6.2**: Verify terrain tile renders (ADT chunks visible)
- [ ] **T2.6.3**: Verify sky dome renders
- [ ] **T2.6.4**: Verify no missing DLL errors or runtime type resolution errors

**Phase 2 Exit Criteria**: MdxViewer lives at `wow-viewer/src/viewer/MdxViewer/`, builds from `wow-viewer/WowViewer.slnx`, and launches successfully with terrain and sky rendering.

---

## Phase 3 — Decouple WoWMapConverter.Core Imports

**Dependencies**: Phase 2 (MdxViewer files are in wow-viewer but still reference WoWMapConverter).

---

### Phase 3.1 — Audit WoWMapConverter.Core usages

- [ ] **T3.1.1**: Grep `wow-viewer/src/viewer/MdxViewer/` for `using WoWMapConverter` — list all namespaces
- [ ] **T3.1.2**: List specific classes used per namespace and files that use them
- [ ] **T3.1.3**: For each class, determine: existing WowViewer.Core.IO equivalent, or needs porting

---

### Phase 3.2 — Port WoWMapConverter.Core.Converters

- [ ] **T3.2.1**: Find `AssetExporter`, `GlbExporter`, `MapGlbExporter` in gillijimproject_refactor
- [ ] **T3.2.2**: Copy to `wow-viewer/src/core/WowViewer.Core.IO/Converters/`
- [ ] **T3.2.3**: Update namespaces: `WoWMapConverter.Core.Converters` → `WowViewer.Core.IO.Converters`
- [ ] **T3.2.4**: Fix any internal references to WoWMapConverter namespaces within these classes
- [ ] **T3.2.5**: Update MdxViewer files using these converters

---

### Phase 3.3 — Port VLM reader

- [ ] **T3.3.1**: Find VLM classes in `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/VLM/`
- [ ] **T3.3.2**: Create `wow-viewer/src/core/WowViewer.Core.IO/Maps/VlmReader.cs`
- [ ] **T3.3.3**: Update MdxViewer files that use VLM classes

---

### Phase 3.4 — Replace Formats.LichKing usages

- [ ] **T3.4.1**: Check what `StandardTerrainAdapter` uses from `WoWMapConverter.Core.Formats.LichKing`
- [ ] **T3.4.2**: Determine if `WowViewer.Core.IO`'s LkAdtReader covers the needed functionality
- [ ] **T3.4.3**: If yes, update MdxViewer to use WowViewer.Core.IO instead
- [ ] **T3.4.4**: If no, port the specific needed class to `WowViewer.Core.IO/Lk/`

---

### Phase 3.5 — Replace Formats.Liquids usages

- [ ] **T3.5.1**: Check what `StandardTerrainAdapter` and `WlLiquidLoader` use from `WoWMapConverter.Core.Formats.Liquids`
- [ ] **T3.5.2**: Determine if `WowViewer.Core.IO`'s AdtLiquidReader/AdtMclqReader cover the needed functionality
- [ ] **T3.5.3**: If yes, update MdxViewer to use WowViewer.Core.IO instead
- [ ] **T3.5.4**: If no, port the specific needed class

---

### Phase 3.6 — Port Diagnostics utilities

- [ ] **T3.6.1**: Find `WoWMapConverter.Core.Diagnostics` classes
- [ ] **T3.6.2**: Create `wow-viewer/src/core/WowViewer.Core/Diagnostics/` with equivalents
- [ ] **T3.6.3**: Update MdxViewer files that use Diagnostics

---

### Phase 3.7 — Remove WoWMapConverter.Core reference

- [ ] **T3.7.1**: Remove the `ProjectReference` to `WoWMapConverter.Core.csproj` from MdxViewer.csproj
- [ ] **T3.7.2**: Grep for any remaining `WoWMapConverter` references in MdxViewer source — fix all
- [ ] **T3.7.3**: Verify zero `WoWMapConverter` strings remain in MdxViewer source

---

### Phase 3.8 — Build verification after decoupling

- [ ] **T3.8.1**: Run `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
- [ ] **T3.8.2**: Fix compilation errors from namespace changes
- [ ] **T3.8.3**: Verify MdxViewer still launches and renders

**Phase 3 Exit Criteria**: MdxViewer.csproj has zero references to any `.csproj` outside `wow-viewer/`. All `WoWMapConverter` namespaces replaced with `WowViewer.*` equivalents.

---

## Phase 4 — Build and Run Verification

**Dependencies**: Phases 1, 2, 3 complete.

---

### Phase 4.1 — Full solution build

- [ ] **T4.1.1**: Run `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` — full solution
- [ ] **T4.1.2**: Verify zero errors on MdxViewer projects

---

### Phase 4.2 — Runtime validation

- [ ] **T4.2.1**: Launch MdxViewer from wow-viewer
- [ ] **T4.2.2**: Open `development/Development_0_0` using staged client at `I:\parp\parp-tools\output\tmp\wowarchive-clients\`
- [ ] **T4.2.3**: Verify terrain renders, sky dome renders, no file I/O errors

---

### Phase 4.3 — Dependency audit

- [ ] **T4.3.1**: Grep `wow-viewer/src/viewer/MdxViewer/` for `gillijimproject_refactor` — should be zero
- [ ] **T4.3.2**: Grep for `H:\CLIENTS` — should be zero
- [ ] **T4.3.3**: Grep for `WoWMapConverter` — should be zero
- [ ] **T4.3.4**: Grep MdxViewer.csproj for any paths outside `wow-viewer/` — should be zero

---

### Phase 4.4 — Stub old location

- [ ] **T4.4.1**: Write README in `gillijimproject_refactor/src/MdxViewer/` saying "Migrated to wow-viewer/src/viewer/MdxViewer/"
- [ ] **T4.4.2**: Update memory bank `gillijimproject_refactor/memory-bank/progress.md`

**Phase 4 Exit Criteria**: Full solution builds, MdxViewer runs, zero external references, old location stubbed.

---

## Phase Summary

| Phase | Tasks | Focus |
|-------|-------|-------|
| Phase 1 | 11 | Verify existing libs + vendor MDX-L_Tool + update MdxViewer.csproj paths |
| Phase 2 | 12 | Move source files, fix relative paths, add to solution, build, launch |
| Phase 3 | 17 | Port/replace all WoWMapConverter.Core usages with WowViewer.* equivalents |
| Phase 4 | 9 | Full validation + dependency audit + stub old location |
| **Total** | **49** | |

## Validation Commands

```bash
# Phase 1: Build MdxViewer with updated wow-viewer paths
dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug

# Phase 2: Build MdxViewer inside wow-viewer
dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug

# Phase 3: Verify no WoWMapConverter references remain
grep -r "WoWMapConverter" i:/parp/parp-tools/wow-viewer/src/viewer/MdxViewer/

# Phase 4: Run MdxViewer from wow-viewer
dotnet run --project i:/parp/parp-tools/wow-viewer/src/viewer/MdxViewer/MdxViewer.csproj -c Debug
```

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| MDX-L_Tool has hidden dependencies on WoWMapConverter | Medium | High | Audit MDX-L_Tool's `using` statements before vendoring; if it references WoWMapConverter, port those too |
| MdxViewer has implicit assumptions about current-directory for shader assets | Low | Medium | Check shader file loading after move; fix paths if needed |
| WinForms Designer files have hardcoded path strings | Medium | Low | Search Designer.cs for path strings before Phase 2 |
| DBCD version in wowdev vs wow.tools.local differs | Low | Medium | Verify wowdev DBCD has same public API before switching — check DBCD.csproj version metadata |
