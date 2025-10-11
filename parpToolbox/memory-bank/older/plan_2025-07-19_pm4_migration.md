# PM4 Object Output Migration Plan — 2025-07-19 (Updated 15:06)

## Current Goal
Finish MSCN integration and ensure `parpToolbox` builds/runs via the new `pm4-region` command.

## Key Context
- Objects grouped by **MSUR.IndexCount**.
- Cross-tile vertices remapped with **MscnRemapper**.
- `Pm4RegionLoader` merges 64 × 64 tiles and applies MSCN automatically.
- `Pm4Scene.ExtraChunks` exposes raw chunks; `IIffChunk` now public.

## Open Technical Blockers
1. **Build errors** (fixed now in-code, pending re-build):
   - `Program.cs` instantiated static `Pm4MsurObjectAssembler` & used missing method. Changed to static calls `AssembleObjectsByMsurIndex` + `ExportMsurObjects`.
   - Duplicate `using` directive in `Pm4Adapter` removed.
2. Audit remaining `IReadOnlyList` usages that assume mutability.
3. Add regression tests for region loader & MSCN remap.

## Task Checklist
- [x] Make `IIffChunk` public to fix CS0053.
- [x] Add `ExtraChunks` to `Pm4Scene`; populate in `Pm4Adapter`.
- [x] Re-enable `AttachMscn` in `Pm4RegionLoader`.
- [x] Fix `Program.cs` region branch to use static assembler methods.
- [x] Remove duplicate `using` warnings in `Pm4Adapter`.
- [ ] Run `dotnet build` and resolve any remaining compile errors.
- [ ] Execute `pm4-region` on sample data; verify “Applying MSCN remap” and zero out-of-bounds indices.
- [ ] Audit/exporters/assemblers for any lingering `IReadOnlyList` expectations.
- [ ] Port remaining PM4BatchTool CLI features, ensuring M2 buckets included.
- [ ] Add integration & regression tests.
- [ ] Update docs (`activeContext.md`, `progress.md`) after each milestone.

---


## Context & Progress
* Objects correctly grouped by **MSUR.IndexCount**.
* Cross-tile vertex loss addressed with **MscnRemapper**.
* `Pm4RegionLoader` concatenates sibling tiles; CLI exposes `pm4-region` command.
* Adapters now emit *mutable* `List<T>` collections so scenes can be merged.

## Open Technical Blockers
1. Remaining `IReadOnlyList` usages in exporters/assemblers expect mutability – audit & fix.
2. `Pm4Adapter` does not yet expose **ExtraChunks (MSCN)**; region loader cannot attach remap automatically.
3. Automated tests needed for multi-tile merge, MSCN remap, object assembly parity.

## Task List
- [x] Implement `MscnRemapper` (done)
- [x] Scaffold `Pm4RegionLoader` & CLI integration (done)
- [ ] Refactor `IReadOnlyList` → `List` in remaining consumers
- [ ] Add `ExtraChunks` collection in `Pm4Scene`; populate in adapters
- [ ] Re-enable `AttachMscn(baseScene)` in region loader
- [ ] Compile & run `pm4-region` on sample data; verify zero OOB indices
- [ ] Integrate MPRL + MPRR + MSLK relationships for full object grouping
- [ ] Port remaining PM4BatchTool CLI features (no skip-M2 logic)
- [ ] Add regression & integration tests
- [ ] Update docs (`activeContext.md`, `progress.md`) after each milestone

## Immediate ACT Steps
1. Finish mutability refactor in exporters / assemblers.
2. Add `ExtraChunks` to `Pm4Scene` & populate in `Pm4Adapter` / `Pd4Adapter`.
3. Re-enable MSCN remap in region loader.
4. Build & execute `pm4-region` on test dataset; inspect logs.
