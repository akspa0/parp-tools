# WoWRollback Consolidation - Master Plan 🎯

**Project**: Consolidate AlphaWDTAnalysisTool + DBCTool.V2 into WoWRollback  
**Status**: Ready for Implementation  
**Date**: 2025-10-04  
**Estimated Time**: 3-4 weeks

---

## 📋 Quick Reference

### Current State
- **3 separate tools**: AlphaWDTAnalysisTool, DBCTool.V2, WoWRollback
- **Complex orchestration**: 573-line PowerShell script
- **Performance**: 52 minutes for large maps
- **CPU usage**: 2-8% (sequential processing)

### Target State
- **1 unified tool**: WoWRollback with all functionality
- **Simple command**: Single CLI command
- **Performance**: 7-8 minutes (6.5x faster!)
- **CPU usage**: 60-80% (multi-threaded)

---

## 🎯 Implementation Phases

### Phase 1: Format Readers (Week 1)
Migrate file format reading into WoWRollback.Core

**New Files**:
```
WoWRollback.Core/Formats/
├── Alpha/
│   ├── WdtAlphaReader.cs        ← Read Alpha WDT
│   ├── AdtAlphaReader.cs        ← Read Alpha ADT
│   └── AdtAlphaConverter.cs     ← Convert Alpha → LK
├── Lk/
│   └── AdtLkWriter.cs           ← Write LK ADT files
└── Dbc/
    ├── DbcReader.cs             ← Generic DBC reading
    └── AreaTableReader.cs       ← Parse AreaTable
```

**Checklist**:
- [ ] Migrate WDT/ADT reading from AlphaWDTAnalysisTool
- [ ] Create LK ADT writer (currently scattered)
- [ ] Migrate DBC reading from DBCTool.V2
- [ ] Add ListfileService for shared listfile loading
- [ ] Write unit tests for each reader

---

### Phase 2: Multi-Threaded Processing (Week 2)
Implement parallel conversion with `Parallel.ForEachAsync`

**New Files**:
```
WoWRollback.Core/Processing/
├── MapConverter.cs          ← Orchestrate Alpha → LK (multi-threaded)
├── TerrainExtractor.cs      ← Extract terrain CSV (parallel)
├── ShadowExtractor.cs       ← Extract shadow CSV (parallel)
└── AreaTableProcessor.cs    ← Match Alpha → LK AreaIDs
```

**Key Pattern**:
```csharp
await Parallel.ForEachAsync(wdtInfo.AdtTiles, 
    new ParallelOptions { MaxDegreeOfParallelism = 8 },
    async (adtNum, ct) =>
{
    // Convert ADT in parallel
    var alphaAdt = AdtAlphaReader.ReadAdt(wdtPath, adtNum, offset);
    var lkAdt = AdtAlphaConverter.ConvertToLk(alphaAdt);
    await AdtLkWriter.WriteAdtAsync(outputPath, lkAdt);
});
```

**Checklist**:
- [ ] Create MapConverter with parallel ADT processing
- [ ] Create TerrainExtractor with thread-safe CSV writing
- [ ] Create ShadowExtractor (similar to TerrainExtractor)
- [ ] Create AreaTableProcessor with smart matching
- [ ] Add progress reporting (Interlocked.Increment)

---

### Phase 3: CLI Integration (Week 3)
Expose functionality through beautiful CLI commands

**New Commands**:
```
wowrollback convert-map <wdt> --output <dir> --threads 8
wowrollback extract-terrain --input <dir> --output <csv>
wowrollback process-areatables --alpha-dbc <path> --lk-dbc <path>
wowrollback compare-versions --alpha-root <dir> --versions <list> --maps <list>
```

**New Files**:
```
WoWRollback.Cli/Commands/
├── ConvertMapCommand.cs
├── ExtractTerrainCommand.cs
├── ProcessAreaTableCommand.cs
└── CompareVersionsCommand.cs  ← Enhanced
```

**Checklist**:
- [ ] Add Spectre.Console package
- [ ] Create ConvertMapCommand with progress bar
- [ ] Create ExtractTerrainCommand
- [ ] Create ProcessAreaTableCommand
- [ ] Enhance CompareVersionsCommand (auto-convert if --alpha-root)
- [ ] Update rebuild-and-regenerate.ps1 (simplify)

---

### Phase 4: Testing (Week 4)
Ensure correctness and performance

**Checklist**:
- [ ] Unit tests (90%+ coverage)
- [ ] Integration tests (full pipeline)
- [ ] Benchmark sequential vs parallel
- [ ] Test in 3.3.5a client (converted maps work)
- [ ] Memory profiling (no leaks)

---

### Phase 5: Cleanup (Week 5)
Archive old tools and polish

**Checklist**:
- [ ] Move AlphaWDTAnalysisTool to `_archived/`
- [ ] Move DBCTool.V2 to `_archived/`
- [ ] Update documentation
- [ ] Polish CLI (help text, colors)
- [ ] Release notes

### Phase 6: 3D Export & Visualization (Weeks 6-10) ⭐ NEW
Consolidate ADTPrefabTool 3D export functionality

**See**: `docs/planning/PHASE_6_3D_EXPORT.md` for full details

**Checklist**:
- [ ] Migrate terrain mesh building from ADTPrefabTool
- [ ] Implement GLB/glTF export (SharpGLTF.Toolkit)
- [ ] Multi-threaded tile export (parallel GLB generation)
- [ ] Texture extraction and embedding
- [ ] Prefab pattern mining (terrain pattern detection)
- [ ] CLI commands (export-3d, mine-prefabs)

**Benefits**:
- ✨ Export tiles as GLB (Unity, Unreal, Blender, web viewers)
- ✨ Embedded textures (single-file assets)
- ✨ Pattern mining (recurring terrain analysis)
- ✨ Multi-threaded (6-7x faster than ADTPrefabTool)
- ✨ Modern PBR materials

---

## 🔑 Key Implementation Details

### Multi-Threading Patterns

#### Thread-Safe Collection
```csharp
var results = new ConcurrentBag<Result>();
await Parallel.ForEachAsync(..., (item, ct) => 
{
    results.Add(ProcessItem(item));
});
```

#### Progress Reporting
```csharp
private int _progress = 0;
var current = Interlocked.Increment(ref _progress);
if (current % 10 == 0)
    Console.WriteLine($"{current}/{total}");
```

#### CSV Writing (Batch)
```csharp
var rows = new ConcurrentBag<string>();
// Collect in parallel...
await File.AppendAllLinesAsync(csvPath, rows); // Single write
```

### Dependencies
- **Keep**: Warcraft.NET, GillijimProject.WowFiles.Alpha, DBCD.IO
- **Add**: Spectre.Console, Microsoft.Extensions.Logging
- **Remove**: AlphaWDTAnalysisTool refs, DBCTool.V2 refs (after migration)

---

## 📊 Expected Performance

| Map | Tiles | Before | After (8 threads) | Speedup |
|-----|-------|--------|-------------------|---------|
| DeadminesInstance | 10 | 2 min | 20 sec | 6x |
| Shadowfang | 15 | 3 min | 30 sec | 6x |
| Azeroth | 128 | 50 min | 7 min | 7x |
| Kalimdor | 140 | 55 min | 8 min | 7x |

**Total workflow**: 52 min → 7-8 min ✅

---

## 🚀 Quick Start (After Implementation)

### Old Workflow (3 tools, 52 min)
```powershell
.\rebuild-and-regenerate.ps1 -Maps Azeroth -Versions 0.5.3.3368 -RefreshCache
```

### New Workflow (1 tool, 8 min)
```powershell
dotnet run --project WoWRollback.Cli -- compare-versions \
  --alpha-root ../test_data \
  --versions 0.5.3.3368 \
  --maps Azeroth \
  --threads 8 \
  --viewer-report \
  --serve
```

---

## 📝 Session Checkpoints

Use these to resume work in fresh chat sessions:

### Starting Phase 1?
**Context**: "Implementing Phase 1 of MASTER_PLAN.md - migrating format readers from AlphaWDTAnalysisTool and DBCTool.V2 into WoWRollback.Core/Formats/. Starting with WdtAlphaReader.cs."

### Starting Phase 2?
**Context**: "Implementing Phase 2 of MASTER_PLAN.md - creating multi-threaded MapConverter using Parallel.ForEachAsync. Format readers from Phase 1 are complete."

### Starting Phase 3?
**Context**: "Implementing Phase 3 of MASTER_PLAN.md - adding CLI commands with Spectre.Console. MapConverter and extractors from Phase 2 are working and tested."

### Starting Phase 4?
**Context**: "Implementing Phase 4 of MASTER_PLAN.md - comprehensive testing. All features implemented, need unit/integration tests and benchmarks."

### Starting Phase 5?
**Context**: "Implementing Phase 5 of MASTER_PLAN.md - cleanup and documentation. All features tested and working, ready to archive old tools."

---

## 📚 Related Documentation

- `05-wowrollback-consolidation.md` - Detailed architecture and design
- `ALPHAWDT_MULTITHREADING.md` - Multi-threading strategy and benchmarks
- `SMART_CACHE_OPTIMIZATION.md` - Cache validation improvements
- `SHADOW_MAPS_COMPLETE.md` - Shadow map feature implementation

---

## ✅ Success Criteria

- [ ] Single tool replaces 3 tools
- [ ] 6-7x performance improvement
- [ ] 90%+ test coverage
- [ ] Converted maps work in 3.3.5a client
- [ ] Beautiful CLI with progress bars
- [ ] Documentation updated
- [ ] Old tools archived (not deleted)

**Ready to begin implementation!** 🚀
