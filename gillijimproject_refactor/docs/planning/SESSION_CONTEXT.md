# Session Context - Quick Resume Guide 🔄

Use this file to quickly resume work in fresh chat sessions.

---

## 📍 Current Status

**Phase**: Planning Complete ✅  
**Next Action**: Begin Phase 1 implementation  
**Branch**: `wowrollback-refactor`

---

## 🎯 Context for Next Session

### One-Liner Summary
"Consolidating AlphaWDTAnalysisTool + DBCTool.V2 into WoWRollback with multi-threading for 6.5x performance boost. See docs/planning/MASTER_PLAN.md for full details."

### Key Decisions Made
1. ✅ Consolidate 3 tools into WoWRollback (approved)
2. ✅ Use `Parallel.ForEachAsync` for multi-threading
3. ✅ Target net9.0 framework
4. ✅ Use Spectre.Console for beautiful CLI
5. ✅ Archive old tools (don't delete, for safety)

### Recent Fixes (2025-10-04)
- ✅ Fixed shadow map path bug (`/overlays/null/null` → proper version/map)
- ✅ Added `--count-tiles` to AlphaWdtAnalyzer for WDT validation
- ✅ Updated overlayManager.js to set shadow layer context dynamically

### Current Code State
- ✅ Shadow maps working in viewer
- ✅ Terrain overlays working
- ✅ Smart cache optimization implemented
- ✅ WDT tile counting via tool (not PowerShell)
- ⏳ AlphaWDTAnalysisTool still separate (to be consolidated)
- ⏳ DBCTool.V2 still separate (to be consolidated)

---

## 📋 What to Say When Resuming

### Starting Fresh Implementation
```
I'm continuing the WoWRollback consolidation project. 
See docs/planning/MASTER_PLAN.md for the full plan.

I'm starting Phase 1: migrating format readers from AlphaWDTAnalysisTool 
into WoWRollback.Core/Formats/Alpha/.

First task: Create WdtAlphaReader.cs to read Alpha WDT files.
```

### Resuming Mid-Phase
```
Continuing Phase [X] of MASTER_PLAN.md. 

Already complete:
- [List completed tasks]

Currently working on:
- [Current task]

Next steps:
- [Next 2-3 tasks]
```

### Need to Check Progress
```
Can you check the status of the WoWRollback consolidation project?
See docs/planning/MASTER_PLAN.md and SESSION_CONTEXT.md.
Which tasks in Phase [X] are complete?
```

---

## 🗺️ Navigation

### Key Files
- `docs/planning/MASTER_PLAN.md` - Complete implementation plan
- `docs/planning/05-wowrollback-consolidation.md` - Detailed architecture
- `docs/planning/ALPHAWDT_MULTITHREADING.md` - Multi-threading strategy

### Main Codebase
- `WoWRollback/WoWRollback.Core/` - Core library (add new formats here)
- `WoWRollback/WoWRollback.Cli/` - CLI commands
- `AlphaWDTAnalysisTool/` - Source for migration
- `DBCTool.V2/` - Source for migration

---

## 🔧 Quick Reference

### Build & Test
```powershell
# Build solution
dotnet build WoWRollback.sln

# Run tests
dotnet test

# Run CLI
dotnet run --project WoWRollback.Cli -- [command]
```

### Current Workflow (Before Consolidation)
```powershell
.\rebuild-and-regenerate.ps1 `
  -Maps @("Azeroth") `
  -Versions @("0.5.3.3368") `
  -AlphaRoot ..\test_data\ `
  -Serve
```

### Target Workflow (After Consolidation)
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

## 📊 Progress Tracker

### Phase 1: Format Readers (Week 1)
- [ ] WdtAlphaReader.cs
- [ ] AdtAlphaReader.cs
- [ ] AdtAlphaConverter.cs
- [ ] AdtLkWriter.cs
- [ ] DbcReader.cs
- [ ] AreaTableReader.cs
- [ ] ListfileService.cs
- [ ] Unit tests

### Phase 2: Multi-Threaded Processing (Week 2)
- [ ] MapConverter.cs
- [ ] TerrainExtractor.cs
- [ ] ShadowExtractor.cs
- [ ] AreaTableProcessor.cs
- [ ] Progress reporting
- [ ] Error handling

### Phase 3: CLI Integration (Week 3)
- [ ] Add Spectre.Console
- [ ] ConvertMapCommand.cs
- [ ] ExtractTerrainCommand.cs
- [ ] ProcessAreaTableCommand.cs
- [ ] Enhance CompareVersionsCommand.cs
- [ ] Update rebuild script

### Phase 4: Testing (Week 4)
- [ ] Unit tests (90%+ coverage)
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] In-game testing
- [ ] Memory profiling

### Phase 5: Cleanup (Week 5)
- [ ] Archive old tools
- [ ] Update documentation
- [ ] Polish CLI
- [ ] Release notes

---

## 🚨 Important Notes

### Don't Break These
- Existing viewer functionality (works well!)
- LK ADT reading (WoWRollback.Core/Formats/Lk/)
- Shadow map overlay feature (just implemented)
- Terrain overlay feature (working)

### Keep Dependencies
- `Warcraft.NET` - LK format support
- `GillijimProject.WowFiles.Alpha` - Alpha format support
- `DBCD.IO` - DBC reading
- Existing listfile logic (community + LK)

### New Dependencies to Add
- `Spectre.Console` (CLI beautification)
- `Microsoft.Extensions.Logging.Abstractions` (if not already present)

---

## 💡 Tips for Implementation

### Start Small
Begin with Phase 1, test each reader thoroughly before moving on.

### Test Incrementally
Don't wait until Phase 4 to test. Write unit tests as you implement each class.

### Compare Outputs
When migrating from old tools, compare outputs byte-by-byte to ensure correctness.

### Use Existing Code
AlphaWDTAnalysisTool and DBCTool.V2 already work - migrate logic, don't rewrite.

### Keep It Clean
Follow existing WoWRollback patterns (namespace structure, naming conventions, etc.)

---

## 🎯 Success Metrics

Track these as you implement:

- [ ] Single `dotnet run` command replaces 3 tools
- [ ] Processing time: 52 min → 7-8 min (measured)
- [ ] CPU usage: 2-8% → 60-80% (observed)
- [ ] Test coverage: 90%+
- [ ] Converted maps work in 3.3.5a client
- [ ] No memory leaks (profiled)
- [ ] Documentation updated
- [ ] Old tools archived

---

## 📞 Getting Help

### If Stuck
1. Check `MASTER_PLAN.md` for detailed guidance
2. Review existing WoWRollback code for patterns
3. Look at source code in AlphaWDTAnalysisTool/DBCTool.V2
4. Check git history for context on design decisions

### When Resuming After Break
1. Read `SESSION_CONTEXT.md` (this file)
2. Check progress tracker above
3. Review most recent commits: `git log --oneline -10`
4. Check which files changed: `git status`

---

**Last Updated**: 2025-10-04  
**Ready to begin Phase 1!** 🚀
