# Progress - WoWRollback Archaeological Tool

## ✅ Completed

### Phase 1: Project Foundation & Memory Bank
- Created dedicated memory bank for WoWRollback project
- Established archaeological perspective and terminology
- Documented core purpose: treating UniqueID ranges as "sedimentary layers" of development history

### Phase 2: Architecture & CLI
- Implemented dual analysis modes:
  - `analyze-alpha-wdt` - Archaeological excavation of source Alpha WDT files
  - `analyze-lk-adt` - Preservation analysis of converted LK ADT files
- Added archaeological terminology to CLI help text
- Created `AlphaWdtAnalyzer` service with proper structure
- Project builds successfully

### Phase 3: Core Services
- `AlphaWdtAnalyzer`: Structured for Alpha WDT file parsing
- `AdtPlacementAnalyzer`: Existing LK ADT analysis (preserved)
- `RangeScanner`: Map-level range aggregation
- `RangeCsvWriter`: Output generation with proper naming (`alpha_<map>`, `lk_<map>`)

## 🔧 In Progress

### Alpha WDT Parsing Implementation
The `AlphaWdtAnalyzer` currently has placeholder TODO sections that need real implementation:
- `ExtractM2ReferencesFromChunk()` - Extract M2 doodad UniqueIDs from Alpha MCNK chunks
- `ExtractWmoReferencesFromChunk()` - Extract WMO building UniqueIDs from Alpha MCNK chunks  
- `IsTilePresent()` - Check tile existence in Alpha WDT structure

## 🎯 Next Steps

### Immediate (Phase 4): Real Data Extraction
1. **Implement Alpha format parsing logic** in `AlphaWdtAnalyzer`
2. **Leverage existing AlphaWDTAnalysisTool parsing capabilities** 
3. **Test with real Alpha WDT files** to verify extraction works
4. **Compare outputs** with existing AlphaWDTAnalysisTool to ensure accuracy

### Future Phases
- **Comparison Analysis**: `analyze-evolution` command comparing Alpha vs LK preservation
- **Archaeological Clustering**: Enhanced clustering that preserves singleton artifacts
- **Rollback Integration**: Complete the rollback functionality for selective content removal
- **Visualization**: Generate reports showing development timeline sedimentary layers

## 🏺 Archaeological Significance

### Current Understanding
- Each UniqueID range = "volume of work" by ancient developers
- Singleton IDs = precious artifacts (experiments, tests, technological changes)
- Gap patterns = development phases and technological shifts
- Modern WoW preserves significant Alpha-era content in fossilized form

### Research Questions
- How much Alpha content survived into modern WoW?
- What UniqueID patterns reveal about development timeline?
- Can we correlate ranges to specific patch versions?
- What do singleton outliers tell us about experimental features?

## 🐛 Known Issues
- Alpha WDT parsing logic needs implementation (currently placeholder)
- Need to verify Alpha ADT format compatibility with existing parsers
- May need to reference AlphaWDTAnalysisTool's working parsing code

## ✨ Success Metrics
- Extract complete UniqueID ranges from real Alpha WDT files
- Generate meaningful archaeological reports
- Preserve all singleton outliers as historical artifacts
- Enable rollback functionality for time-travel map views
