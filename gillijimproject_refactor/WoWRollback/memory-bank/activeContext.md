# Active Context - WoWRollback Archaeological Tool

## Current Focus
**Phase 1: Alpha WDT Analysis Implementation**
- Implementing Alpha WDT file parsing to extract UniqueID ranges 
- Building archaeological analysis capabilities to identify "volumes of work" by artists
- Preserving singleton IDs and outliers as important historical artifacts

## Recent Discoveries
- Empty CSV output indicates the tool was incorrectly analyzing LK ADT files instead of Alpha WDT source files
- The tool architecture needs dual analysis modes:
  1. `analyze-alpha-wdt` - Extract ranges from source Alpha WDT files
  2. `analyze-lk-adt` - Extract ranges from converted LK ADT files (existing functionality)

## Current Architecture Issues
- `AdtPlacementAnalyzer` currently parses LK ADT files using `GillijimProject.WowFiles.LichKing` classes
- Need to add `AlphaWdtAnalyzer` that can parse Alpha WDT files directly
- Must leverage existing UniqueID clustering logic from `AlphaWDTAnalysisTool.UniqueIdClusterer`

## Next Steps
1. **Implement Alpha WDT Parser**: Create `AlphaWdtAnalyzer` service that can read raw Alpha WDT files
2. **Preserve Archaeological Artifacts**: Ensure singleton IDs and small gaps are captured, not filtered out
3. **Add CLI Commands**: Implement both analysis modes (`analyze-alpha-wdt`, `analyze-lk-adt`)
4. **Generate Per-Map CSVs**: Output `alpha_ranges_by_map_<map>.csv` and `lk_ranges_by_map_<map>.csv`

## Key Technical Requirements
- Parse Alpha WDT files directly (not converted ADT files)
- Use clustering logic similar to `UniqueIdClusterer.FindClusters()` but preserve outliers
- Each UniqueID range represents a "volume of work" by ancient developers
- Singleton IDs are precious artifacts showing experiments/tests/enhancements

## Archaeological Perspective
We're digital archaeologists uncovering fossilized remains of 20+ year old game development. Every UniqueID tells a story of work performed long ago, and the patterns reveal the sedimentary layers of WoW's evolution from 2000 to present day.
