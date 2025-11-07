## Crosswalk Mapping Pattern (CSV-Only)

### Sources
- Crosswalk CSVs: `Area_patch_crosswalk_*.csv`, `Area_crosswalk_v*.csv`
- Loaded via CLI `--crosswalk-dir` (recursive) or `--crosswalk-file` (single)

### Decision Order (Applied Per MCNK) — 2025-10-25
1. Target map-locked numeric: `TryMapByTarget(mapId, aNum)`
2. Target map-name numeric: `TryMapByTargetName(targetMapName, aNum)`
3. Per-source-map numeric (by map name): `TryMapBySrcAreaSimple(mapName, aNum)`
4. Exact numeric match (only if strict=false): `TryMapBySrcAreaNumber(aNum)`
5. Optional pivot (only if `--chain-via-060`): `TryMapViaMid(mapId, aNum)`
6. Else write `0` (unmapped)

### Tie-Breaks and Guards
- Prefer child targets over parents for target-locked matches
- Strict cross-map guard: if a candidate's target map ≠ `mapId`, discard it

### Write Position
- `MCNK.AreaId` resides at `mcnkOffset + 8 + 0x34`

### Invariants
- Always emit `<Map>.wdt` alongside LK ADTs in the LK output folder

## Energy-Efficient Preflight Pattern
- Before Prepare, check per-map outputs and skip redundant steps:
  - Skip LK ADT export if `lk_adts/World/Maps/<map>` exists and is complete.
  - Skip crosswalk generation if DBCTool `compare/v2` CSVs exist for the map.
  - Skip tile layer analysis if `tile_layers.csv` and `layers.json` exist.
  - Always log “SKIP <step> (reason)” for transparency; provide a “Force rebuild” override.

## GUI Runner Pattern
- GUI orchestrates CLI services with overlay + inline logs.
- Auto-navigation: Load → Build on completion; Build → Layers after Prepare success.
- Feature gating: hide data-dependent panels (e.g., Area Groups) until required artifacts exist.
 - CASC datasets: do not WDT‑gate; prefer LK client for `<map>.wdt` lookup; else prompt user with a file picker.

## CSV Parsing Pattern
- Use CsvHelper with ClassMaps for `tile_layers.csv` and `areas.csv`.
- Tolerate header variants and 7/8-column tile CSVs; avoid brittle string.Split.

# Listfile Management Pattern (2025-10-29)

## Two-Listfile Strategy
- Target allowlist (3.3.5): authoritative set of asset paths used in outputs (path-only).
- Modern community listfile: FDID-aware superset for discovery/renames.

## JSON Snapshots
- `snapshot-listfile --client-path <dir> --alias <major.minor.patch.build> --out <json>`
- Snapshot schema: entries `{ path, fdid? }` with metadata (source, clientRoot, version alias, generatedAt).
- Aliases are full build strings; derive from `.build.info`, DBD, or path heuristics.

## Diff Utilities
- `diff-listfiles --a <fileA> --b <fileB> --out <dir>`
- Outputs added_paths.csv, removed_paths.csv, changed_fdid.csv.

## Asset Gate
- Apply before writing outputs (name tables first, placements next):
  - Accept asset only if present in target (3.3.5) listfile; otherwise drop.
  - Emit `dropped_assets.csv` with type and path.
- Current integration: `pack-monolithic-alpha-wdt` gates MDNM/MONM.

# System Patterns - WoWRollback.RollbackTool Architecture

## Three-Tool Separation of Concerns

```
┌─────────────────────────────┐
│ AlphaWDTAnalysisTool        │  Analysis Phase
│ (EXISTS - Reuse!)           │
├─────────────────────────────┤
│ • Scan WDT/ADT files        │
│ • Extract UniqueID ranges   │
│ • Output CSVs/JSON          │
│ • Already has clustering    │
└─────────────────────────────┘
           ↓ CSVs/JSON
┌─────────────────────────────┐
│ WoWRollback.RollbackTool    │  Modification Phase
│ (NEW - Build This!)         │
├─────────────────────────────┤
│ • Read analysis data        │
│ • Modify WDT in-place       │
│ • Bury objects by Z coord   │
│ • Fix terrain holes (MCNK)  │
│ • Disable shadows (MCSH)    │
│ • Generate MD5 checksums    │
└─────────────────────────────┘
           ↓ Modified WDT
┌─────────────────────────────┐
│ WoWDataPlot                 │  Visualization Phase
│ (REFOCUS - Viz Only!)       │
├─────────────────────────────┤
│ • Pre-generate overlays     │
│ • Lightweight HTML viewer   │
│ • No modification logic     │
│ • Pure presentation layer   │
└─────────────────────────────┘
```

## Asset Source Abstraction
- `IAssetSource` contract for reading WoW data assets
- Implementations:
  - `FileSystemSource` (loose files)
  - `CascSource` (WoWFormatLib/CascLib with listfile)
  - `MpqSource` (phase 2)
- Priority resolution: Loose > CASC > MPQ
- Build detection: infer from `.build.info` or path; allow user override

## Shared Pipeline Pattern
- A single pipeline service orchestrates rollback + crosswalk mapping + LK export
- CLI calls the pipeline directly; Viewer backend exposes endpoints that call the same service
- Invariant: always emit `<Map>.wdt` alongside LK ADTs (see Invariants above)

## Core Rollback Architecture

### In-Memory Modification Pattern
```
1. Load entire WDT into byte array (wdtBytes)
2. Parse ADT offsets from MAIN chunk
3. For each ADT:
   a. Create AdtAlpha instance (offset into wdtBytes)
   b. Get Mddf/Modf chunk references
   c. Modify Z coordinate in chunk.Data
   d. Copy modified data back to wdtBytes
4. Write modified wdtBytes to output file
5. Generate MD5 checksum
```

### Chunk Access Pattern
```csharp
// AdtAlpha provides parsed chunk access
var adt = new AdtAlpha(wdtPath, offsetInFile, adtNum);
var mddf = adt.GetMddf();           // Returns Mddf chunk
var modf = adt.GetModf();           // Returns Modf chunk

// Modify in-place
for (int i = 0; i < mddf.Data.Length; i += 36) {
    uint uid = BitConverter.ToUInt32(mddf.Data, i + 4);
    if (uid > threshold) {
        // Bury it: modify Z at offset +12
        byte[] newZ = BitConverter.GetBytes(-5000.0f);
        Array.Copy(newZ, 0, mddf.Data, i + 12, 4);
    }
}

// Write back to file
int fileOffset = adt.GetMddfDataOffset();
Array.Copy(mddf.Data, 0, wdtBytes, fileOffset, mddf.Data.Length);
```

## Spatial MCNK Mapping Pattern

### Coordinate Space Transformations
```
World Coords → ADT Tile → MCNK Chunk

Given placement at (worldX, worldY, worldZ):
1. tileX = floor(worldX / 533.33)
2. tileY = floor(worldY / 533.33)
3. localX = worldX - (tileX * 533.33)
4. localY = worldY - (tileY * 533.33)
5. mcnkX = floor(localX / 33.33)
6. mcnkY = floor(localY / 33.33)
7. mcnkIndex = (mcnkY * 16) + mcnkX
```

### MCNK Hole Flag Management
```
For each buried WMO:
1. Calculate which MCNK(s) it overlaps
2. Locate MCNK header via MHDR offsets
3. Clear Holes field at offset +0x40
4. Write modified header back to wdtBytes
```

## Pre-Generation Pattern (Not On-the-Fly)

### Why Pre-Generate?
- **Performance**: No computation during viewing
- **Reliability**: Generate once, view forever
- **Simplicity**: Pure HTML+JS, no server needed
- **Portability**: Works on any platform with a browser

### Overlay Generation Strategy
```
For each percentile threshold (10%, 25%, 50%, 75%, 90%):
1. Load minimap BLPs
2. Plot placements (green=kept, red=buried)
3. Save as PNG: overlays/{map}_uid_{min}-{max}.png
4. Add entry to overlay-index.json
```

## Command Structure

### Analyze Command
```bash
WoWRollback analyze --input Azeroth.wdt --output analysis/azeroth.json
```
Output: UniqueID statistics, suggested thresholds

### Generate Overlays Command
```bash
WoWRollback generate-overlays --analysis azeroth.json --output overlays/azeroth/
```
Output: PNG overlays + manifest JSON

### Rollback Command
```bash
WoWRollback rollback --input Azeroth.wdt --output rollback/Azeroth.wdt \
  --max-uniqueid 10000 --clear-holes --disable-shadows
```
Output: Modified WDT + MD5 checksum

## Error Handling Philosophy
- **Fail Early on Invalid Input**: Bad WDT path = immediate error
- **Continue on Chunk Errors**: Skip malformed chunks, process rest
- **Validate Before Writing**: Sanity check modifications before output
- **Preserve Originals**: Never modify input files in-place

## Data Integrity Patterns

### MD5 Checksum Generation
```csharp
using var md5 = System.Security.Cryptography.MD5.Create();
var hash = md5.ComputeHash(wdtBytes);
var hashString = BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
File.WriteAllText($"{mapName}.md5", hashString);
```
**Purpose**: WoW client validates minimap files via .md5 files

### File Offset Tracking
```csharp
// AdtAlpha stores its offset for writeback calculations
private readonly int _adtFileOffset;

public int GetMddfDataOffset() {
    int mhdrStartOffset = GetAdtFileOffset() + ChunkLettersAndSize;
    int mddfChunkOffset = mhdrStartOffset + _mhdr.GetOffset(0x0C);
    return mddfChunkOffset + ChunkLettersAndSize; // Skip header
}
```

### Byte Array Modification Pattern
```csharp
// Modify chunk data in-memory
byte[] newValue = BitConverter.GetBytes(valueToWrite);
Array.Copy(newValue, 0, chunk.Data, offsetInChunk, sizeof(float));

// Copy back to file bytes
Array.Copy(chunk.Data, 0, wdtBytes, fileOffset, chunk.Data.Length);
```

## Testing Strategy

### Validation Checklist
- [x] Small maps (< 100 tiles) - Tested on development maps
- [x] Large maps (900+ tiles) - **TESTED: Kalimdor 951 tiles!**
- [x] Alpha 0.5.3 compatibility - **PROVEN WORKING**
- [ ] Later Alpha versions (0.5.5, 0.6.0)
- [ ] Beta versions
- [ ] 1.x retail versions
- [ ] WotLK 3.x versions

### Success Criteria Per Test
1. Tool completes without crashing
2. Output WDT is valid (correct file size)
3. MD5 checksum generated
4. Modified placements have Z = -5000.0
5. Unmodified placements unchanged
6. File loadable in WoW client (manual test)

---

# New Patterns (2025-10-30)

## StatsService (Global Heatmap)
- Purpose: compute dataset‑wide UniqueID min/max across `<build>/*/tile_layers.csv`.
- Output: `heatmap_stats.json` at build root `{ minUnique, maxUnique, perMap: {map: {min,max}}, generatedAt }`.
- Triggers: recompute when any `tile_layers.csv` newer than stats file; lazy on demand.

## FdidResolver
- Inputs: community listfile (CSV), JSON snapshots (from `snapshot-listfile`).
- Normalization: lowercased, forward slashes, DBFilesClient case variants.
- APIs: `ResolvePathToFdid(path)`, `ResolveFdidToPath(fdid)`.
- Diagnostics: write `unresolved_paths.csv` with reason categories.

## MCCV Analyzer
- Reads ADTs (MPQ/CASC or loose) and checks per MCNK:
  - presence of MCCV, HOLES flag state, “hidden by holes”.
- Outputs:
  - `<map>/mccv_presence.csv` (tile_x,tile_y,chunk_idx,has_mccv,holes_set)
  - Optional `<map>/mccv/<map>_<x>_<y>.png` (decoded BGRA to PNG)

## Tile Presence CSV
- Aggregates tile existence from: minimaps, placements (tile_layers.csv), terrain scan, MCCV analyzer.
- Output: `<map>/tile_presence.csv` with booleans per tile.
- GUI uses this to draw gray gridlines for tiles that exist without placements.

## UI Scopes and Overlays
- Heatmap Scope: Local | Global (build) | Global (epoch).
- Layer Scope: Tile | Selection | Map.
- Overlays: Show Empty Tiles, Show MCCV.

# New Patterns (2025-11-07)

## Placement & Gating Rules
- Build MDNM/MONM from the union of all referenced names across scanned tiles; do not gate the name tables.
- Do not gate placements. Always resolve local indices to global indices and write all MDDF/MODF.
- Keep Alpha placement coordinates in X,Z,Y order. Normalize names (`/`→`\`, `.m2`→`.mdx`).
- Build MCRF per-chunk from computed global indices; never gate references.

## Water Rules
- Prefer MH2O-derived MCLQ when present. Set MCNK liquid flags and `offsLiquid` accordingly.
- Compose per-tile flags (fishable/fatigue) and only set `dont_render` when a subtile truly does not exist.
- Ensure min/max heights and vertex heights are populated for 9×9 grid; write type per 8×8 tile.

## Logging & Diagnostics
- Tee all console output to a timestamped log file via `--log-file`/`--log-dir`.
- Emit diagnostics CSVs: kept assets, dropped assets, `objects_written.csv` (per-tile MDDF/MODF counts), `mclq_summary.csv`.
