# DBCTool.V2 API

This document describes the programmatic API exposed by DBCTool.V2 for AreaTable mapping so other tools can reuse it.

## Namespaces and types

- Namespace: `DBCTool.V2.Core`
  - `AreaIdMapperV2`

## AreaIdMapperV2

A lightweight service that loads source (Alpha/Classic) and target (3.3.5) DBCs through DBCD, builds indexes, and maps `(sourceContinentId, areaNumber)` pairs to LK `AreaTable.ID`.

### Construction

```csharp
// Returns null if required folders are missing or unreadable
AreaIdMapperV2? mapper = AreaIdMapperV2.TryCreate(
    dbdDir: "path/to/WoWDBDefs" /* definitions or parent with /definitions */, 
    srcAlias: "0.5.3" /* or 0.5.5 / 0.6.0 */, 
    srcDir:   "path/to/alpha/dbcs", 
    dir335:   "path/to/3.3.5/dbcs");
```

- `srcAlias` supports: `0.5.3`, `0.5.5`, `0.6.0`
- Uses build canonicalization: 0.5.3.3368, 0.5.5.3494, 0.6.0.3592, 3.3.5.12340

### Mapping

```csharp
bool ok = mapper.TryMapArea(
    contRaw: 0 /* source ContinentID from AreaTable row / WDT mapID */, 
    areaNum: 0xZZZZYYYY /* (zone<<16)|subzone from MCNK Unknown3 */,
    out int targetAreaId,
    out string method);
```

- Returns `true` if a non-zero target was identified; `false` otherwise. The `method` indicates how the mapping was achieved.

### Method values

- `exact` — zone-only chain matched within the target map
- `zone_only` — same as above (no sub match intended)
- `pivot_060` / `pivot_060_zone_only` — mapped via 0.6.0 pivot to LK
- `rename_global` / `rename_global_child` — unique rename found across LK (cross-map)
- `rename_fuzzy` / `rename_fuzzy_child` — fuzzy-rename (edit distance ≤ 2) with unique best
- `onmapdungeon_0` — special-case; force 0
- `unmatched` — no mapping could be determined
- `fallback0` — (consumer emitted 0 due to unmatched)

### Matching rules (deterministic)

- Map-lock: primary matching constrained to the crosswalked LK map for the source ContinentID
- Zone-only: chain is `[zoneName]`; for subzones, resolve the zone name from the sub’s zoneBase row on the same continent
- Parent-agnostic: caller should set `tgt_parentID = targetAreaId` if producing a patch
- Cross-map rename is a deliberate second stage and clearly labeled by `method`
- Special-case `On Map Dungeon` → 0

### Utilities
- Normalization: lowercased, stripped punctuation/spaces, drop leading `the `
- Built-in aliases: `shadowfang` → `shadowfang keep`, `south seas` → `south seas unused`

## Integration guidance

- Alpha ADT writing (MCNK): map before writing `AreaId` at `mcnk + 8 + 0x34`
- Remember precedence if you also support explicit remap files:
  - Prefer explicit map-aware → explicit global → V2 mapper
- For unknowns, it’s safe to write 0. The engine will handle this gracefully.

## See also
- `DBCTool.V2/README.md` for CLI usage and output CSVs
- `DBCTool.V2/Cli/CompareAreaV2Command.cs` for the CSV writer that uses the same rules
