# AreaID Restoration Approach (Alpha 0.5.x → LK 3.3.5)

Preservation-first mapping of Alpha-era terrain to LK AreaIDs using strict, numeric, auditable crosswalks. No DBC edits and no heuristic filling. This document explains the data model, the crosswalks, how the patcher consumes them, and how to verify coverage.

- Goal: Restore canonical AreaIDs on exported WotLK ADTs while preserving the Alpha source of truth.
- Philosophy: Period-correctness over cosmetic perfection. If Alpha didn’t encode a subzone, we don’t invent it.
- Scope: 0.5.3 and 0.5.5 Alpha inputs, optional 0.6.0 pivot, LK 3.3.5 targets.

---

## 1) Data model and constraints

- Alpha MCNK AreaId packs as a 32-bit value: `hi16 = zone`, `lo16 = sub`.
  - Validation: the `ParentAreaNum` of a child row must equal the zone base.
  - See `compare/alpha_areaid_decode_v2.csv` for decoded rows and validation flags.
- Target is LK 3.3.5 AreaTable IDs and names (for reporting only).
- Optional pivot via 0.6.0 to bridge early renames and zone reshuffles.

Why this matters: If `lo16 == 0`, the ADT chunk carries only the zone base. The strict patcher will therefore write the LK zone ID, not a child.

---

## 2) Crosswalk generation (DBCTool.V2)

DBCTool.V2 emits per-map crosswalk CSVs under `DBCTool.V2/dbctool_outputs/session_*/compare/v2/`:

- `Area_patch_crosswalk_via060_map{mapId}_{src}_to_335.csv`
  - Source rows are Alpha (0.5.3/0.5.5); optional via 0.6.0; targets are LK 3.3.5 IDs.
  - Only rows with non-zero `tgt_areaID` are used by the patcher.
- Dumps and audits for transparency:
  - `AreaTable_dump_{src}.csv`, `AreaTable_dump_3.3.5.csv`
  - `AreaTable_mapping_{src}_to_335.csv`
  - `alpha_areaid_decode_v2.csv`

Crosswalks are map-locked based on the source map. We also keep per-map patch files to avoid cross-map leakage.

---

## 3) Patcher consumption (AlphaWDTAnalysisTool)

- Loader: `AlphaWdtAnalyzer.Core/Export/DbcPatchMapping.cs`
  - Loads only via060 rows with non-zero targets.
  - Populates a per-(mapName, src_areaNumber) numeric map.
  - Captures `tgt_parentID` for auditing but doesn’t use it for mapping.
- Writer: `AlphaWdtAnalyzer.Core/Export/AdtWotlkWriter.cs`
  - For each chunk, get Alpha `alpha_raw` from the source ADT capture.
  - Strict lookup: `TryMapBySrcAreaSimple(mapName, alpha_raw)`.
  - If found: write the LK AreaID (`reason=patch_csv_num`), else write `0`.
  - Emits per-tile verify CSVs: `tile_x,tile_y,chunk_index,alpha_raw,lk_areaid,tgt_parentid,on_disk,reason,lk_name`.

This strict path deliberately avoids any heuristic matching, LK-name lookups, or fallbacks. The result is fully auditable and reproducible.

---

## 4) Overrides policy and file format

When numeric evidence exists but is missing in the generated crosswalks, we add rows to a per-map overrides file:

- Example: `Area_patch_crosswalk_overrides_map1_0.5.5_to_335.csv` (Kalimdor)
- Header (stable schema):
```
src_mapId,src_mapName,src_areaNumber,src_parentNumber,src_name,tgt_mapId_xwalk,tgt_mapName_xwalk,tgt_areaID,tgt_parentID,tgt_name
```
- Rules:
  - Numeric-only. One row per `src_areaNumber`.
  - Map-locked. No cross-map targets.
  - Never overwrite explicit mappings with heuristics.

---

## 5) Verification and aggregation

- During export, enable verify output. Then use the aggregator script:
  - Script: `tools/agg-area-verify.ps1`
  - Command:
    ```powershell
    pwsh -NoProfile -ExecutionPolicy Bypass -File "tools/agg-area-verify.ps1" -ExportRoot "<export root>" -Map Kalimdor
    ```
  - Outputs:
    - `<export root>/csv/maps/_summary/alpha_areas_used.csv`
    - `<export root>/csv/maps/_summary/unknown_area_numbers.csv`
    - Per-map splits like `alpha_areas_used_Kalimdor.csv`
  - Columns include `area_hi16` and `area_lo16` to filter zones (`lo16==0`) vs subzones.

Use the per-map files to confirm which `alpha_raw` values truly appear on disk and whether they mapped. Add overrides only for real, observed `alpha_raw` values that remained unmapped.

---

## 6) Observed outcomes (Alpha 0.5.3 / 0.5.5 → LK 3.3.5)

- Kalimdor
  - Tanaris: Only zone-level present (`alpha_raw=0x00100000`, `lo16=0`) → LK 440.
  - Silithus: Effectively “On Map Dungeon” in Alpha (no `src_areaNumber`) → chunks remain `0`.
  - Felwood, Azshara: No subzone alpha numbers observed on disk → zone-level mapping.
  - Feralas: `Thalanaar` present (`alpha_raw=0x000A0001`) → LK 489.
- Eastern Kingdoms
  - Blasted Lands: `Demonic Stronghold` (`alpha_raw=0x00020002`) maps to LK 73 (`The Tainted Scar`) via via060.

These outcomes reflect the source material: if Alpha didn’t paint subzones into the ADTs, strict mapping won’t synthesize them.

---

## 7) Alternative not chosen (documented for completeness)

“LK-painting” approach: scan LK 3.3.5 ADTs for AreaIDs and paint them back onto Alpha terrain to force modern names everywhere.

- Pros: Complete modern naming coverage.
- Cons: Not period-correct; loses provenance; can introduce cross-map or era-mismatch artifacts.
- Status: Not used in this preservation path, but can be explored as a separate variant.

---

## 8) Future enhancements

- Suggestions generator: `alpha_to_335_suggestions_v2.csv` (map-locked, zone-locked, via060-anchored fuzzy candidates for human review).
- Name normalization/aliases (apostrophes, accents, US/UK spelling) to improve safe candidate sets.
- Bridging via 1.0.0 and 2.0.0 dumps to strengthen rename inference chains.

---

## 9) Quick start

Prerequisite: Follow [Input Data Preparation](input-data-prep.md) to organize your `tree/` folder before running the tools.

1. Generate DBCTool crosswalks for 0.5.x (and optional 0.6.0) and 3.3.5.
2. Assemble patch directory for the patcher containing at least:
   - `Area_patch_crosswalk_via060_map{0|1}_0.5.3_to_335.csv`
   - `Area_patch_crosswalk_via060_map{0|1}_0.5.5_to_335.csv`
   - `Area_patch_crosswalk_overrides_map{0|1}_0.5.5_to_335.csv` (optional, for local fixes)
3. Run AlphaWdtAnalyzer export with `--dbctool-patch-dir` and `--dbctool-lk-dir`, plus `--verbose` for verify CSVs.
4. Run `tools/agg-area-verify.ps1` on the export root and inspect the per-map summaries.
5. If any real `alpha_raw` remain unmapped, add exact numeric override rows.

---

## 10) Rationale

This process prioritizes digital preservation: it restores what the Alpha data actually encoded, produces auditable CSVs, avoids DBC mutation, and keeps the door open to backporting modern map edits onto old engines without rewriting core game data. It also makes future refinements straightforward—new evidence can be added as numeric overrides without destabilizing existing mappings.
