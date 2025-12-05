# System Patterns

## Alpha Format Patterns

### Monolithic WDT Pattern
- **Single File**: The `.wdt` file acts as a container for the entire map.
- **Embedded Tiles**: MHDR, MCIN, and 256 MCNKs are written inline for each tile referenced by MAIN.
- **Global Metadata**: MDNM (M2 names) and MONM (WMO names) are global to the map and reside in the WDT header.
- **Relative Offsets**: MHDR offsets (`offsInfo`, `offsTex`, etc.) are relative to `MHDR.data` (start + 8).
- **Absolute Offsets**: MCIN and MAIN use absolute file offsets.

### Format Parity Pattern
- **Byte-Perfect Goals**: Where possible, output should match the original Alpha client's expectations exactly (e.g., WMO name counting with trailing empty string).
- **Reversed FourCC**: Writers must reverse FourCCs on disk (`KNCM`, `YLCM`) to match Alpha's big-endian-like expectations on little-endian systems (or vice-versa legacy artifact).
- **Raw Chunk Data**: MCVT, MCNR, MCSH, MCAL, MCSE are written as raw data blobs without headers in the MCNK stream.

## Architecture
- Standalone, self-contained CLI with two phases:
  - Phase A: `export-lk` – export LK ADTs from an original Alpha WDT.
  - Phase B: `pack-alpha` – pack Alpha WDT from LK ADTs with toggles to isolate issues.
- Destination-based MCRF re-bucketing: compute destination tile/chunk in Alpha coords, aggregate refs per target tile/chunk, then write per-tile MCRF.
- MCLQ synthesis: detect MH2O via forward FourCC with reversed fallback; synthesize MCLQ where appropriate.
- MCSE scan: hardened, bounded scan that accepts both offset-based and linear scan approaches.
- CSV diagnostics emitted consistently: `mcnk_sizes.csv`, `placements_mddf.csv`, `placements_modf.csv`, `mcrf_diag.csv`.

## Standalone LK→Alpha Converter
- Consumes LK 3.3.5 WDT/ADTs produced by `export-lk` / AlphaWDTAnalysisTool.
- Applies reverse crosswalks (3.3.5→0.5.3) for M2/WMO paths, textures, and other metadata.
- Rebuilds Alpha MPHD/MAIN/MDNM/MONM and per-tile ADTs with **no asset gating**.
- Preserves all MDDF/MODF placements and WMO UniqueIDs; MCRF is rebuilt from per-chunk placement refs.
 - Provides a `roundtrip` orchestration in `AlphaLkToAlphaStandalone` that:
   - Starts from an Alpha WDT/ADT set.
   - Uses `AlphaWdtAnalyzer.Core.AdtExportPipeline` + DBCTool.V2 crosswalk CSVs to export LK 3.3.5 ADTs into a local `lk_export` tree.
   - Feeds those LK ADTs into a standalone LK→Alpha pipeline for diagnostics and future full Alpha ADT reconstruction.
 - Emits `areaid_roundtrip.csv` as a numeric AreaID check, joining:
   - Original Alpha per-MCNK "area IDs" (inferred from monolithic Alpha WDT via `AdtAlpha`).
   - Crosswalked LK `McnkHeader.AreaId` values from the exported LK ADTs.

## Testing Strategy
- Golden-file checks on representative tiles for CSV outputs and `tile-diff` subchunk parity.
- A/B toggles (`--no-coord-xform`, `--dest-rebucket`, `--emit-mclq`, `--scan-mcse`) to isolate root causes deterministically.
