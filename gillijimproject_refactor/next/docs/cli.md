# CLI

Commands (scaffold):

- `convert` — Convert Alpha → LK ADTs
  - `--wdt-alpha <path>`
  - `--out <dir>`
  - `--dbc-alpha <path>`
  - `--dbc-wotlk <path>`
  - `--build-alpha <ver>` (optional; e.g., `0.5.3` or leave unset)
  - `--build-wotlk <ver>` (optional; default used if unset, e.g., `3.3.5.12340`)
  - `--areaid-overrides <path>` (optional)
  - `--map-overrides <path>` (optional; JSON dictionary of alpha MapID → LK MapID overrides)

- `analyze` — UniqueID + asset presence analysis
  - `--input <dir>`
  - `--assets-root <dir>` (repeatable)
  - `--report-out <dir>`

- `fix-areaids` — Preview or re-emit ADTs with corrected AreaIDs
  - `--dbc-alpha <path>`
  - `--dbc-wotlk <path>`
  - `--build-alpha <ver>` (optional)
  - `--build-wotlk <ver>` (optional)
  - `--areaid-overrides <path>` (optional)
  - `--map-overrides <path>` (optional; JSON dictionary of alpha MapID → LK MapID overrides)
  - `--out <dir>` (optional)
  - `--dry-run`

- `gen-alpha-wdt` — Generate an Alpha WDT embedding raw LK ADT payloads
  - `--map <MapName>`
  - `--in <dir>` (directory containing LK root ADTs named like `<MapName>_x_y.adt`)
  - `--out <path>` (output Alpha .wdt file)
  - `--no-empty-mdnm` (optional; by default an empty MDNM chunk is included)
  - `--include-empty-monm` (optional; include an empty MONM chunk)
  - `--wmo-based` (optional; include empty MODF and set MPHD flag)

Note: Map.dbc is loaded automatically from the same DBC directory as AreaTable.dbc for each build.

Note: both `convert` and `fix-areaids` will compute an Alpha→LK AreaID mapping using DBCD and a Map.dbc crosswalk. If `--out` is provided, a report is written:
- `areaid_mapping.json` (full mapping, ambiguous and unmatched lists, and `maps` crosswalk section)
- `areaid_mapping_summary.txt` (counts only: maps and areas)

## Examples

```bash
# Help
-dotnet run --project next/src/GillijimProject.Next.Cli -- --help

# Convert (example paths)
-dotnet run --project next/src/GillijimProject.Next.Cli -- convert \
  --wdt-alpha C:/data/alpha/World.wdt \
  --dbc-alpha C:/dbc/alpha/AreaTable.dbc \
  --dbc-wotlk C:/dbc/wotlk/AreaTable.dbc \
  --map-overrides C:/overrides/map_overrides.json \
  --out C:/output/World_LK

# Generate an Alpha WDT from a directory of LK ADTs
-dotnet run --project next/src/GillijimProject.Next.Cli -- gen-alpha-wdt \
  --map World \
  --in C:/data/lk/World \
  --out C:/output/World_alpha.wdt \
  --include-empty-monm
```

## gen-alpha-wdt (reverse WDT generation)

Notes:
- The writer constructs chunks in this order: MVER, MPHD, MAIN, MDNM (optional), MONM (optional), MODF (when `--wmo-based`).
- The MAIN grid is 64x64 entries, 16 bytes each. The first 4 bytes store the absolute file offset to the embedded tile's MHDR; the next 4 bytes store the embedded payload size; the remaining 8 bytes are zero.
- ADT payloads are appended raw. The writer scans each ADT for the MHDR marker (`RDHM` on disk or `MHDR`) to compute the offset.
- Only root ADTs are considered (`<map>_x_y.adt`). Texture/object ADTs are ignored.
