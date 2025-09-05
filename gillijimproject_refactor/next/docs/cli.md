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

## Liquids Flags

- `--liquids on|off` — Master switch for liquid conversion (default: on)
- `--liquid-precedence <order>` — Override precedence, e.g. `magma>slime>river>ocean`
- `--liquid-id-map <path>` — JSON mapping of liquid type names to LK LiquidType IDs
- `--green-lava` — Enable green lava variant (mapping behavior TBD)

Notes:
- Precedence determines which liquid wins when multiple instances overlap.
- Mapping values should match entries in LiquidType.dbc for LK; defaults are placeholders.
- The `convert` command uses a real Alpha-era MCLQ extractor (not a stub) to read liquid data from Alpha ADTs.

### Liquids Extraction (Alpha ADT)
- Extractor: `AlphaMclqExtractor` at `next/src/GillijimProject.Next.Core/IO/AlphaMclqExtractor.cs`.
- Heuristics: tries multiple offset origins for `ofsLiquid` (header end, data start, chunk begin).
- Layouts by MCNK flags:
  - Water: depth+height per vertex
  - Ocean: depth-only per vertex; heights inferred from `heightMin`
  - Magma: UV+height per vertex (UV ignored downstream)
- Tiles: 8×8 bytes where low nibble is `MclqLiquidType`, high nibble is `MclqTileFlags` (e.g., `ForcedSwim`, `Fatigue`).
- Robustness: bounds-checked reads; per-chunk errors produce no liquids for that chunk rather than failing the conversion.
- CLI prints a liquids summary including a count of liquid-bearing chunks.

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
