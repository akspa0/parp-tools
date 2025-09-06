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

## WDL Exporters

### wdl-obj — Export WDL as OBJ (per tile and merged)

Usage:

```bash
wdl-obj --in <path-to.wdl> [--out-root <dir>] [--scale <double>] [--height-scale <double>] [--no-normalize-world] [--no-skip-holes] [--only-merged] [--only-tiles]
```

- Flags:
  - `--scale` — XY scale multiplier applied after normalization (default: 1.0)
  - `--height-scale` — Z (up) multiplier applied to heights only (default: 1.0)
  - `--no-normalize-world` — Disable XY normalization (default normalization ON)
  - `--no-skip-holes` — Include faces where MAHO hole mask is set (default: skip/omit holes)
  - `--only-merged` — Only write merged OBJ
  - `--only-tiles` — Only write per-tile OBJs

- Outputs (RunDir):
  - `<out-root>/<fileStem_MMDDYY_HHMMSS>/wdl-obj.log`
  - `<out-root>/<fileStem_MMDDYY_HHMMSS>/summary.json`
  - `<out-root>/<fileStem_MMDDYY_HHMMSS>/tiles/*.obj` (unless `--only-merged`)
  - `<out-root>/<fileStem_MMDDYY_HHMMSS>/merged.obj` (unless `--only-tiles`)

### wdl-glb — Export WDL as GLB (per tile and merged)

Usage:

```bash
wdl-glb --in <path-to.wdl> [--out-root <dir>] [--scale <double>] [--height-scale <double>] [--no-normalize-world] [--no-skip-holes] [--only-merged] [--only-tiles]
```

- Same flags and outputs as `wdl-obj`. Uses local SharpGLTF source (v1.0.5) referenced from `lib/SharpGLTF` (Core + Toolkit projects).

### Coordinate Mapping and Normalization

- XY normalization: by default, cell size = 533.3333333 / 16 ≈ 33.3333333 (ADT block width / 16 cells).
  - XYCellScale = (NormalizeWorld ? 33.3333333 : 1.0) × `--scale`.
  - Merged spacing = 16 × XYCellScale per tile.
- Axis mapping in exports (OBJ and GLB):
  - X = column index `i` × XYCellScale
  - Y = row index `j` × XYCellScale
  - Z = −height × `--height-scale` (Z is up; negative sign chosen to ensure front-face winding in common viewers)
- Triangles: two per cell. Winding is flipped relative to height-up change to avoid backface culling in viewers.
- Holes: when not using `--no-skip-holes`, faces in MAHO-masked cells are omitted.
- Resolution: WDL grid is 17×17 per tile (1/16th the horizontal resolution of ADT’s 145×145).

### Examples

```bash
# OBJ normalized (default), per-tile and merged
-dotnet run --project next/src/GillijimProject.Next.Cli -- wdl-obj --in C:/data/Azeroth.wdl --out-root ./out

# GLB normalized with vertical exaggeration
-dotnet run --project next/src/GillijimProject.Next.Cli -- wdl-glb --in C:/data/Azeroth.wdl --height-scale 1.5

# Disable normalization (raw grid units)
-dotnet run --project next/src/GillijimProject.Next.Cli -- wdl-obj --in C:/data/Azeroth.wdl --no-normalize-world

# Only merged OBJ
-dotnet run --project next/src/GillijimProject.Next.Cli -- wdl-obj --in C:/data/Azeroth.wdl --only-merged

## OBJ Transform (obj-xform)

Transform OBJ(s) with axis flips, Z-rotations (0/90/180/270), translations, and automatic parity-aware winding swap.

- Example: apply server-origin subtraction measured for ADT and flip X to mirror left

```bash
dotnet run --project next/src/GillijimProject.Next.Cli -- obj-xform \
  --in "C:/proj/adt/dev_0_0.obj" \
  --out-dir ./out/aligned/adt \
  --preset adt-to-wdl --flip-x
```

- Manual flags

```bash
obj-xform --in <path|glob> --out-dir <dir> \
  --flip-x --flip-y --flip-z \
  --rotate-z 0|90|180|270 \
  --translate-x <v> --translate-y <v> --translate-z <v> \
  [--no-auto-winding]
```

## ADT↔WDL Alignment (align-adt-wdl)

Discover ADT→WDL orientation (flipX/flipY/rotZ) and translation by grid correlation, then optionally write an aligned OBJ.

```bash
# With the test files you provided
dotnet run --project next/src/GillijimProject.Next.Cli -- align-adt-wdl \
  --wdl-obj next/test_data/alignment/WDL/tile_0_0.obj \
  --adt-obj next/test_data/alignment/ADT/development_0_0.obj \
  --write-obj --out-dir ./out/aligned/adt
```

The command prints suggested obj-xform flags and writes `<name>_aligned.obj` when `--write-obj` is set.
