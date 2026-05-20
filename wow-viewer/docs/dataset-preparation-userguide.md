# Dataset Preparation User Guide

**Purpose**: Extract every usable terrain signal from World of Warcraft game clients into NPZ tensor shards for ML model training.

**Current coverage**: 6 clients — Alpha 0.5.3, 0.5.5, 0.7.0, 3.0.1, 3.3.5, 4.0.0.

---

## 1. Staged Clients

Client data lives in `output/tmp/wowarchive-clients/`. Each folder contains a full WoW install:

```
output/
  tmp/
    wowarchive-clients/
      0_5_3_3368/World of Warcraft/    ← monolithic Alpha WDT (single .wdt per map)
      0_5_5_3494/World of Warcraft/    ← monolithic Alpha WDT
      0_7_0_3694/World of Warcraft/    ← split ADT (separate .adt files per tile)
      3_0_1_8303/World of Warcraft/    ← split ADT
      3_3_5_12340/World of Warcraft/   ← split ADT (WotLK)
      4_0_0_11927/World of Warcraft/   ← split ADT (Cataclysm)
```

**Format differences by era:**

| Version | WDT Type | ADT Storage | Notes |
|---------|----------|-------------|-------|
| 0.5.3 / 0.5.5 | Monolithic Alpha | Embedded in WDT | MDNM/MONM name tables, non-interleaved MCVT/MCNR |
| 0.7.0+ | Split retail | Separate .adt files | MCIN → MCNK subchunk indexing, MH2O liquid (WotLK+) |

### 1.1 Data Layout

**Alpha (0.5.3 / 0.5.5):**
```
Data/
  World/Maps/Azeroth/Azeroth.wdt.MPQ    ← per-map MPQ containing monolithic WDT
  texture.MPQ                             ← shared texture BLP files
  misc.MPQ                                ← shared assets
```

**Retail (0.7.0+ / 3.x / 4.x):**
```
Data/
  world.MPQ                               ← WDT + ADT files at World/Maps/*/
  texture.MPQ                             ← terrain tileset BLPs
  World/Minimaps/<map>/mapXX_YY.blp       ← minimap tile BLPs
```

### 1.2 Minimap Resolution

Minimap tiles are stored as BLP files with MD5-hashed filenames. The mapping is in `md5translate.trs` (retail) or `md5translate.txt` (Alpha) inside the MPQ archives. The tool uses `Md5TranslateResolver` to convert tile coordinates to hash filenames.

---

## 2. Data Extraction

### 2.1 Build the Tool

```powershell
dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```

### 2.1.1 Python launcher

The canonical Python environment is `wow-viewer/data-harvester/.venv`.
If `.venv\Scripts\python.exe` is broken in this checkout, run Python commands
through:

```powershell
cd wow-viewer/data-harvester
.\scripts\run-data-harvester-python.ps1 -c "import sys; print(sys.executable)"
```

### 2.2 Single Tile Extraction

```powershell
dotnet run --project wow-viewer/tools/harvest/WowViewer.Tool.Harvest -c Debug -- extract-unified \
  --client-root "output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft" \
  --map Azeroth \
  --tile-x 32 \
  --tile-y 32 \
  -o "Azeroth_32_32.npz"
```

Output: one NPZ file with all available signals for that tile.

### 2.3 Batch Map Extraction

```powershell
dotnet run --project wow-viewer/tools/harvest/WowViewer.Tool.Harvest -c Debug -- harvest-map-mpq \
  --client-root "output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft" \
  --map Azeroth \
  --output-dir "output/datasets/3_3_5_12340" \
  --limit 100
```

- Skips tiles that don't exist (no ADT file)
- Skips already-extracted `.npz` files (resume-safe)
- Use `--limit N` to cap tiles per run
- Omitting `--limit` extracts ALL tiles for that map

**Expected output per client:**
| Client | Azeroth tiles | ~File size per tile |
|--------|-------------|---------------------|
| 0.5.3 | 685 | 1.5-3 MB |
| 0.5.5 | 755 | 1.5-3 MB |
| 0.7.0 | 755 | 2-4 MB |
| 3.0.1 | 687 | 2-4 MB |
| 3.3.5 | 687 | 2-4 MB |
| 4.0.0 | 839 | 2-5 MB |

### 2.4 Command Reference

| Command | Purpose |
|---------|---------|
| `extract-unified` | Extract single tile from MPQ client |
| `harvest-map-mpq` | Batch extract all tiles from a map via MPQ |
| `harvest-stream` | Stream all tiles as length-prefixed NPZ blobs to stdout (for V16 pipeline) |
| `harvest-tile` | Extract single tile from loose ADT file on disk |
| `harvest-map` | Batch extract from loose ADT directory on disk |
| `synthetic-minimap` | Composite tileset textures + MCAL alpha → minimap (not yet wired) |

**harvest-stream flags:**
| Flag | Description |
|------|-------------|
| `-c, --client-root` | Path to WoW client root (required) |
| `-m, --map` | Map name (required) |
| `-n, --limit` | Max tiles to extract |
| `--tile-workers` | Parallel tile workers inside `harvest-stream` (default: up to 8, ordered output preserved) |
| `-b, --build` | Client build version for version-aware ADT profile |

The `harvest-stream` command writes binary NPZ blobs to stdout using a
length-prefixed protocol: 4-byte magic `NPZB` + 4-byte LE length + NPZ data.
An `ENDS` sentinel marks end-of-stream. All diagnostics go to stderr.
Tile extraction can run in parallel via `--tile-workers`, but stream emission
still preserves deterministic tile order for downstream repair and dataset
index workflows.
This is the input path for the V16 Zarr dataset builder — no intermediate
NPZ files are written to disk.

**extract-unified flags:**
| Flag | Description |
|------|-------------|
| `-c, --client-root` | Path to WoW client root (required) |
| `-m, --map` | Map name, e.g. "Azeroth" (required) |
| `-x, --tile-x` | Tile X coordinate 0-63 |
| `-y, --tile-y` | Tile Y coordinate 0-63 |
| `-o, --output` | Output NPZ path (default: Desktop) |
| `--export-placements` | Also write placement catalog JSON |
| `-s, --synthetic-minimap` | Also generate tileset-composited minimap PNG |

**harvest-map-mpq flags:**
| Flag | Description |
|------|-------------|
| `-c, --client-root` | Path to WoW client root (required) |
| `-m, --map` | Map name (required) |
| `-o, --output-dir` | Output directory for NPZ files (required) |
| `-n, --limit` | Max tiles to extract |

---

## 3. NPZ Signal Reference

Each `.npz` file contains NumPy arrays (`.npy` format inside ZIP) and a `metadata.json` entry.

### 3.1 Always Present (All Clients)

| Array | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `height_257` | (257,257) | float32 | Per-vertex terrain heights (MCVT) |
| `height_65` | (65,65) | float32 | Downsampled heights |
| `height_17` | (17,17) | float32 | Coarse heights |
| `mcnr_normal_xyz` | (257,257,3) | float32 | Per-vertex normals (MCNR) |
| `mcly_texture_ids` | (16,16,4) | int32 | Texture IDs per chunk per layer |
| `mcly_layer_mask` | (16,16,4) | bool | Which layers are active per chunk |
| `mcal_alpha_pack_256` | (256,256,4) | float32 | Alpha blend weights (0-1) |
| `mcsh_shadow_mask_256` | (256,256) | float32 | MCSH shadow occupancy |
| `hole_mask_16` | (16,16) | bool | Per-chunk hole flags |
| `minimap_rgb_256` | (256,256,3) | uint8 | Game minimap tile |

### 3.2 Conditional (Data Dependent)

| Array | Shape | Dtype | When Present |
|-------|-------|-------|--------------|
| `mccv_rgb` | (257,257,3) | float32 | MCCV vertex colors (WotLK+, typically 4.x) |
| `mh2o_surface_height` | (257,257) | float32 | MH2O liquid height (WotLK+) |
| `mh2o_depth` | (257,257) | float32 | MH2O liquid depth |
| `mh2o_type_mask` | (257,257) | int32 | MH2O liquid type per vertex |
| `mclq_surface_height` | (257,257) or (272,272) | float32 | MCLQ legacy liquid height |
| `mclq_type_mask` | (257,257) or (272,272) | int32 | MCLQ liquid type |
| `wl_liquid_mask` | (257,257) | float32 | WL* loose-file liquid mask |
| `wl_liquid_height` | (257,257) | float32 | WL* liquid height |
| `unified_liquid_mask` | (257,257) | float32 | Combined liquid mask (MH2O > MCLQ > WL*) |
| `unified_liquid_height` | (257,257) | float32 | Combined liquid height |

### 3.3 Object & Placement Data (Retail + Alpha)

| Array | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `object_mask_257` | (257,257) | float32 | Binary footprint of all MDDF/MODF placements |
| `object_precise_mask_257` | (257,257) | float32 | Anti-aliased object silhouette |
| `shadow_residual_mask_256` | (256,256) | float32 | Shadow not explained by objects |
| `placement_mddf_data` | (N, 9) | float32 | MDDF flat array: nameId, uniqueId, pos, rot, scale |
| `placement_modf_data` | (N, 14) | float32 | MODF flat array: nameId, uniqueId, pos, rot, bounds |

### 3.4 Metadata JSON

```json
{
  "tile_name": "Azeroth_32_32",
  "map_name": "Azeroth",
  "build_key": "alpha" | "3.3.5.12340" | ...,
  "source_adt_path": "...",
  "available_signals": ["height_257", ...],
  "mcly_texture_names": ["Tileset/AlteracMtns/AlteracDirtBase02.blp", ...],
  "placement_mddf_names": ["Character/Orc/Wyvern/Wyvern.mdx", ...],
  "placement_modf_names": ["World/wmo/Azeroth/Buildings/Stormwind/...", ...],
  "placement_mddf_count": 758,
  "placement_modf_count": 6,
  "minimap_source_tag": "mpq_blp"
}
```

### 3.5 Client-Specific Signal Matrix

| Signal | 0.5.3 | 0.5.5 | 0.7.0 | 3.0.1 | 3.3.5 | 4.0.0 |
|--------|-------|-------|-------|-------|-------|-------|
| height/normals/tex/alpha | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| shadow/holes | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| minimap | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| object/precise masks | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| shadow_residual | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| placement data + names | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| mcly_texture_names | — | — | ✓ | ✓ | ✓ | ✓ |
| MCCV vertex colors | — | — | — | — | — | ✓ |
| MH2O liquid | — | — | — | — | ✓ | ✓ |
| WL liquid fallback | ✓ | ✓ | — | — | — | — |

---

## 4. Coordinate Conventions

### 4.1 Tile Positioning

- MAIN grid: `tileIndex = tileY * 64 + tileX` (row-major, matching the 0.5.3 client)
- Alpha WDT uses absolute MHDR offsets in 16-byte MAIN entries
- Retail WDT uses flag bytes in 8-byte MAIN entries

### 4.2 Chunk Positioning Within Tile

- MCNK header: `IndexX` (offset 0x04) = column, `IndexY` (offset 0x08) = row
- Heightmap: `heightmap[row, col]` where `row = IndexY * 16 + localY`, `col = IndexX * 16 + localX`
- Alpha pack: `alpha_pack[row, col, layer]` same convention
- Texture IDs: `tex_ids[row, col, layer]` where row = IndexY, col = IndexX

### 4.3 Placement Coordinates

- **Alpha MDDF/MODF**: Raw file (X, Z, Y) → renderer position: `(MapOrigin - rawY, MapOrigin - rawX, rawZ)`
- **Retail MDDF/MODF**: Raw file (X, Z, Y) → renderer position via `ComputePositionFromAdt`
- Projection to 257×257 grid: `pixelX = (pos.X - tileWorldX) / tileSize * 256`, `pixelY = (pos.Y - tileWorldY) / tileSize * 256`

### 4.4 Height Base Offsets

- **Alpha (0.5.3/0.5.5)**: Base height at MCNK header offset `0x68` (Unused1 / Position.Z)
- **Retail (0.7.0+)**: Base height at MCNK header offset `0x70` (PosZ)
- LK MCNK header layout: PosX@0x68, PosY@0x6C, PosZ@0x70

---

## 5. Visualization & Verification

### 5.1 Quilt Generation

```powershell
cd wow-viewer/data-harvester
.\scripts\run-data-harvester-python.ps1 scripts/quilt_view.py <npz_path> <output_png>
```

Produces a 10-panel image showing: height, normal-X/Y/Z, shadow, alpha layers 1-3, texture ID grid, and minimap.

### 5.2 Placement Inspection

Placement sample data is written as JSON alongside the NPZ when using `--export-placements`. The metadata JSON in every NPZ already includes counts and resolved model path lists.

### 5.3 Quick Data Check

```python
import numpy as np
f = np.load("Azeroth_32_32.npz")
print("Signals:", list(f.keys()))
h = f["height_257"]
print(f"Heights: min={h.min():.1f} max={h.max():.1f}")
```

---

## 6. Known Limitations

- **0.6.0 ADT profile**: Not yet tested through `AdtProfile060070Baseline`
- **PM4 path masks**: Only generated for retail when loose PM4 files exist alongside ADT
- **Texture compositing**: `synthetic-minimap` not yet wired (reads NPZ but doesn't composite)
- **MPQ compression**: `NativeMpqService` only supports zlib (type 0x02) — bzip2/PKWARE/LZMA not supported
- **DBC/DB2 enrichment**: WorldSafeLocs, AreaTable, GroundEffects, LiquidType metadata not yet in NPZ
- **MCRF per-chunk references**: Only object aggregate counts, not the actual reference index lists
---

## 7. V16 Consolidated Zarr Dataset

V16 replaces the per-tile NPZ shard approach with a single Zarr store per
client build. Data streams directly from the C# harvester into Zarr — **no
intermediate NPZ files on disk**.

### 7.1 Build a V16 Dataset

```bash
# 1. Build the C# harvester
dotnet build wow-viewer/WowViewer.slnx -c Debug

# 2. Build the Zarr dataset
cd wow-viewer/data-harvester

# Single build (auto-discovered terrain maps):
uv run python scripts/build_v16_dataset.py build --build 3_3_5_12340

# Specific maps:
uv run python scripts/build_v16_dataset.py build --build 3_3_5_12340 --maps Azeroth Northrend

# Limit tiles (testing):
uv run python scripts/build_v16_dataset.py build --build 3_3_5_12340 --limit 100

# Multiple builds:
uv run python scripts/build_v16_dataset.py build --builds 3_3_5_12340 4_0_0_11927
```

Output: `wow-viewer/output/datasets/v16/<build_key>.zarr/`

The builder now prints live streaming progress and forwards harvester stderr.
Each run stages into `wow-viewer/output/datasets/v16/<build_key>.zarr.partial/`
and only promotes that staged store to the final `.zarr/` path after
successful finalization.
When `--maps` is omitted, the builder now auto-discovers V16-usable maps from
WDT/archive probe results and skips pure WMO-only, zero-tile, and no-V16-usable
maps before streaming. If a discovered map still produces zero usable V16 tiles
at stream time, the builder warns and skips that map instead of aborting the
whole build.

### 7.2 Zarr Store Layout

Each `<build_key>.zarr/` directory contains:

- Flat chunked Zarr arrays (one per signal, indexed by tile row number)
- `index.parquet` — Parquet table with tile metadata and `has_*` availability flags

The `has_*` columns tell the training loop which signals are real data vs
zero-filled. This eliminates per-sample feature gating complexity.

### 7.3 V16 vs V15

| Aspect | V15 (legacy) | V16 |
|--------|-------------|-----|
| Format | 23K+ individual NPZ files | Single Zarr store per build |
| Liquid data | Missing from most shards | Included for all tiles (zero-filled when absent) |
| Feature standardization | Varies per shard | Every tile has every array |
| Indexing | Path-based (directory walk) | Parquet index with `has_*` flags |
| Compression | Per-file zip level 3 | Blosc-zstd-5 with bitshuffle |
| Temp files | Thousands of NPZ shards | None (pipe-based streaming) |
| Approx size | ~1.5-5 MB/tile (23K files) | ~100 KB/tile (1 Zarr per build) |

---

*Last updated: 2026-05-16 — adds V16 Zarr dataset pipeline and harvest-stream command.*
