# Advanced CLI Usage Guide — wow-viewer Tools

This guide covers all CLI tools in `wow-viewer/tools/`. Format: `dotnet run --project <project> -c Debug -- <command> [args]`.

---

## 1. Format Inspector (`WowViewer.Tool.Inspect`)

General-purpose format inspection. Supports M2, MDX, BLP, ADT/WDT, WMO, PM4, archive/listfile operations.

### M2 Models

```powershell
# Basic inspect
dotnet run --project tools/inspect -c Debug -- m2 inspect --input Creature/Orc/Orc.m2

# With skin profile
dotnet run --project tools/inspect -c Debug -- m2 inspect --input Creature/Orc/Orc.m2 --profile-index 0

# Virtual file from staged client
dotnet run --project tools/inspect -c Debug -- m2 inspect --archive-root <staged> --virtual-path Creature/Orc/Orc.m2
```

### MDX Models

```powershell
dotnet run --project tools/inspect -c Debug -- mdx inspect --input <file.mdx>
dotnet run --project tools/inspect -c Debug -- mdx export-json --input <file.mdx> --output report.json
```

### PM4 Files

```powershell
# Inspect all chunks
dotnet run --project tools/inspect -c Debug -- pm4 inspect --input <file.pm4>

# Export asset signals (corpus for matching)
dotnet run --project tools/inspect -c Debug -- pm4 export-asset-signals --archive-root <staged> --seed-placements <tile_obj0.adt> --kind all --output corpus.json

# Match assets against corpus
dotnet run --project tools/inspect -c Debug -- pm4 match-assets --input <file.pm4> --asset-corpus corpus.json --output report.json
```

### WMO / ADT / BLP / Audio

```powershell
dotnet run --project tools/inspect -c Debug -- wmo inspect --input <file.wmo>
dotnet run --project tools/inspect -c Debug -- map inspect --input <tile_00_00.adt>
dotnet run --project tools/inspect -c Debug -- blp inspect --input <file.blp>
dotnet run --project tools/inspect -c Debug -- audio alpha-area --archive-root <staged>
```

### Archive / Listfile

```powershell
# Build listfile cache (required before batch operations)
dotnet run --project tools/inspect -c Debug -- archive build-listfile-cache --archive-root <staged-client> --cache-key <build-name>

# Listfile cache lives at output/cache/archive-listfiles/<sanitized-key>.json
```

---

## 2. Format Converter (`WowViewer.Tool.Converter`)

```powershell
# Alpha WDT → LK (modern) format
dotnet run --project tools/converter -c Debug -- alpha-to-lk --input <alpha.wdt> --output <dir>

# LK WDT → Alpha format
dotnet run --project tools/converter -c Debug -- lk-to-alpha --input <lk.wdt> --output <dir>
```

---

## 3. Terrain Tensor Harvester (`WowViewer.Tool.Harvest`)

Extracts terrain tensors (height, normals, textures, alpha, liquids) from staged game clients into NPZ or Zarr stores. Used by the V16/V18 ML training pipeline.

```powershell
# Single map from loose files
dotnet run --project tools/harvest -c Debug -- harvest-map --input-dir <adt_dir> --minimap-root <dir> --output <shard.npz>

# Map from staged MPQ client
dotnet run --project tools/harvest -c Debug -- harvest-map-mpq --client-root <staged> --map-name <map> --output <shard.npz>

# Full dataset build (Zarr store)
dotnet run --project tools/harvest -c Debug -- harvest-dataset --client-root <staged> --builds <build1,build2> --output <zarr-store>
```

### V22 stream seam (Spec 086)

Preferred operator path: use the Python builder as the single entrypoint and let it call the C# harvester for you.

```powershell
cd wow-viewer/data-harvester
uv run python scripts/build_v22_dataset.py harvest-build --client-root ../../output/tmp/wowarchive-clients/3_3_5_12340 --map Azeroth --limit 1 --output ../output/datasets/v22/3_3_5_12340_smoke.zarr

uv run python scripts/inspect_v22_dataset.py summary --store ../output/datasets/v22/3_3_5_12340_smoke.zarr
uv run python scripts/inspect_v22_dataset.py tile --store ../output/datasets/v22/3_3_5_12340_smoke.zarr --tile-index 0 --output-json ../output/tmp/v22_tile_0.json
```

Low-level seam (still supported): `harvest-stream` is a **binary stdout producer**, not a direct dataset writer. It emits raw V22 tile blobs to standard output and can still be redirected to a file before the Python Zarr writer consumes it.

```powershell
# 1. Emit the raw V22 stream to a file (stdout redirected)
cmd /c "dotnet run --project wow-viewer/tools/harvest/WowViewer.Tool.Harvest -c Debug -- harvest-stream --stream-profile v22 --client-root output/tmp/wowarchive-clients/3_3_5_12340 --map Azeroth --limit 1 1> output\tmp\v22_stream.bin 2> output\tmp\v22_stream.log"

# 2. Build the V22 Zarr store from that stream
cd wow-viewer/data-harvester
uv run python scripts/build_v22_dataset.py build --stream ../output/tmp/v22_stream.bin --output ../output/datasets/v22/3_3_5_12340_smoke.zarr

# 3. Inspect the resulting store
uv run python scripts/inspect_v22_dataset.py summary --store ../output/datasets/v22/3_3_5_12340_smoke.zarr
uv run python scripts/inspect_v22_dataset.py tile --store ../output/datasets/v22/3_3_5_12340_smoke.zarr --tile-index 0 --output-json ../output/tmp/v22_tile_0.json
```

Notes:

- Use `--limit`, **not** `--tile-limit`, on `harvest-stream`.
- `harvest-stream` does **not** write `--output`; stdout redirection is the transport seam.
- `build_v22_dataset.py harvest-build` is now the canonical single-command operator path.
- `build_v22_dataset.py build` remains the low-level "I already have a stream file" entrypoint.

---

## 4. Headless Validation Capture (`WowViewer.Tool.ValidationCapture`)

Captures rendered terrain + object frames from the WoWViewer renderer in headless mode. Used for object-mask ground truth generation.

```powershell
# GPU-viewer-style capture
dotnet run --project tools/validation-capture -c Debug -- capture --gpu-viewer-style --client-root <staged> --map <map> --tile <x_y> --output <dir>

# Real scene dry run (debug diagnostics)
dotnet run --project tools/validation-capture -c Debug -- capture --real-scene-dry-run --client-root <staged> --map <map> --tile <x_y>

# Batch capture with all variants (primary, noliquids, noobjects, objectsonly)
dotnet run --project tools/validation-capture -c Debug -- capture --gpu-viewer-style --client-root <staged> --map <map> --tile <x_y> --variants all
```

---

## 5. M2 Animation Pose Farm (`WowViewer.Tool.AnimFarm`)

Extracts bone animation keyframes from M2/MDX models as BVH motion files + normalized pose clip JSON sidecars.

```powershell
# Single model dump (BVH + pose clip + manifest)
dotnet run --project tools/animfarm -c Debug -- dump --input <path/to/model.m2> --output <outdir>

# Skeleton introspection (no animation)
dotnet run --project tools/animfarm -c Debug -- skeleton --input <path/to/model.m2> --output <outdir>

# Batch mode (requires listfile cache)
dotnet run --project tools/animfarm -c Debug -- batch --client-root <staged> --cache-key <build> --output <outdir> --include "creature/orc/.*"

# With FBX output (instead of BVH)
dotnet run --project tools/animfarm -c Debug -- dump --input <model.m2> --output <outdir> --with-fbx --with-bvh false

# Limit batch for testing
dotnet run --project tools/animfarm -c Debug -- batch --client-root <staged> --cache-key <build> --output <outdir> --limit 10
```

### Output Structure (dump mode)

```
<output>/
├── manifest.json              # Bone hierarchy + sequence list + source hash
├── <sequenceName>.bvh         # BVH motion file per non-alias sequence
├── clip.<sequenceName>.poseclip.json  # Mixamo-normalized pose clip per sequence
└── (future: .fbx files)
```

### Output Structure (batch mode)

```
<output>/
├── Creature/
│   └── Orc/
│       ├── Orc.m2/
│       │   ├── manifest.json
│       │   ├── Stand.bvh
│       │   ├── Walk.bvh
│       │   ├── Run.bvh
│       │   ├── clip.Stand.poseclip.json
│       │   └── clip.Walk.poseclip.json
│       └── OrcFemale.m2/
│           └── ...
├── library.index.json          # Top-level index: tags, paths, summary stats
└── errors.jsonl                # Per-model errors (JSON lines)
```

---

## 6. General CLI Patterns

All tools follow consistent conventions:

| Flag | Alias | Description |
|------|-------|-------------|
| `--input` | `-i` | Input file path |
| `--output` | `-o` | Output directory/file path |
| `--archive-root` | `-r` | Staged game client root |
| `--virtual-path` | `-v` | Virtual path within archive |
| `--listfile` | `-l` | External listfile path |
| `--cache-key` | `-k` | Listfile cache key (build name) |
| `--cache-dir` | `-d` | Listfile cache directory |
| `--help` | `-h` | Tool-specific usage |
| `--include` | | Regex filter (batch modes) |
| `--exclude` | | Regex filter (batch modes) |
| `--limit` | | Max items (test mode) |
| `--quiet` | | Suppress stderr progress |

---

## 7. Common Workflows

### Inspect a model → export animation → load in Blender

```powershell
# 1. Inspect the model
dotnet run --project tools/inspect -c Debug -- m2 inspect --input <model.m2> --sequence-index 0 --time-ms 1000

# 2. Farm animations
dotnet run --project tools/animfarm -c Debug -- dump --input <model.m2> --output <anim-out>

# 3. Open <anim-out>/Stand.bvh in Blender with File > Import > BVH
```

### Build a pose library from a staged client

```powershell
# 1. Build listfile cache
dotnet run --project tools/inspect -c Debug -- archive build-listfile-cache --archive-root <staged> --cache-key <build>

# 2. Batch farm, filtering to creatures
dotnet run --project tools/animfarm -c Debug -- batch --client-root <staged> --cache-key <build> --output <pose-lib> --include "creature/.*"

# 3. Query library index
python -c "import json; d=json.load(open('<pose-lib>/library.index.json')); print([c for c in d['clips'] if 'walk' in c['tags']])"
```

### Compare PM4 objects across tiles

```powershell
# 1. Export asset signals from two tiles
dotnet run --project tools/inspect -c Debug -- pm4 export-asset-signals --seed-placements <tile1_obj0.adt> --kind m2 --output tile1.json
dotnet run --project tools/inspect -c Debug -- pm4 export-asset-signals --seed-placements <tile2_obj0.adt> --kind m2 --output tile2.json

# 2. Match each tile's objects against the other's corpus
dotnet run --project tools/inspect -c Debug -- pm4 match-assets --input <tile1.pm4> --asset-corpus tile2.json --output match.json
```

### Harvest terrain tensors for ML

```powershell
# 1. Verify the dataset pipeline
dotnet run --project tools/harvest -c Debug -- harvest-map-mpq --client-root <staged> --map-name Azeroth_30_48 --output test.npz --dry-run

# 2. Full harvest (adds to Zarr store)
dotnet run --project tools/harvest -c Debug -- harvest-map-mpq --client-root <staged> --map-name Azeroth_30_48 --output test.npz

# 3. Train model (Python)
cd data-harvester
uv run python scripts/train_v18.py --dataset <zarr-store> --epochs 100
```
