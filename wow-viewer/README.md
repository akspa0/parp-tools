# wow-viewer

Shared WoW file I/O, dataset generation, and terrain AI training pipeline.

---

## Terrain Adapter System (Phase A Complete)

The terrain adapter system provides a unified interface for loading terrain tiles from any WoW format:

- **`ITerrainAdapter`** — interface for tile loading, placement resolution, and name tables
- **`AlphaTerrainAdapter`** — bridges `AlphaWdtReader` → per-chunk `TerrainChunkData` (Alpha 0.5.3)
- **`TerrainTileTensorPack.ToTileLoadResult()`** — converts LK flat-array format → `TerrainChunkData` 
- **`AlphaTileData.ToTileLoadResult()`** — converts Alpha flat arrays → `TerrainChunkData`

**Key types:** `TerrainChunkData`, `TerrainLayer`, `LiquidChunkData`, `MddfPlacement`, `ModfPlacement`, `TileLoadResult`

See `docs/architecture/wow-viewer-library-completeness-plan-2026-05-06.md` for the full phased plan.

---

## V11 Terrain AI

V11 is a single-stage multi-task ConvNeXt-based terrain model. Replaces the broken v10 two-stage pipeline.

### Extraction

Two working paths — both filesystem-only, no archive headaches:

**V9 pipeline** (MCAL/MCLY now included via `dataset-build-cache`):
```
wowviewer-converter dataset-scan --client-root <staged_client> --map <map> --build <label>
wowviewer-converter dataset-audit --input scan.json --output audit.json
wowviewer-converter dataset-curate --input audit.json --output curated.json
wowviewer-converter dataset-build-cache --input curated.json --output-dir <cache_dir>
```

**V10-native single-pass** (filesystem, no temp ADTs):
```
wowviewer-converter dataset-build-v10-stage1 --input-dir <adt_dir> --minimap-root <minimap_dir> --output-dir <out_dir>
```

Both produce NPZ shards with these signals:

| Signal | Shape | Notes |
|--------|-------|-------|
| `minimap_rgb_256` | 256×256×3 | uint8 |
| `height_257` | 257×257 | Full height |
| `height_65` | 65×65 | Mid height |
| `height_17` | 17×17 | Coarse height |
| `mcal_alpha_pack_256` | 1024×1024×4 → 256×256×4 at load | Alpha blend weights |
| `mcly_texture_ids` | 16×16×4 | Texture file IDs per layer |
| `mcly_layer_mask` | 16×16×4 | Active layer flags |
| `mcnr_normal_xyz` | 257×257×3 | Normals |
| `object_mask_257` | 257×257 | Placement footprints |
| `object_precise_mask_257` | 257×257 | Precise footprints |
| `pm4_path_mask` | 257×257 | Pathfinding lines |
| `pm4_building_footprint_mask` | 257×257 | Building shapes |
| `hole_mask_16` | 16×16 | Mesh holes |

### Training

```
pip install timm accelerate lion-pytorch
python scripts/train_v11.py <shard_dir_or_manifest> --output-dir runs/v11 --epochs 300 --batch-size 32
```

**Architecture:** ConvNeXt V2 Tiny encoder (28.6M) + U-Net decoder + multi-task heads. 35.5M params total.

| Head | Output | Loss |
|------|--------|------|
| height_17 | 17×17×1 | L1 + gradient |
| height_65 | 65×65×1 | L1 + gradient |
| height_257 | 257×257×1 | L1 + gradient |
| mcal_alpha | 256×256×4 | L1 (sigmoid) |
| mcly_class | 16×16×N | CE |
| hole_mask | 16×16×1 | BCE |

**26 input channels:** minimap(3) + mcal(4) + normals(3) + mccv(3, 3x dropout) + coarse_height(1) + liquid(2) + objects(2) + pm4(3) + hole(1) + derived(4).

**Key features:** EMA, uncertainty-weighted loss, cosine+warmup schedule, signal dropout (15%), gradient clipping, LRU cache (2GB).

### Inference

```
python scripts/infer_v11.py <checkpoint/best_ema.pt> <shard_dir> --export-obj --output-dir out
```

Outputs per tile: NPZ with predicted arrays, OBJ+MTL+texture mesh, JSON report with MAE.

---

## Shared Libraries

| Library | Contents |
|---------|----------|
| `WowViewer.Core` | Core contracts, maths, dataset manifests, tensor-pack models |
| `WowViewer.Core.IO` | File readers, chunk parsers, archive virtualization, ADT/WDT/WMO/BLP/DBC |
| `WowViewer.Core.Runtime` | Runtime consumers, world-session state, bridge code |
| `WowViewer.Core.PM4` | PM4 parser — the most mature format library |

New format work lands in `Core` / `Core.IO` first, surfaces through `WowViewer.Tool.Converter` or `WowViewer.Tool.Inspect`.

---

## Tools

### Converter CLI
```
dotnet run --project .\tools\converter\WowViewer.Tool.Converter
```

Commands: `dataset-scan`, `dataset-merge`, `dataset-audit`, `dataset-curate`, `dataset-build-cache`, `dataset-build-v10-stage1`, `extract-v10-tensors`, `detect`, `export-tex-json`, `mine-v10-*`, `label-v10-mcly`.

### Inspect CLI
```
dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect
```

Read-only probing: `archive`, `blp`, `m2`, `mdx`, `map`, `lit`, `pm4`, `wmo`.

### Harvest CLI
```
dotnet run --project .\tools\harvest\WowViewer.Tool.Harvest
```

MPQ-backed extraction: `extract-unified` — reads WDT/ADT from MPQ archives via `NativeMpqService`, routes through `AlphaWdtReader` or `AdtTensorPackBuilder`, outputs NPZ shards.

### Desktop App (Paused)
`WowViewer.App` is on hold until the V11 terrain model is trained and validated.

---

## Prerequisites

- .NET 10 SDK
- PowerShell on Windows
- Python 3.11+ with PyTorch, timm, accelerate
- Your own lawful game data

## Build

```powershell
.\scripts\bootstrap.ps1
dotnet build .\WowViewer.slnx -c Debug
```

## Data Policy

Bring Your Own Data. Do not distribute proprietary game data, generated corpora, or model outputs derived from copyrighted sources.
