# wow-viewer

`wow-viewer` is the active standalone codebase for:

- reading World of Warcraft terrain and client data across multiple eras
- exporting model-ready V16 Zarr datasets and legacy NPZ terrain shards from real game clients
- inspecting formats like ADT, WDT, PM4, BLP, WMO, MDX, and M2
- converting Alpha-era maps to later formats and reverse (Cataclysm back to Alpha)
- serving as the foundation for the V14 and V16 terrain-model pipelines
- **future: standalone viewer** — porting MdxViewer's rendering and world-session logic into `wow-viewer` is a long-range goal

It is not a toy viewer prototype anymore. A major reason this repo exists is to turn real staged client data into trainable terrain datasets and then consume those datasets with the Python training stack.

## What It Does

The project currently has four practical jobs:

1. **Shared WoW file I/O**
   Read Alpha, pre-release, Wrath, and Cataclysm terrain formats from staged clients.
2. **Dataset generation**
   Export V16 Zarr datasets and legacy NPZ shards with height, normals, alpha layers, liquid, masks, minimap, placements, and placement-derived signals.
3. **Inspection and validation**
   Probe client files and generated outputs without opening the old legacy tools.
4. **Format conversion**
  Convert Alpha monolithic WDT terrain to LK ADT/WDT/WDL output, convert LK/Cataclysm split ADT terrain into monolithic LK ADT output for loose-overlay workflows, and convert LK/Cataclysm ADT terrain back into Alpha-compatible monolithic WDT output.

## V16 Dataset And Training

The current end-to-end terrain-AI lane is V16:

1. `WowViewer.Tool.Harvest harvest-stream` reads staged client data and emits lean length-prefixed raw tile blobs over stdout.
2. `data-harvester/scripts/build_v16_dataset.py build` consumes that stream and writes one finalized Zarr store per client build under `wow-viewer/output/datasets/v16/<build>.zarr/`.
3. `build_v16_dataset.py repair-index` can repair existing `index.parquet` coordinate bookkeeping without regenerating the tensor arrays.
4. `validate_v16_training_ready.py` proves that `V16Dataset`, a real `DataLoader`, and the current `V15Model` architecture can consume the built store.
5. `train_v16.py` trains directly from the finalized V16 Zarr corpus.

This is the modern workflow to care about if the goal is terrain-model training tonight, not just file conversion demos.

Key V16 artifacts:

- `wow-viewer/output/datasets/v16/<build>.zarr/` — finalized per-build tensor store
- `index.parquet` — one row per usable tile, including `build`, `map`, `tile_x`, `tile_y`, signal-presence flags, and placement counts
- `placements.parquet` — per-placement rows with asset path linkage for placement-aware follow-up models
- `<build>.rejected_tiles.jsonl` — rejected missing-required tiles so dropped rows do not disappear into console noise
- `validation/<build>.training_readiness.json` — trainer-readiness report from the validator
- `inspection/<build>.validation_audit_overview.png` — human-eye QA artifact for each finalized store

Current V16 corpus status:

- finalized stores exist for `0_5_3_3368`, `0_5_5_3494`, `0_7_0_3694`, `3_0_1_8303`, `3_3_5_12340`, and `4_0_0_11927`
- all six current `signal_validation.json` files pass
- all six current stores have visual QA artifacts under `wow-viewer/output/datasets/v16/inspection/`
- `0_7_0_3694` still carries the expected allowed warning for zero `has_holes_16` coverage

Repo-level starting points:

- [data-harvester README](./data-harvester/README.md)
- [V16 terrain model spec](./docs/architecture/v16-terrain-model-spec-2026-05-16.md)
- [V16 harvest recovery plan](./docs/architecture/v16-harvest-recovery-plan-2026-05-17.md)

## Spec-Driven Workflow (Spec Kit)

Spec Kit is installed in `wow-viewer` and should be used for non-trivial feature slices before implementation.

- Integration files: `.specify/`
- Codex skills: `.agents/skills/speckit-*`
- Local agent guardrails: `wow-viewer/AGENTS.md`

Typical flow:

1. Run `$speckit-specify` to write or refine the feature spec.
2. Run `$speckit-plan` to generate an implementation plan from that spec.
3. Run `$speckit-tasks` to produce actionable task slices.
4. Run `$speckit-implement` to execute tasks with validation evidence.

## Current Status

### Done

- Unified terrain type system
- Native MPQ archive reader (`NativeMpqService`)
- Harvest/tensor-pack extraction for staged clients
- NPZ shard serialization
- Alpha placement export and resolved model names
- WL liquid fallback
- Minimap lookup via `md5translate`
- AlphaToLk terrain-domain conversion pipeline
- LkToAlpha terrain-domain conversion pipeline (focused round-trip proof, extended with real-data MdxViewer validation)
- ADT NPZ shard preservation of unconsumed raw ADT-family chunks as uint8 blobs
- ADT NPZ promotion of spec-backed preservation signals for `MAMP`, `MFBO`, `MCMT`, `MCLV`, `MCSE`, `MCRF`, `MCRD`, and `MCRW`
- Alpha tile NPZ preservation of raw embedded tile chunks alongside decoded signals
- **AlphaWdtWriter structural fixup**: MAIN grid order corrected to row-major (matching the 0.5.3 client), all 256 MCNKs always emitted with full subchunk structure, and client-required empty MDDF/MODF/MCRF chunks emitted when needed
- **Canonical alphaWDT read/write stack**: `AlphaWdtReader`, `AlphaWdtWriter`, `AlphaTerrainAdapter`, `AlphaToLkConverter`, and `LkToAlphaConverter` now carry the shared alphaWDT contract; `MdxViewer` is a compatibility consumer, not the format owner
- **LkAdtWriter FourCC fix**: all chunk IDs use `FourCC.FromString().ToFileBytes()` instead of `Encoding.ASCII.GetBytes()` for I/O boundary consistency
- **Asset name fixup** (`--target-client-root`): filters placements against assets actually present in the target client (including Alpha per-asset `.wmo.MPQ`/`.mdx.MPQ` wrappers) and removes target-incompatible placements instead of writing fake placeholder asset paths into the alpha output
- **Tileset bundling** (`--bundle-tilesets`): extracts unique BLP textures from source client, writes to `tilesets/{map_name}/`, fixes up WDT MTEX references to local paths
- **Model bundling** (`--bundle-m2s`): converts source `M2` doodads into local bundled `MDX` assets, copies or rewrites bundled model textures beside those outputs, and rewrites Alpha placement paths plus `TEXS` entries to the local bundled BLP paths
- **End-to-end validation**: `convert-lk-to-alpha` output loads and renders successfully in legacy MdxViewer with staged 0.5.3 client (terrain-only and filtered-placements modes)

### Validated

- `0.5.3.3368` (Alpha, used as target for reverse-converted maps)
- `0.5.5.3494`
- `0.7.0.3694`
- `3.0.1.8303`
- `3.3.5.12340`
- `4.0.0.11927` (Cataclysm split ADT source for LkToAlpha conversion)
- **LK→Alpha 4.0.0 Azeroth**: 839/839 tiles converted, validated in MdxViewer against staged 0.5.3 client

### Important Workflow Rule

For full dataset preparation, use the **harvest-first** path.

Use:

- `WowViewer.Tool.Harvest`
- `harvest-map-mpq`
- staged clients under `output\tmp\wowarchive-clients\...`

Do **not** use the older converter-side `dataset-scan` / `dataset-audit` / `dataset-build-cache` chain as the primary shard builder for V14 work. Those commands are legacy manifest/audit helpers, not the canonical full-signal extraction path.

## AlphaWDT Ownership

alphaWDT file semantics live in `wow-viewer`, not in `MdxViewer`.

- Canonical shared surfaces: `AlphaWdtReader`, `AlphaWdtWriter`, `AlphaTerrainAdapter`, `AlphaToLkConverter`, `LkToAlphaConverter`
- `MdxViewer` is the compatibility/runtime host for validation and consumption only
- Future `MdxViewer` alphaWDT read/write work should reuse the shared reader/writer and domain models rather than growing another legacy parser such as `AlphaEmbeddedAdtReader`

Current validated alphaWDT rules:

- `MAIN` is row-major (`tileY * 64 + tileX`)
- all `256` MCNKs are always emitted for embedded tiles
- `MCRF` stays FourCC-wrapped and contiguous inside MCNK payloads
- odd-sized top-level chunks are contiguous; do not pad between them
- placements use the shared raw-file rotation convention and the writer does not subtract `180` degrees on yaw
- doodads stay single-owner in `MCRF` with containing-chunk-first selection; WMOs keep overlap-based multi-chunk refs
- target-client asset presence must come from target archives, wrapper scan, and loose files, not external listfiles

External wiki handoff drafts for Alpha, ADT/v18, PM4, and PD4 live under `wow-viewer/docs/wowdev-wiki/`.

## Supported Client Eras

`wow-viewer` is built around real client data, not mock assets.

| Era | Example build | Notes |
|---|---|---|
| Alpha | `0.5.3.3368`, `0.5.5.3494` | Monolithic WDT terrain |
| Pre-release | `0.7.0.3694` | Early retail-style terrain |
| Wrath pre-release | `3.0.1.8303` | Archive-backed validation target |
| Wrath retail | `3.3.5.12340` | Main LK validation target |
| Cataclysm beta | `4.0.0.11927` | Split ADT / MCCV-era target, reverse-convert to Alpha |

## Quick Start

### 1. Build

```powershell
dotnet build .\wow-viewer\WowViewer.slnx -c Debug
```

### 2. Convert Cataclysm Map to Alpha WDT

```powershell
dotnet run --project .\wow-viewer\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -c Debug -- convert-lk-to-alpha --client-root "output\tmp\wowarchive-clients\4_0_0_11927\World of Warcraft" --map Azeroth --output "output\roundtrip-validation\Azeroth.alpha.wdt"
```

### 3. Validate with MdxViewer

See the **Manual Validation with MdxViewer** section below for the complete workflow.

### 4. Build A V16 Dataset

```powershell
cd .\wow-viewer\data-harvester
uv run python scripts\build_v16_dataset.py build --build 3_3_5_12340
```

### 5. Validate Trainer Readiness

```powershell
cd .\wow-viewer\data-harvester
uv run python scripts\validate_v16_training_ready.py --build 3_3_5_12340
```

### 6. Repair Existing Index Coordinates Without Rebuilding Arrays

```powershell
cd .\wow-viewer\data-harvester
uv run python scripts\build_v16_dataset.py repair-index --build 3_3_5_12340
```

### 7. Start Training

```powershell
cd .\wow-viewer\data-harvester
uv run python scripts\train_v16.py --builds 3_3_5_12340 --train-max-tiles 4000 --train-epoch-tiles 1350 --gpu-duty-cycle 100
```

## Output Layout

The modern dataset workflow writes under `wow-viewer\output\datasets\...`.

Typical structure:

```text
wow-viewer/output/datasets/full_shard_batch_staged_native/
  map_lists/
  manifests/
  shards/
    0_5_3_3368/
      Azeroth/
      Kalimdor/
    3_3_5_12340/
      Azeroth/
      Northrend/
      ...
  validation_samples/
  visualizations/
```

Current V16 structure:

```text
wow-viewer/output/datasets/v16/
  0_5_3_3368.zarr/
    zarr.json
    height_257/
    minimap_rgb/
    index.parquet
    placements.parquet
    _resume_state.json
  0_5_3_3368.rejected_tiles.jsonl
  validation/
    0_5_3_3368.training_readiness.json
```

Round-trip validation output layout:

```text
wow-viewer/output/roundtrip-validation/4_0_0_11927/
  Azeroth.alpha.wdt          # Converted Alpha WDT
  Azeroth.alpha.wdl          # Converted WDL
  loose-alpha-overlay/       # Ready-to-use overlay for MdxViewer
    World/Maps/Azeroth/
      Azeroth.wdt
      Azeroth.wdl
    tilesets/Azeroth/        # Extracted BLP textures (if --bundle-tilesets)
      Tileset/...
    mdxs/Azeroth/            # Local bundled doodad MDX outputs (if --bundle-m2s)
      Creature/...
    wmos/Azeroth/            # Local bundled WMO outputs + rewritten local textures (if --bundle-wmos)
      World/...
  captures-0_5_3/            # MdxViewer capture output
    Azeroth/0.5.3.3368/*.png
```

## Main Tools

### Harvest CLI

```powershell
dotnet run --project .\wow-viewer\tools\harvest\WowViewer.Tool.Harvest\WowViewer.Tool.Harvest.csproj -c Debug -- <command>
```

Key commands:

- `harvest-map-mpq`
- `harvest-map`
- `harvest-tile`
- `extract-unified`
- `synthetic-minimap`

Examples:

```powershell
dotnet run --project .\wow-viewer\tools\harvest\WowViewer.Tool.Harvest\WowViewer.Tool.Harvest.csproj -c Debug -- harvest-map-mpq --client-root "I:\parp\parp-tools\output\tmp\wowarchive-clients\0_5_5_3494\World of Warcraft" --map Azeroth --output-dir "I:\parp\parp-tools\wow-viewer\output\datasets\demo\0_5_5_3494\Azeroth"
```

```powershell
dotnet run --project .\wow-viewer\tools\harvest\WowViewer.Tool.Harvest\WowViewer.Tool.Harvest.csproj -c Debug -- extract-unified --client-root "I:\parp\parp-tools\output\tmp\wowarchive-clients\3_3_5_12340\World of Warcraft" --map Azeroth --tile-x 32 --tile-y 32 --output "I:\parp\parp-tools\wow-viewer\output\datasets\single_tile\Azeroth_32_32.npz"
```

### Python data-harvester tooling

```powershell
cd .\wow-viewer\data-harvester
uv run python scripts\build_v16_dataset.py <command>
```

Important commands:

- `build --build <key>` — build one V16 dataset store
- `build --build <key> --resume` — resume a staged partial build
- `stats --build <key>` — report row counts plus raw-vs-compressed size savings
- `repair-index --build <key>` — rewrite `index.parquet` from metadata only
- `inspect_v16_dataset.py --build <key> --backfill-summary --write-images` — sample and summarize an existing store
- `validate_v16_training_ready.py --build <key>` — prove the trainer stack can consume the dataset
- `train_v16.py --builds <keys...>` — train from finalized V16 Zarr stores

The detailed Python-side workflow lives in [data-harvester/README.md](./data-harvester/README.md).

### Inspect CLI

```powershell
dotnet run --project .\wow-viewer\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -c Debug -- <command>
```

Useful commands:

- `map inspect`
- `map uniqueid-report`
- `map uniqueid-filter`
- `pm4 inspect`
- `pm4 audit`
- `blp inspect`
- `wmo inspect`
- `m2 inspect`
- `mdx inspect`

Example:

```powershell
dotnet run --project .\wow-viewer\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -c Debug -- map inspect --input "I:\parp\parp-tools\wow-viewer\output\datasets\alpha_to_lk\Azeroth\Azeroth_32_32.adt"
```

```powershell
dotnet run --project .\wow-viewer\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -c Debug -- map uniqueid-report --input ".\wow-viewer\test_data\original_development\World\Maps\development\development.wdt" --input ".\wow-viewer\test_data\original_development\World\Maps\development\development_0_0_obj0.adt" --output ".\output\build-validation\uniqueid-report-smoke.json"
```

```powershell
dotnet run --project .\wow-viewer\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -c Debug -- map uniqueid-filter --input ".\output\build-validation\uniqueid-report-smoke.json" --max-uniqueid 100000 --kind m2 --output ".\output\build-validation\uniqueid-filter-smoke.json"
```

### Converter CLI

```powershell
dotnet run --project .\wow-viewer\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -c Debug -- <command>
```

Current important commands:

- `convert-alpha-to-lk` — Alpha monolithic WDT → LK ADT/WDT/WDL
- `convert-split-adt-to-lk` — staged client plus loose overlay split ADT family → monolithic LK ADTs plus regenerated WDT, with optional LK donor-root fallback, optional Alpha donor fallback for missing tiles, and optional JSON auditing via `--report`
- `convert-lk-to-alpha` — LK/Cataclysm split ADT → Alpha monolithic WDT/WDL
- `dataset-list-maps`
- `detect`

**`convert-lk-to-alpha` flags:**

| Flag | Description |
|---|---|
| `--client-root <dir>` / `-c` | Source client root (e.g. 4.0.0) |
| `--map <name>` / `-m` | Map name from Map.dbc |
| `--output <path>` / `-o` | Output Alpha WDT path |
| `--output-wdl <path>` / `--wdl` | Output WDL path (default: `output.wdl`) |
| `--asset-root <dir>` | Optional repeated asset search root for loose `--input` conversions; searched in order before the input directory when bundling tilesets |
| `--tileset-provenance-report <report.json>` | Optional JSON report recording which asset root supplied each bundled tileset texture |
| `--target-client-root <dir>` / `-tcr` | Target client root for asset filtering (e.g. 0.5.3); placements missing from that client are removed from the alpha output |
| `--terrain-only` / `-to` | Strip all placements (for crash-free validation) |
| `--bundle-tilesets` / `-bt` | Accepted for compatibility; tileset extraction is now on by default |
| `--no-bundle-tilesets` / `-nbt` | Disable default tileset extraction |
| `--bundle-m2s` / `-bm` | Opt in to local bundled MDX generation for doodads |
| `--bundle-wmos` / `-bw` | Opt in to local bundled WMO generation |
| `--limit <N>` / `-n` | Limit tile count (for testing) |
| `--verbose` / `-v` | Verbose logging |

Examples:

```powershell
# Basic conversion (839 tiles, ~95s, tilesets bundled by default)
dotnet run --project .\wow-viewer\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -c Debug -- convert-lk-to-alpha --client-root "output\tmp\wowarchive-clients\4_0_0_11927\World of Warcraft" --map Azeroth --output "Azeroth.alpha.wdt"

# Split-family LK output from a staged client plus loose overlay
dotnet run --project .\wow-viewer\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -c Debug -- convert-split-adt-to-lk --client-root "output\tmp\wowarchive-clients\0_6_0_3592\World of Warcraft" --overlay-root ".\wow-viewer\test_data\original_development" --map development --output-dir ".\output\build-validation\development-lk" --limit 5

# Patch missing development roots from the 2021 WoWMuseum 3.3.5 backport while preserving original obj0 placements when present
dotnet run --project .\wow-viewer\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -c Debug -- convert-split-adt-to-lk --client-root "output\tmp\wowarchive-clients\0_6_0_3592\World of Warcraft" --overlay-root ".\wow-viewer\test_data\original_development" --lk-donor-root ".\wow-viewer\test_data\WoWMuseum\335-dev" --map development --output-dir ".\output\build-validation\development-lk-museum" --limit 12

# Write a per-tile audit report showing whether each tile came from original split data, LK donor, Alpha donor, or failed as missing-everything
dotnet run --project .\wow-viewer\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -c Debug -- convert-split-adt-to-lk --client-root "output\tmp\wowarchive-clients\0_6_0_3592\World of Warcraft" --overlay-root ".\wow-viewer\test_data\original_development" --lk-donor-root ".\wow-viewer\test_data\WoWMuseum\335-dev" --map development --output-dir ".\output\build-validation\development-lk-museum" --report ".\output\build-validation\development-lk-museum\split_report.json" --limit 12

# Borrow explicitly listed missing development tiles from staged 0.5.3 Azeroth
dotnet run --project .\wow-viewer\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -c Debug -- convert-split-adt-to-lk --client-root "output\tmp\wowarchive-clients\0_6_0_3592\World of Warcraft" --overlay-root ".\wow-viewer\test_data\original_development" --map development --alpha-donor-client-root "output\tmp\wowarchive-clients\0_5_3_3368\World of Warcraft" --alpha-donor-map Azeroth --alpha-donor-tiles "63,1-3" --output-dir ".\output\build-validation\development-lk-donor" --limit 46

For `convert-split-adt-to-lk`, donor precedence is intentionally asymmetric: if a tile root `.adt` is missing, the command can borrow the root from `--lk-donor-root`, but when the original loose overlay still has `_obj0` or `_tex0` sidecars, those original sidecars win over donor sidecars. This preserves original development placement UniqueIDs whenever original object data still exists, instead of inheriting the donor map's rewritten placement IDs.

When `--report` is supplied, the command writes one JSON entry per processed tile with fields such as `outcome`, `rootSourceKind`, `objectSourceKind`, `originalObjectPlacementsPreserved`, and `message`. The important audit outcomes are `converted-original`, `borrowed-lk-donor`, `borrowed-alpha-donor`, `missing-everything`, and `error`.

# With asset filtering against target client
dotnet run --project .\wow-viewer\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -c Debug -- convert-lk-to-alpha --client-root "output\tmp\wowarchive-clients\4_0_0_11927\World of Warcraft" --map Azeroth --output "Azeroth.alpha.wdt" --target-client-root "output\tmp\wowarchive-clients\0_5_3_3368\World of Warcraft"

# Terrain-only (no assets, crash-proof)
dotnet run --project .\wow-viewer\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -c Debug -- convert-lk-to-alpha --client-root "output\tmp\wowarchive-clients\4_0_0_11927\World of Warcraft" --map Azeroth --output "Azeroth.alpha.wdt" --terrain-only

# Disable default tileset bundling
dotnet run --project .\wow-viewer\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -c Debug -- convert-lk-to-alpha --client-root "output\tmp\wowarchive-clients\4_0_0_11927\World of Warcraft" --map Azeroth --output "Azeroth.alpha.wdt" --no-bundle-tilesets

# Loose LK input with mixed-era tileset roots and provenance capture
dotnet run --project .\wow-viewer\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -c Debug -- convert-lk-to-alpha --input ".\output\build-validation\development-lk-museum-report" --asset-root ".\wow-viewer\test_data\0.6.0\World of Warcraft" --asset-root ".\output\tmp\wowarchive-clients\3_3_5_12340\World of Warcraft" --tileset-provenance-report ".\output\build-validation\development.alpha.tileset-provenance.json" --output ".\output\build-validation\development.alpha.wdt"

# Quick test (5 tiles)
dotnet run --project .\wow-viewer\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -c Debug -- convert-lk-to-alpha --client-root "output\tmp\wowarchive-clients\4_0_0_11927\World of Warcraft" --map Azeroth --output "test.alpha.wdt" --limit 5
```

### Loose development files: practical converter recipes

Use this when working from repo-local loose development files (overlay + donors) instead of a fully self-contained client map.

#### 1) Build monolithic LK ADTs from split development files

```powershell
dotnet run --project .\wow-viewer\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -c Debug -- convert-split-adt-to-lk --client-root "output\tmp\wowarchive-clients\0_6_0_3592\World of Warcraft" --overlay-root ".\wow-viewer\test_data\original_development" --map development --output-dir ".\wow-viewer\output\build-validation\development-lk" --report ".\wow-viewer\output\build-validation\development-lk\split_report.json"
```

#### 2) Fill missing roots from LK donor files, then fallback to Alpha donor tiles

```powershell
dotnet run --project .\wow-viewer\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -c Debug -- convert-split-adt-to-lk --client-root "output\tmp\wowarchive-clients\0_6_0_3592\World of Warcraft" --overlay-root ".\wow-viewer\test_data\original_development" --lk-donor-root ".\wow-viewer\test_data\WoWMuseum\335-dev" --alpha-donor-client-root "output\tmp\wowarchive-clients\0_5_3_3368\World of Warcraft" --alpha-donor-map Azeroth --alpha-donor-tiles "63,1-3" --map development --output-dir ".\wow-viewer\output\build-validation\development-lk-donors" --report ".\wow-viewer\output\build-validation\development-lk-donors\split_report.json"
```

#### 3) Convert those loose LK ADTs back to Alpha WDT/WDL

```powershell
dotnet run --project .\wow-viewer\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -c Debug -- convert-lk-to-alpha --input ".\wow-viewer\output\build-validation\development-lk-donors" --asset-root ".\wow-viewer\test_data\0.6.0\World of Warcraft" --asset-root ".\output\tmp\wowarchive-clients\3_3_5_12340\World of Warcraft" --tileset-provenance-report ".\wow-viewer\output\build-validation\development.alpha.tileset-provenance.json" --output ".\wow-viewer\output\build-validation\development.alpha.wdt" --output-wdl ".\wow-viewer\output\build-validation\development.alpha.wdl"
```

#### `convert-split-adt-to-lk` options (loose development workflow)

| Option | Purpose |
|---|---|
| `--client-root` | Staged client root used for archive reads and base map context |
| `--overlay-root` | Loose file overlay root containing `World\Maps\<map>\...` files |
| `--lk-donor-root` | Optional LK loose donor root used when tile root `.adt` is missing |
| `--alpha-donor-client-root` + `--alpha-donor-map` | Optional Alpha donor fallback source |
| `--alpha-donor-tiles` | Optional tile subset for Alpha donor fallback (e.g. `63,1-3`) |
| `--report` | Writes per-tile JSON audit (`converted-original`, `borrowed-lk-donor`, `borrowed-alpha-donor`, `missing-everything`, `error`) |

Notes:

- Donor precedence is intentional: original overlay sidecars (`_obj0`/`_tex0`) win over donor sidecars when present.
- Use staged clients under `output\tmp\wowarchive-clients\...`; do not use deprecated external client roots.

## Manual Validation with MdxViewer

The legacy `MdxViewer` (in `gillijimproject_refactor`) is the canonical runtime validation tool for converted Alpha WDT maps. Here is the complete workflow:

### 1. Build the converter

```powershell
dotnet build .\wow-viewer\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -c Debug
```

### 2. Convert a map

```powershell
dotnet run --project .\wow-viewer\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -c Debug -- convert-lk-to-alpha --client-root "output\tmp\wowarchive-clients\4_0_0_11927\World of Warcraft" --map Azeroth --output "output\Azeroth.alpha.wdt"
```

### 3. Prepare the loose overlay

The MdxViewer expects the WDT/WDL in a `World\Maps\{mapname}\{mapname}.wdt` structure under a loose overlay root:

```powershell
$overlayDir = "output\overlay"
$mapDir = "$overlayDir\World\Maps\Azeroth"
mkdir -Force $mapDir
Copy-Item "output\Azeroth.alpha.wdt" "$mapDir\Azeroth.wdt"
Copy-Item "output\Azeroth.alpha.wdl" "$mapDir\Azeroth.wdl"
```

If using the default tileset bundling output:

```powershell
Copy-Item -Recurse "output\tilesets\Azeroth" "$overlayDir\tilesets\Azeroth"
```

### 4. Build MdxViewer

```powershell
dotnet build .\gillijimproject_refactor\src\MdxViewer\MdxViewer.CrossPlatform.csproj -c Debug
```

The executable is at `gillijimproject_refactor\src\MdxViewer\bin\Debug\net10.0\ParpToolsWoWViewer.exe`.

### 5. Run MdxViewer with startup automation

**Terrain-only validation (no placements, lowest risk of crash):**

```powershell
& "gillijimproject_refactor\src\MdxViewer\bin\Debug\net10.0\ParpToolsWoWViewer.exe" --verbose --partial-load --game-path "output\tmp\wowarchive-clients\0_5_3_3368\World of Warcraft" --build "0.5.3.3368" --loose-map-overlay "$PWD\output\overlay" --world "World\Maps\Azeroth\Azeroth.wdt" --capture-shot current --capture-output "$PWD\output\captures" --capture-after-frames 30 --capture-with-ui --exit-after-capture
```

**With asset filtering (placements missing from 0.5.3 are removed before write):**

Same command as above, but generate the WDT with `--target-client-root` pointing to 0.5.3:

```powershell
dotnet run --project .\wow-viewer\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -c Debug -- convert-lk-to-alpha --client-root "output\tmp\wowarchive-clients\4_0_0_11927\World of Warcraft" --map Azeroth --output "output\Azeroth_filtered.alpha.wdt" --target-client-root "output\tmp\wowarchive-clients\0_5_3_3368\World of Warcraft"
```

Then repeat the overlay + MdxViewer steps above.

### 6. What to check in the output

In the MdxViewer log output (console or saved to `opencode\tool-output\`):

- `[TerrainAdapter] Error reading chunk` — should be zero. Any occurrence means the WDT is structurally malformed.
- `[TerrainAdapter] Tile (N,N): 256 chunks` — every loaded tile should have 256 chunks.
- `[TerrainRenderer] Now rendering N batched tiles` — confirms GPU-side rendering is active.
- `[MCLQ]` — liquid data is being parsed.
- `[WMO] Parse complete: v14, N groups, M materials` — WMO assets are loading.
- `[Export] [Capture] Saved with-ui frame:` — capture was saved.
- `ExitCode=0` — clean exit.

### 7. MdxViewer startup flags reference

| Flag | Description |
|---|---|
| `--verbose` | Enable verbose diagnostic logging |
| `--partial-load` | AOI streaming (default) — only loads tiles as camera moves |
| `--full-load` | Load all tiles at startup |
| `--game-path <dir>` | Base MPQ client root |
| `--build <version>` | Client build version label |
| `--loose-map-overlay <dir>` | Loose map overlay directory (contains `World\Maps\...`) |
| `--world <path>` | WDT path (virtual or absolute) to load at startup |
| `--capture-shot <name>` | Camera shot name (`"current"` for current camera) |
| `--capture-output <dir>` | Output directory for captures |
| `--capture-after-frames <N>` | Wait N frames before capturing |
| `--capture-with-ui` | Include UI in capture |
| `--capture-no-ui` | Scene-only capture |
| `--exit-after-capture` | Close after capture |

## Data Signals in NPZ Shards

Depending on build and map contents, harvested shards can include:

- `height_257`
- `height_129`
- `height_65`
- `height_33`
- `height_17`
- `mcnr_normal_xyz`
- `mcal_alpha_pack_256`
- `mcly_texture_ids`
- `mcly_layer_mask`
- `hole_mask_16`
- `minimap_rgb_256`
- `mcnk_flags_16`
- `mh2o_surface_height`
- `mh2o_type_mask`
- `mh2o_presence_mask`
- `mclq_surface_height`
- `mclq_type_mask`
- `mclq_presence_mask`
- `wl_liquid_mask`
- `wl_liquid_height`
- `unified_liquid_mask`
- `unified_liquid_height`
- `object_mask_257`
- `object_precise_mask_257`
- `object_instance_mask_257`
- `mddf_mask_257`
- `modf_mask_257`
- `object_filtered_mask_257`
- `placement_mddf_data`
- `placement_modf_data`
- metadata and provenance fields

The final V16 training contract is the consolidated Zarr store plus
`index.parquet` / `placements.parquet`, not these transient shard blobs. The
point of the harvester is to preserve decoded game signals so the Zarr dataset
can keep every fixed-shape terrain/loss signal available for training and QA.

## Alpha/LK Conversion

The Alpha/LK converter lane is validated as a terrain-domain conversion path, not as full file-spec preservation.

Examples already proven:

- `0.5.5` Azeroth: `755/755` tiles
- `0.5.5` Kalimdor: `972/972` tiles
- `0.5.5` EmeraldDream: `256/256` tiles
- `4.0.0` Azeroth → Alpha: `839/839` tiles (MdxViewer real-data validated)

Still open:

- full chunk-for-chunk ADT/WDT preservation across Alpha and LK; current converters rebuild a reduced terrain-domain model and do not preserve every source chunk family
- AreaID crosswalk wiring
- split ADT output for later clients
- Mdx↔M2 converters and WMO v14↔v17 converters

The reverse LK-to-Alpha converter now exists in `wow-viewer` and is covered by focused `LkToAlphaRoundTripTests`, including `MH2O <-> MCLQ` liquid preservation through the shared conversion path. Extended features include `--target-client-root` target-aware placement removal and `--bundle-tilesets` texture extraction.

For harvested ADT tiles, the NPZ shard contract now also preserves unconsumed ADT-family chunks as raw uint8 blobs under `raw_chunks/...` inside the `.npz`, with metadata entries describing source file kind, chunk id, and MCNK location when applicable. That closes the previous hard drop of format data the current tensor pack does not yet decode.

For pre-Cataclysm root ADTs, the tensor-pack path now also promotes `MCRF` into first-class NPZ entries as per-chunk doodad and WMO count grids plus flattened reference-index arrays. `MCSE` is now promoted as per-chunk emitter counts, decoded entry ids and positions when the standard `0x1C` layout is present, and exact per-entry byte matrices; raw `MCSE` fallback is intentionally still retained until broader real-data stride coverage is proven. For Cataclysm+ split object tiles, the same reference shape is used for `MCRD` and `MCRW`. Earlier promoted preservation signals already include `MAMP`, `MFBO`, `MCMT`, and `MCLV`. Staged real-data smoke coverage for both `MCSE` and `MCRF` now scans multiple staged client roots and common map families, but those checks remain availability-based smoke rather than hard-pinned positive regressions in this environment.

For Alpha WDT tiles, harvested tensor packs now also carry raw embedded tile chunks under `raw_chunks/alpha/...`, so Alpha harvesting has the same preservation backstop as the ADT-family path even when only a subset of tile semantics is currently decoded.

## Future: MdxViewer Port to wow-viewer

A long-range goal is to port the rendering and world-session logic from the legacy `gillijimproject_refactor/src/MdxViewer` into `wow-viewer/src/viewer/WowViewer.App`. This will:

- Eliminate the dependency on the legacy reference codebase for runtime validation
- Allow the viewer to benefit from `wow-viewer`'s shared I/O and format libraries directly without going through adapter layers
- Enable standalone viewer builds that don't reference external projects

The current `WowViewer.App` shell exists but needs significant expansion to match MdxViewer's world-session rendering, terrain management, WMO rendering, M2/skin rendering, and UI surfaces.

## Repository Layout

| Path | Purpose |
|---|---|
| `src/core/WowViewer.Core` | shared contracts, terrain models, dataset manifests |
| `src/core/WowViewer.Core.IO` | file readers, writers, archive access, DBC helpers |
| `src/core/WowViewer.Core.PM4` | PM4 library |
| `src/core/WowViewer.Core.Runtime` | runtime-side consumers |
| `src/viewer/WowViewer.App` | future standalone viewer (in progress) |
| `tools/harvest` | canonical NPZ shard builder |
| `tools/inspect` | format inspection and validation |
| `tools/converter` | converters and some legacy helper commands |
| `data-harvester/scripts` | Python visualization and validation helpers |
| `output/datasets` | canonical dataset output root |
| `output/roundtrip-validation` | converter output + captures |

## What To Show People

If you want a short demo flow:

1. Build one V16 Zarr dataset from a staged client build.
2. Run `stats` to show tile counts, signal coverage, and compression savings.
3. Run `validate_v16_training_ready.py` to show the current trainer stack can really read it.
4. Open one generated summary or visualization from `inspect_v16_dataset.py`.
5. Then show `convert-alpha-to-lk` or `convert-lk-to-alpha` if you want the file-conversion side of the repo.
6. Show MdxViewer loading the converted Alpha WDT and rendering it against the staged 0.5.3 client.

That tells the actual story of the project much better than waving around stale one-off tools or pretending the viewer shell is the only important part.

## Development Notes

- staged clients under `output\tmp\wowarchive-clients\...` are the canonical archive-backed inputs
- real dataset outputs belong under `wow-viewer\output\datasets\...`
- round-trip validation outputs belong under `wow-viewer\output\roundtrip-validation\...`
- use `uv` for Python under `wow-viewer\data-harvester`
- bring your own lawful game data

## See Also

- `data-harvester/README.md`
- `docs/architecture/v16-terrain-model-spec-2026-05-16.md`
- `docs/architecture/v16-harvest-recovery-plan-2026-05-17.md`
- `docs/architecture/wow-viewer-full-porting-roadmap.md`
- `docs/architecture/v14-model-and-refactor-plan-2026-05-06.md`
- workspace `AGENTS.md`
- `gillijimproject_refactor/memory-bank/activeContext.md` and `progress.md`

## Data Policy

Bring Your Own Data.

Do not distribute proprietary client data, harvested corpora, or derived outputs from copyrighted game assets.
