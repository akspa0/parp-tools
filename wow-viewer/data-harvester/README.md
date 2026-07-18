# wow-viewer data-harvester

Current state first. Old long-form lane history trimmed out.

## Current active lanes

- **Spec 109 `109-v50-clean-room-audit` — active dataset lane.** V50 is the canonical dataset
  target for new work: fail-closed trust boundary, complete per-build Zarr stores with real
  content-hash identity, immutable curriculum manifests. See "V50 quickstart" below.
- Spec 089 `089-dav2-height-predictor` — active model lane (currently trains against the older V22
  contract; migrating to V50 is tracked in Spec 109/111, not yet complete).
- Spec 088 `088-v22-enrichment-from-v18` — V22 dataset contract feeding 089; legacy relative to V50,
  kept live only until 089's training path moves over.

Background only:

- Spec 047 — focused V18 operator path.
- Spec 076 / 077 — paused/background research surfaces.
- Spec 074 / 075 / 066 / 067 / 068 — historical only.

## V50 quickstart

Build the full V50 clean-room dataset corpus for the configured `0_5_3_3368` client (all four
terrain-bearing world maps: Kalimdor, Azeroth, PVPZone02, Kalidar — build, finalize, and two
curation passes each: strict object-free and object-inclusive; ~15-20 minutes total):

```powershell
cd wow-viewer/data-harvester
uv run python scripts/v50_pipeline_runner.py --confirm
```

Omit `--confirm` to dry-run first (prints every command without launching anything); add
`--sample N` to cap tiles per map for a smoke test. A single dirty tile or one map's failure no
longer aborts the whole run — the script prints a per-map summary at the end. Full detail, per-map manual commands, and the
current one-client-build scope limitation are in
[`../docs/dataset-preparation-userguide.md`](../docs/dataset-preparation-userguide.md#8-v50-clean-room-dataset-current-canonical-lane)
and [`../specs/109-v50-clean-room-audit/quickstart.md`](../specs/109-v50-clean-room-audit/quickstart.md).

## Environment

```powershell
cd wow-viewer/data-harvester
uv sync
```

Run all Python entrypoints from this directory:

```powershell
uv run python <script>
```

## Canonical docs

- [../AGENTS.md](/I:/parp/parp-tools/wow-viewer/AGENTS.md)
- [../docs/DOCUMENTATION-STATUS.md](/I:/parp/parp-tools/wow-viewer/docs/DOCUMENTATION-STATUS.md)
- [../specs/088-v22-enrichment-from-v18/spec.md](/I:/parp/parp-tools/wow-viewer/specs/088-v22-enrichment-from-v18/spec.md)
- [../specs/089-dav2-height-predictor/spec.md](/I:/parp/parp-tools/wow-viewer/specs/089-dav2-height-predictor/spec.md)
- [../specs/077-minimap-deconstruction-engine/user-guide.md](/I:/parp/parp-tools/wow-viewer/specs/077-minimap-deconstruction-engine/user-guide.md)

## V22 quickstart

V22 is built from an existing V18 store. Canonical output is `paths_only`.

### 1. Build enrichment stream

```powershell
cd wow-viewer/data-harvester
uv run python scripts/build_v22_dataset.py enrich `
  --v18-store ../output/datasets/v18/3_3_5_12340.zarr `
  --client-root ../output/tmp/wowarchive-clients/3_3_5_12340 `
  --enrichment-output ../output/tmp/v22_enrich/3_3_5_12340.bin `
  --build-key 3_3_5_12340 `
  --limit 1
```

### 2. Build V22 store

```powershell
cd wow-viewer/data-harvester
uv run python scripts/build_v22_dataset.py build `
  --v18-store ../output/datasets/v18/3_3_5_12340.zarr `
  --enrichment ../output/tmp/v22_enrich/3_3_5_12340.bin `
  --output ../output/datasets/v22/3_3_5_12340.zarr
```

### 3. Inspect

```powershell
cd wow-viewer/data-harvester
uv run python scripts/build_v22_dataset.py stats `
  --store ../output/datasets/v22/3_3_5_12340.zarr
```

## V23 quickstart

### Local proof suite

```powershell
cd wow-viewer/data-harvester
uv run python -m pytest tests/v23 -m v23 -q
```

### Train smoke

```powershell
cd wow-viewer/data-harvester
uv run python scripts/train_v23_height.py `
  --dataset-dir ../output/datasets/v22 `
  --builds 0_5_3_3368 3_3_5_12340 `
  --memory-profile 12gb `
  --run-name v23_smoke_local
```

### Deterministic inference

```powershell
cd wow-viewer/data-harvester
uv run python scripts/infer_v23_height.py `
  --store ../output/datasets/v22/3_3_5_12340.zarr `
  --checkpoint ../models/v23/height/runs/v23_smoke_local/checkpoints/v23_best.pt `
  --tile-index 0 `
  --output-dir ../output/inference/v23_smoke_local
```

## Focused V18 operator lane

Use this only when user explicitly wants focused V18 terrain runs.

### Build focused curation manifest

```powershell
cd wow-viewer/data-harvester
uv run python scripts/build_v18_curation_manifest.py `
  --run-name v18_focus_terrain_v1 `
  --workers 4 `
  --chunk-size 128
```

### Train focused height

```powershell
cd wow-viewer/data-harvester
uv run python scripts/train_v18_focus.py height `
  --device cuda `
  --epochs 40 `
  --curation-manifest ../output/datasets/v18/curation/v18_focus_terrain_v1 `
  --run-name v18_height_focus_full_v1
```

### Train focused normal

```powershell
cd wow-viewer/data-harvester
uv run python scripts/train_v18_focus.py normal `
  --device cuda `
  --epochs 40 `
  --curation-manifest ../output/datasets/v18/curation/v18_focus_terrain_v1 `
  --run-name v18_normal_focus_full_v1
```

## RunPod / remote

- Spec 089 remote work is the active next proof owner; no more local training runs unless explicitly reopened.
- Use [../docs/runpod-integration-cookbook.md](/I:/parp/parp-tools/wow-viewer/docs/runpod-integration-cookbook.md) for generic bundle/runtime flow.
- Package the V18 curation manifest with the V22 bundle:

```powershell
cd wow-viewer/data-harvester
uv run python scripts/package_v23_runpod.py `
  --bundle-name v23_curated_2k_keymaps `
  --dataset-dir ../output/datasets/v22 `
  --builds 0_5_3_3368 3_3_5_12340 `
  --tileset-prune-table ../output/datasets/v22/tileset_prune_v23_union.json `
  --curation-manifest ../output/datasets/v18/curation/v18_focus_terrain_all_v1/kept_tiles.parquet `
  --include-v22-subset-tiles 2000 `
  --output-tar runpod/v23/dist/v23_curated_2k_keymaps.tar
```

On the Pod:

```bash
bash runpod/v23/install_deps.sh
bash runpod/v23/verify_bundle.sh
bash runpod/v23/smoke.sh
bash runpod/v23/train.sh
```

The no-arg `train.sh` default is the curated 2K key-map run with visible per-step logging, validation every second epoch, SDC/GPCT/bias-free masking enabled, and startup batch autotune.

## Historical lanes

- Spec 076 / 077 outputs remain reusable research inputs.
- Spec 074 / 075 / 066 / 067 / 068 stay on disk as historical evidence.
- Spec 086 / 087 are superseded by Spec 088 and should not be used as live operator docs.
