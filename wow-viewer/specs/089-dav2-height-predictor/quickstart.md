# Quickstart: 089 — DA-V2-Small LoRA Height Predictor with Cross-Tile Consistency

**Phase 1 quickstart. Companion to `plan.md`, `research.md`, and `data-model.md`.**

This is the operator/developer on-ramp for Spec 089. It covers the routing fix, Phase 0 validation gate, V22 dataset prerequisite, and the planned local/RunPod command surface.

---

## 1. Routing and Scope

- Active spec directory: `wow-viewer/specs/089-dav2-height-predictor/`
- Active implementation surface: `wow-viewer/data-harvester/src/harvester/v23/`
- Dataset prerequisite: Spec 088 V22 stores under `wow-viewer/output/datasets/v22/`
- RunPod packaging pattern owner: Spec 079
- Spec Kit PowerShell helpers must be run from `wow-viewer/` (or below), not the monorepo root, so `.specify/` is discoverable.

This spec is currently a planning-and-Phase-0 lane. Do **not** start Phase 1 dataset work until the Phase 0 validation commands succeed.

---

## 2. Prerequisites

- Python 3.11+ managed by `uv`
- `wow-viewer/data-harvester/pyproject.toml` synced locally
- At least one V22 store, ideally `wow-viewer/output/datasets/v22/3_3_5_12340.zarr/`
- Trusted staged clients only under `output/tmp/wowarchive-clients/` if a fresh V22 store must be built

`H:\CLIENTS` is forbidden.

---

## 3. Phase 0 Validation Gate

Run from `wow-viewer/data-harvester/`.

```powershell
cd wow-viewer/data-harvester
uv sync
uv run python -c "import harvester.v23"
uv run pytest tests/v23 -m v23 -q
```

Expected result right now:

- `uv sync` resolves the added V23 dependencies
- `import harvester.v23` succeeds
- pytest collects only the gated V23 surface and exits cleanly

If these commands fail, stop. Do not begin Phase 1 implementation.

---

## 4. V22 Dataset Prerequisite

If `wow-viewer/output/datasets/v22/3_3_5_12340.zarr/` already exists, inspect it first:

```powershell
cd wow-viewer/data-harvester
uv run python scripts/inspect_v22_dataset.py summary `
    --store ../output/datasets/v22/3_3_5_12340.zarr
```

If it does not exist, follow Spec 088's two-step build:

```powershell
cd wow-viewer/data-harvester
uv run python scripts/build_v22_dataset.py enrich `
    --v18-store ../output/datasets/v18/3_3_5_12340.zarr `
    --client-root ../output/tmp/wowarchive-clients/3_3_5_12340 `
    --enrichment-output ../output/tmp/v22_enrich/3_3_5_12340.bin `
    --build-key 3_3_5_12340 `
    --limit 1

uv run python scripts/build_v22_dataset.py build `
    --v18-store ../output/datasets/v18/3_3_5_12340.zarr `
    --enrichment ../output/tmp/v22_enrich/3_3_5_12340.bin `
    --output ../output/datasets/v22/3_3_5_12340.zarr
```

The bounded proof script for Spec 088 remains the preferred validation route when rebuilding from scratch.

---

## 5. Planned Phase 1 Dataset Commands

These commands become active once `channels.py`, `dataset.py`, and the prune-table script exist.

```powershell
cd wow-viewer/data-harvester
uv run python scripts/build_tileset_prune_table.py `
    --dataset-dir ../output/datasets/v22 `
    --builds 0_5_3_3368 3_3_5_12340 4_0_0_11927 `
    --top-k 256 `
    --output ../output/datasets/v22/tileset_prune_v23_union.json

uv run python -c "from harvester.v23 import V23HeightDataset; ds = V23HeightDataset('../output/datasets/v22/3_3_5_12340.zarr', build='3_3_5_12340'); print(ds[0]['input'].shape, ds[0]['target_height'].shape)"
```

Expected contract:

- input tensor shape `[15, 256, 256]` in `full` mode
- target tensor shape `[1, 257, 257]`

---

## 6. Planned Training Smoke

This command is the future Phase 5 smoke path after the model, losses, and trainer exist:

```powershell
cd wow-viewer/data-harvester
uv run python scripts/train_v23_height.py `
    --dataset-dir ../output/datasets/v22 `
    --builds 3_3_5_12340 `
    --tileset-prune-table ../output/datasets/v22/tileset_prune_v23_union.json `
    --epochs 2 `
    --train-max-tiles 4 `
    --val-max-tiles 2 `
    --batch-size 4 `
    --gpct-k 4 `
    --gpct-weight 0.1 `
    --sdc-weight 0.1 `
    --bias-free-mask-ratio 0.15 `
    --deterministic `
    --seed 42 `
    --run-name smoke_v23
```

Expected artifacts:

- `models/v23/height/runs/smoke_v23/checkpoints/`
- validation preview PNGs
- metrics/config metadata with commit SHA and data hashes

---

## 7. Planned Deterministic Inference / CAI Proof

```powershell
cd wow-viewer/data-harvester
uv run python scripts/infer_v23_height.py `
    --checkpoint models/v23/height/runs/smoke_v23/checkpoints/v23_height_best.pt `
    --v22-store ../output/datasets/v22/3_3_5_12340.zarr `
    --build 3_3_5_12340 `
    --tiles 30,48 `
    --cai-r 16 `
    --deterministic `
    --seed 42 `
    --save-preview `
    --output-dir models/v23/height/runs/smoke_v23/inference_seed42
```

Repeat with `--seed 12345` and compare the outputs bit-for-bit.

---

## 8. Planned RunPod Bundle Flow

V23 does not invent a second Pod bootstrap route. It follows Spec 079's pattern after the V23 packager exists:

```powershell
cd wow-viewer/data-harvester
uv run python scripts/package_v23_runpod.py `
    --bundle-name v23_smoke_bundle `
    --v22-store-subset-path ../output/datasets/v22/3_3_5_12340.zarr `
    --tileset-prune-table ../output/datasets/v22/tileset_prune_v23_union.json `
    --output-tar runpod/v23/dist/v23_smoke_bundle.tar
```

Pod-side validation sequence:

```bash
bash runpod/v23/install_deps.sh
bash runpod/v23/verify_bundle.sh
bash runpod/v23/smoke.sh
bash runpod/v23/train.sh
```

---

## 9. Per-Phase Checklist

Before moving from one implementation phase to the next:

1. The current phase's pytest target passes.
2. Any required real-data proof for that phase is recorded.
3. `wow-viewer/specs/089-dav2-height-predictor/{plan,tasks}.md` still matches the current work.
4. `wow-viewer/memory-bank/activeContext.md` and `progress.md` are updated if task status changed materially.

---

## 10. Related Reading

- Spec: `specs/089-dav2-height-predictor/spec.md`
- Plan: `specs/089-dav2-height-predictor/plan.md`
- Research: `specs/089-dav2-height-predictor/research.md`
- Data model: `specs/089-dav2-height-predictor/data-model.md`
- V22 schema: `docs/architecture/v22-dataset-signals-2026-06-30.md`
- Spec 088 quickstart: `specs/088-v22-enrichment-from-v18/quickstart.md`
- Spec 079 spec: `specs/079-runpod-integration-guide/spec.md`
- Workspace continuity: `wow-viewer/memory-bank/activeContext.md`
