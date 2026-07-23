# Quickstart: Per-Object Occlusion-Aware Masks

**Spec**: [spec.md](spec.md) | **Date**: 2026-07-22

All commands are PowerShell-7-ready, run from `wow-viewer/data-harvester/` unless noted. The
assistant prepares and validates every command dry-run-first; **you run every heavy step**
(FR-012). Real corpus: `../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v3.zarr` (one
directory above `data-harvester/`); split: the Spec 116 `v50-held-out-split-v1` artifact.

## 1. US1 — get the signal into the store

### 1a. Regenerate configs (assistant-runnable, seconds)

After the catalog rows and the one new C# instance array land:

```powershell
uv run python scripts/v50_generate_manifest_template.py --catalog-doc ../docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md --build-id 0_5_3_3368 --release v50.1 --output v50_configs/v50-manifest-template-0_5_3_3368.json --signals-output v50_configs/v50-signals-0_5_3_3368.json
uv run python -m pytest tests/v50/test_manifest_template_matches_catalog.py -q
```

### 1b. Rebuild the store (USER-RUN, heavy — harvests from `H:\CLIENTS`)

Same build command as the current v50 pipeline (Spec 109 quickstart §5, `build` subcommand of
`scripts/v50_build_dataset.py`). No flag changes: the regenerated signals config selects the three
new arrays automatically. The strict object-geometry path already runs during harvest; the
instance-array addition adds one int32 paint per visible fragment — negligible time. Expect the
same wall-clock as the previous full rebuild.

### 1c. Audit the harvested masks (read-only, minutes)

```powershell
uv run python scripts/spec118_audit_object_masks.py --store ../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v3.zarr --output ../output/datasets/v50/v50.1/object-mask-audit.json --write
```

Check: marked fraction p50 in the single-digit-to-tens of percent on object tiles (NOT 80–90%);
underground-heavy tiles ≈0; exclusion counts reported honestly; per-instance class consistency
violations = 0. Eyeball proof: overlay `object_geometry_visible_mask_257` on `minimap_rgb` for one
city tile and one underground-heavy tile (US1 acceptance 1).

## 2. US2 — object-masked loss proof (USER-RUN training)

Paired runs, identical except the flag. Coarse stage shown; repeat for the detailer if desired.

```powershell
uv run python scripts/v50_train_direct_geometry.py --store ../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v3.zarr --held-out-split <spec116-split.json> --source authored --pct-start 0.1 --output ../output/runs/geom-nomask-v1 --confirm-run
uv run python scripts/v50_train_direct_geometry.py --store ../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v3.zarr --held-out-split <spec116-split.json> --source authored --pct-start 0.1 --object-mask-weight 1.0 --output ../output/runs/geom-objmask-v1 --confirm-run
```

Read the run records: relief-stratified MAE on the **object-touched subset** of held-out tiles
(SC-003), plus aggregate (FR-008). A null result is a valid, reportable outcome — record it and
stop before US3 (US2 is the cheap gate for the whole direction).

## 3. US3 — segmenter (USER-RUN training, then assistant-validatable inference)

```powershell
uv run python scripts/spec118_train_objects.py --store ../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v3.zarr --held-out-split <spec116-split.json> --base 24 --epochs 100 --output ../output/runs/objects-v1 --confirm-run
```

Gate (research D-07): held-out visible-object pixel IoU ≥ 0.40 median on object-touched tiles,
per-class recall ≥ 0.50. Then inference on a hand-painted OOD tile (no store needed):

```powershell
uv run python scripts/spec118_infer_objects.py --checkpoint ../output/runs/objects-v1/checkpoint_best.pt --inputs <hand-painted.png> --output ../output/runs/objects-v1/ood --write
```

Human-verify the marked building region (SC-004). Finally, bridge into the geometry chain
(zero trainer changes — dry-run of both trainers against the bridge output is the proof):

```powershell
uv run python scripts/spec118_objects_to_feature_map.py --store ../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v3.zarr --checkpoint ../output/runs/objects-v1/checkpoint_best.pt --output ../output/datasets/v50/v50.1/featuremap-objects-v1.zarr --write
uv run python scripts/v50_train_direct_geometry.py --store ../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v3.zarr --held-out-split <spec116-split.json> --feature-store ../output/datasets/v50/v50.1/featuremap-objects-v1.zarr --output ../output/runs/geom-plus-objects-dryrun
```

### 3b. Augment the existing deconfounding — object prior ALONGSIDE the terrain-feature prior

The promoted deconfounded chain (Spec 115 `v3`) already occupies `--feature-store` with the
terrain-feature classifier's map (roads-as-slopes). Objects occlude *ground height* — a **different**
confound — so the object prior must sit *alongside* that map, not replace it. `--feature-store` is
**repeatable**: pass it once per prior, in a fixed CLI order, and the trainer concatenates their
channels onto RGB (`in_channels = 3 + Σ class_counts`; here `3 + 4 + 2 = 9`). A later prior augments
the deconfounding; it does not evict the earlier one.

```powershell
# both priors together (terrain-feature first, objects second — remember this order)
uv run python scripts/v50_train_direct_geometry.py --store ../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v3.zarr --held-out-split <spec116-split.json> --source authored --pct-start 0.1 --feature-store ../output/datasets/v50/v50.1/featuremap-terrain-v3.zarr --feature-store ../output/datasets/v50/v50.1/featuremap-objects-v1.zarr --output ../output/runs/geom-deconf-plus-objects-v1 --confirm-run
```

The detailer and the coarse **materializer** take the same repeatable `--feature-store` — pass the
priors in the **same order** the coarse checkpoint was trained with, or the channels won't line up:

```powershell
uv run python scripts/v50_materialize_coarse_relief.py --store ../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v3.zarr --checkpoint ../output/runs/geom-deconf-plus-objects-v1/checkpoint_best.pt --feature-store ../output/datasets/v50/v50.1/featuremap-terrain-v3.zarr --feature-store ../output/datasets/v50/v50.1/featuremap-objects-v1.zarr --output ../output/datasets/v50/v50.1/coarse-deconf-plus-objects-v1.zarr --write
```

Paired comparison (SC-003 owner): the `--feature-store terrain` baseline vs. `terrain + objects`,
relief-stratified MAE on the **object-touched** held-out subset. A null result is reportable — it
says the object prior adds nothing the terrain-feature map didn't already carry.

## Validation commands (assistant-run, fast)

```powershell
uv run python -m pytest tests/spec118/ -q
uv run python -m pytest tests/spec118/ tests/v50/ -q
uv run ruff check src/harvester/spec118 src/harvester/v50 scripts/spec118_*.py
dotnet build ../WowViewer.slnx -c Debug
dotnet test ../WowViewer.slnx -c Debug --filter "FullyQualifiedName~VisibleObjectMask|FullyQualifiedName~RawArraySerializer"
```
