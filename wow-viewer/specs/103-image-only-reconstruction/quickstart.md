# Quickstart: Spec 103 v7 revival

Agent-prepared commands. **The USER runs every capture/training/GPU step** (AGENTS RULE 0).
Python runs from `wow-viewer/data-harvester/`; dotnet runs from `wow-viewer/`.

> **Local training is OFF (2026-07-14):** the local GPU overheated mid-run. All training moves
> to RunPod. Local CPU sanity tests (§0) are still fine to run anytime.

## Start here: curate harder, inspect, then package the next RunPod run

Do not train from the older 2,253-tile clean-only bundle. The next run starts with the map-canvas
prefab proof in §3d, because family/context coverage—not raw tile count—is now the corpus contract.

1. **Run and inspect the bounded prefab curation** (§3d). Confirm `split_audit` is clean, every
   selected tile adds a declared coverage token, and every duplicate names its representative.

2. **Build the bundle from the new reduced manifest** (local, CPU; USER runs):

   ```powershell
   cd wow-viewer/data-harvester
   uv run python scripts/package_spec103_runpod.py `
       --store ../output/datasets/v18/3_3_5_12340.zarr `
       --curation-manifest ../output/spec103/prefab_curation_v1 `
       --output-root ../output/cloud-packages/spec103 `
       --bundle-name spec103_prefab_bundle_v1
   ```

   Produces `../output/cloud-packages/spec103/spec103_prefab_bundle_v1.tar`; its size and selected
   row count are intentionally determined by the curation proof rather than pre-declared here.

3. **Start a pod and transfer the tar** — see full §5 below for `runpodctl`/`scp` details and
   prior RunPod lessons (`project_v24_runpod_migration` memory: US datacenters only).

4. **On the pod**, from the untarred bundle root:

   ```bash
   bash runpod/spec103/install_deps.sh     # uv sync only, no HF downloads needed
   bash runpod/spec103/verify_bundle.sh    # import + manifest + bundled focused pytest
   bash runpod/spec103/smoke.sh            # ~1 min, proves the pod actually works
   bash runpod/spec103/train.sh            # the real run (v8 default, --resume-safe)
   ```

   Watch `models/spec103/runs/<run>/val_previews/` and the `noprior_l1_g` column. If it's
   stopped/preempted, re-running `train.sh` picks back up from `checkpoint_last.pt`.
   Copy `checkpoint_best.pt` + `val_previews/` back down when done.

If the run shows odd terracing/banding in the previews, the first thing to try is
`OUTPUT_HEAD_MODE=linear_unclamped_train bash runpod/spec103/train.sh` — see the banding note
in §1 below for why. Full RunPod details (env vars, what's in the bundle, size numbers): §5.

The rest of this document records the exact preparation and evidence paths.

## 0. Sanity (CPU, safe anytime)

```powershell
uv run pytest tests/spec103 -q
```

## 1. Synthetic PoC (MVP — US1)

```powershell
# 1a. author known-height tiles (fast, CPU). Prints ALL commands below with resolved paths.
uv run python scripts/spec103_make_synthetic_adts.py --output ../output/spec103/synthetic
```

Then, from `wow-viewer/` (USER runs; commands are printed by 1a with exact paths):

```powershell
# 1b. blank ADT+WDT+WDL per tile (frozen writers used as-is) — one command per tile
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -c Release -- map generate-blank ...

# 1c. patch the known heights into the blank ADTs (tiles are non-adjacent by construction,
#     so the patcher's seam stitching never mutates a pattern)
dotnet run --project tools/converter/WowViewer.Tool.Converter -c Release -- terrain-patch-adt ...

# 1d. capture per-tile renders (GPU). This is one-tile top-down orthographic capture;
#     each PNG receives a required <png>.lighting.json provenance sidecar.
dotnet run --project tools/capture/WowViewer.Tool.Capture -c Release -- render ...
```

Back in `data-harvester/`:

```powershell
# 1e. assemble the 13-channel synthetic store (CPU). Use --synthesize-minimaps to proceed
#     with the labeled hillshade fallback before a capture run exists.
uv run python scripts/spec103_build_synthetic_store.py `
    --manifest ../output/spec103/synthetic/synthetic_manifest.json `
    --minimap-dir ../output/spec103/synthetic/captures `
    --output ../output/datasets/spec103/synthetic_v1.zarr

# 1f. TRAIN (GPU — USER runs). Holds out the complete crater pattern.
uv run python scripts/train_spec103_v7.py `
    --store ../output/datasets/spec103/synthetic_v1.zarr `
    --output ../output/spec103_v7_synth_v1 `
    --val-key pattern --val-value crater `
    --epochs 60 --batch 4 --wdl-prior-dropout 0.25
```

> **§1 demoted to smoke test (2026-07-14):** procedural patterns don't replicate real terrain
> and the WDL prior trivially solves them (l1_global ≈ 0.0006 at init — prior-dominated, no
> learning signal). Use §1 only to verify the pipeline runs end-to-end; **soundness is proven in
> §3 (real data)**, which is now cheap with the v8 default. Real-terrain-derived shadow synthesis
> is the §4 / T018 lane.

The trainer defaults to **`--arch v8`** (V8LeanUNet, 6.2M params / 16.4 GFLOPs — built for
minutes-to-signal local iteration); pass `--arch v7` for the original 117M MultiChannelUNetV7
ablation. Same 13-ch contract, trestle, loss, and checkpoint layout either way; inference
resolves the arch from the checkpoint. v8's VRAM headroom allows raising `--batch` 2-4x.

**Banding investigation (2026-07-14):** `--output-head-mode {legacy_clamped, linear_unclamped_train}`
is now exposed (default `legacy_clamped`, v7-faithful). The default's tanh+hard-clamp residual
head is a likely source of v7's reported output banding (tanh saturation clusters the residual
near +-scale instead of spanning it); `linear_unclamped_train` clamps only at eval time. Cheap
to A/B on the same checkpoint layout. v8 also gets ICNR init on its PixelShuffle upsampling
(prevents a checkerboard-artifact class v7 never had). Full writeup:
[research-v8-optimization.md](research-v8-optimization.md) §6. Verified separately: no precise
numeric signal (height/WDL prior/normals) is routed through 8-bit image encoding anywhere in
this pipeline — only `minimap_rgb` is uint8, correctly, since it is the deployment image.

Watch `val_previews/` (minimap | prior | prediction | GT) and the `noprior_l1_g` column —
that is the prior-dropout robustness. The procedural run is a smoke only; T011 was superseded,
and model-quality proof belongs to the curated real-corpus run plus label-free holdout checks.

## 2. Review + label-free validation (US2)

```powershell
# 2a. inference on the held-out pattern (GPU — USER runs)
uv run python scripts/infer_spec103_v7.py `
    --store ../output/datasets/spec103/synthetic_v1.zarr `
    --checkpoint ../output/spec103_v7_synth_v1/checkpoint_best.pt `
    --val-key pattern --val-value crater `
    --output ../output/spec103_v7_synth_v1/predictions

# deployment-shaped variant: add --drop-prior (image + flat prior only)

# 2b. OBJ meshes for eyeball review (CPU)
uv run python scripts/spec103_export_mesh.py `
    --predictions ../output/spec103_v7_synth_v1/predictions `
    --store ../output/datasets/spec103/synthetic_v1.zarr `
    --output ../output/spec103_v7_synth_v1/meshes

# 2c. label-free acceptance (CPU). --gt-store adds DEV-ONLY diagnostics; never gates.
uv run python scripts/validate_spec103_labelfree.py `
    --predictions ../output/spec103_v7_synth_v1/predictions `
    --report ../output/spec103_v7_synth_v1/labelfree_report.json `
    --gt-store ../output/datasets/spec103/synthetic_v1.zarr
```

Predictions are `terrain-patch-adt`-compatible: patch them into blank ADTs (as in 1c) to view
the reconstructed terrain in WoWViewer itself.

## 3. Real clean data — curate first, then train

Spec Principle #5: height under an object is occluded in the minimap, so object tiles are
**impossible targets** and must be dropped, not learned. Curation is mandatory, not optional.

```powershell
# 3a. verify + pin the real store (CPU, read-only; V18 already pairs minimap + height)
uv run python scripts/spec103_build_real_store.py `
    --store ../output/datasets/v18/3_3_5_12340.zarr `
    --output ../output/spec103/real_store_contract.json

# 3b. CURATE + BUCKET (CPU, read-only, ~a few min). Drops object-contaminated, blank, and
#     height/normal-mismatch tiles; writes an auditable manifest + per-map/regime buckets.
#     Default --max-object-coverage 0.0 drops ANY object. Recorded V18 proof: 5134 total,
#     2650 object-free before the other gates, 2253 kept after blank/signal filtering.
uv run python scripts/spec103_curate_dataset.py `
    --store ../output/datasets/v18/3_3_5_12340.zarr `
    --output ../output/spec103/curation_v18_v1

# 3c. TRAIN on the curated set. LOCAL GPU TRAINING IS OFF (2026-07-14, overheating) --
#     use §5 (RunPod) instead. Command kept for reference / local resume only.
uv run python scripts/train_spec103_v7.py `
    --store ../output/datasets/v18/3_3_5_12340.zarr `
    --curation-manifest ../output/spec103/curation_v18_v1 `
    --output ../output/spec103_v7_real_v1 `
    --val-key map --val-value Azeroth `
    --epochs 80 --batch 8 --wdl-prior-dropout 0.25
```

Without `--curation-manifest` the trainer still drops object tiles by default
(`--max-object-coverage 0.0` — drops ANY object); pass `1.0` only for the v7-faithful keep-all
ablation. The 13-channel input contract is unchanged; only the tile *selection* changes.
Architecture: `--arch v8` (lean, default) or `--arch v7` (117M reference) — see §1f note.

### 3d. Reduce the clean set by full-map terrain-art prefab coverage (CPU; USER runs)

This consumes the existing bounded Spec 076 Azeroth/Kalimdor map-canvas proof; it does not re-mine
tile-local brushes. Broader runs use the same contract by pointing `--analysis-root` at a directory
containing every selected map canvas, region catalog, and optional prefab-membership catalog.

```powershell
uv run python scripts/spec103_curate_prefabs.py `
    --store ../output/datasets/v18/3_3_5_12340.zarr `
    --analysis-root ../output/analysis/full-map-fractal-brush-library/test_height_azeroth_kalimdor_335_v2 `
    --clean-manifest ../output/spec103/curation_v18_v1/curation_manifest.parquet `
    --val-map Azeroth `
    --output ../output/spec103/prefab_curation_v1
```

Review these before training:

```text
pattern_evidence_ledger.parquet
tile_pattern_coverage.parquet
curation_manifest.parquet
curation_summary.json
```

`curation_summary.json` records input hashes, reduction counts, parameters, and the leakage audit;
the two evidence Parquets and manifest retain family/context coverage, tileset anomalies, and
representative lineage. Do not start the next training run until the complete-map holdout and
`prefab_family_id` leakage checks are both clean.

## 4. Shadow capture lane (T018, exploratory — USER runs)

The deterministic fixed-light capture contract is Spec 102 N011-N013
(`WowViewer.Tool.ValidationCapture capture --variants ...`, objects/textures/liquids off).
Run it against the patched synthetic map from 1c and correlate shadow luminance with the
known height gradients; record findings in `research-v7-contract.md` §8.

## 5. RunPod training (2026-07-14 — the current training path)

Local GPU training is off (overheating). `scripts/package_spec103_runpod.py` builds a small,
self-contained bundle: the v8 (default) / v7 trainer code plus a **field-and-row-subsetted**
copy of the V18 store — only the 6 arrays `train_spec103_v7.py` reads
(`minimap_rgb`/`height_257`/`normal_xyz`/`liquid_mask`/`liquid_height`/`object_precise_mask`,
not the other 18 fields the full V18 schema carries), only the curation-kept tiles. The earlier
clean-only proof measured **3.2 GB full store -> 127 MB bundle (138 MB tar)** at 2253/5134 tiles;
that is historical evidence, not the target size for the new prefab-reduced corpus. All shipped
tiles pass through the trainer's `V7TileDataset`, producing finite `(13,256,256)` inputs. No
pretrained weights, no HF downloads
— v8/v7 both train from scratch.

```powershell
# 5a. build the bundle (CPU, ~1 min). Re-run with --overwrite after re-curating.
uv run python scripts/package_spec103_runpod.py `
    --store ../output/datasets/v18/3_3_5_12340.zarr `
    --curation-manifest ../output/spec103/prefab_curation_v1 `
    --output-root ../output/cloud-packages/spec103 `
    --bundle-name spec103_prefab_bundle_v1
```

Transfer `../output/cloud-packages/spec103/spec103_prefab_bundle_v1.tar` to the RunPod volume
(`runpodctl send` / `scp`; see `[[project_v24_runpod_migration]]` memory for prior transfer
lessons — US datacenters only). On the pod, from the bundle root:

```bash
tar xf spec103_prefab_bundle_v1.tar && cd spec103_prefab_bundle_v1
bash runpod/spec103/install_deps.sh    # uv sync only, no HF downloads
bash runpod/spec103/verify_bundle.sh   # import + manifest + bundled focused pytest
bash runpod/spec103/smoke.sh           # ~1 min: 2 epochs, 16 tiles, proves the pod works
bash runpod/spec103/train.sh           # the real run; env-var configurable, --resume-safe
```

## 6. Synthetic lighting profiles and canonical captures (USER runs capture/client work)

### 6a. Export exact LIT colors without inventing direction or MCSH

Run from `wow-viewer/`. Use only a staged client below
`../output/tmp/wowarchive-clients/`; choose the map's actual LIT virtual path.

```powershell
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -c Release -- `
    lit profile `
    --archive-root <staged-client-root> `
    --virtual-path World/Maps/Azeroth/lights.lit `
    --game-time 0.00,0.25,0.35,0.50,0.75 `
    --output ../output/spec103/lighting/Azeroth_lit_profile.json
```

The JSON binds the source SHA256, LIT version/layout, unique default/global clear group, required
direct/ambient/fog tracks, optional sky tracks, and every sampled time. It intentionally does not
claim a recovered sun direction or MCSH coefficient. Pass this JSON to the store builder with
`--lighting-profile ../output/spec103/lighting/Azeroth_lit_profile.json` to synthesize grouped
neutral-albedo variants; those client-derived colors force the `private_byod` rights class.

### 6b. Capture a LIT-colored terrain tile (private BYOD)

```powershell
dotnet run --project tools/capture/WowViewer.Tool.Capture -c Release -- `
    render `
    --client-root ../output/spec103/synthetic/adt_patched `
    --tile-name Spec103Synthetic_30_30 `
    --output ../output/spec103/synthetic/captures/Spec103Synthetic_30_30.png `
    --resolution 256 `
    --game-time 0.35 `
    --lighting-source lit `
    --lit-client-root <staged-client-root> `
    --lit-virtual-path World/Maps/Azeroth/lights.lit
```

The capture writes `Spec103Synthetic_30_30.png.lighting.json`. The store builder verifies its PNG
hash and source evidence. Because the colors came from a client LIT file, the resulting training
store is labeled `private_byod`; rendering does not turn it into clean synthetic data.

### 6c. Build grouped clean authored time variants

First create the analytic terrain manifest with an operator-supplied license and rights assertion:

```powershell
uv run python scripts/spec103_make_synthetic_adts.py `
    --output ../output/spec103/synthetic_licensed `
    --source-license "<SPDX-or-license-name>" `
    --source-rights-assertion "<operator-owned-or-authorized-source>"
```

After the USER runs the printed ADT commands, build grouped variants from unlit generated height:

```powershell
uv run python scripts/spec103_build_synthetic_store.py `
    --manifest ../output/spec103/synthetic_licensed/synthetic_manifest.json `
    --lighting-time 0.25 `
    --lighting-time 0.35 `
    --lighting-time 0.50 `
    --synthesize-mcsh `
    --require-licensed-synthetic `
    --output ../output/datasets/spec103/synthetic_lighting_v1.zarr
```

These variants use the explicitly authored profile and MCSH approximation, never client-exact
labels. All rows from one terrain source share `source_group_id`, and the trainer keeps that group
in one partition.

### 6d. Export and consume an exact-build Light* DBC profile (private BYOD)

Use the exact staged build and the target tile's renderer-space center. The command resolves all
five database tables through DBCD plus the bundled WoWDBDefs and writes every table/definition hash,
global/local zone weight, record join, 18 color bands, and 6 float bands.

```powershell
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -c Release -- `
    light profile `
    --archive-root <staged-client-root> `
    --build 1.12.1.5875 `
    --map-id 0 `
    --renderer-position <x,y,z> `
    --game-time normalized:0.25,normalized:0.35,normalized:0.50 `
    --output ../output/spec103/lighting/Azeroth_1.12.1.5875_light_profile.json
```

Then synthesize grouped RGB variants from owned/generated height and the recovered DBC colors:

```powershell
uv run python scripts/spec103_build_synthetic_store.py `
    --manifest ../output/spec103/synthetic_licensed/synthetic_manifest.json `
    --lighting-profile ../output/spec103/lighting/Azeroth_1.12.1.5875_light_profile.json `
    --synthesize-mcsh `
    --output ../output/datasets/spec103/synthetic_dbc_private_v1.zarr
```

The exported direct/ambient/fog colors remain exact-build evidence. Sun direction and the MCSH
strength/bake are still labeled authored gaps, and `--require-licensed-synthetic` deliberately
rejects this client-derived profile instead of mislabeling it clean synthetic.

`train.sh` env vars (all optional): `ARCH` (v8 default, v7 ablation), `EPOCHS` (80),
`BATCH` (24 — v8's VRAM headroom over v7), `LR`, `WDL_PRIOR_DROPOUT` (0.25),
`OUTPUT_HEAD_MODE` (`legacy_clamped` default; try `linear_unclamped_train` per the banding
note above), `VAL_KEY`/`VAL_VALUE` (map/Azeroth). It always passes `--resume`, so an
interrupted/preempted pod picks back up from `checkpoint_last.pt` instead of restarting.
Copy `models/spec103/runs/<run>/checkpoint_best.pt` + `val_previews/` back down when done.
