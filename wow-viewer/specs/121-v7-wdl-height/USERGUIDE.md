# User Guide: Spec 121 — V7-Style WDL-Prior Height Reconstruction

**Audience**: the operator (you). Every command is PowerShell-ready and user-run (RULE 0).
**Working directory for all commands**: `i:\parp\parp-tools\wow-viewer\data-harvester`

```powershell
cd i:\parp\parp-tools\wow-viewer\data-harvester
```

## What this lane is

Two small models, trained separately, chained at inference:

1. **Stage A** — `MitB0LatticeNet` (SegFormer-B0 backbone, 3.47M params): minimap RGB →
   545-point WDL height lattice (the low-res prior v7 needed).
2. **Stage B** — existing residual detailer: minimap + predicted prior → final 257×257 height.

Precise object masks (`object_geometry_visible_mask_257`) are a **loss weight only**. Nothing in
this lane segments, classifies, detects, or retrieves objects on minimaps — that line is dead
(Spec 119's own measurement killed it: object instances are 5–29px blobs).

**Naming**: dataset release is **v50.2** (v50.1 signals + WDL lattice + object-mask arrays).
Model ids: `mit_b0_lattice`, `detailer_mit_b0_v1`. Nothing here is called v24/v25; those lanes
are dead and untouched.

---

## Phase 0 — Verify what you already have (read-only, ~30s)

The existing v50.1 store (`curriculum-0_5_3_3368-dual_v3.zarr`) already has the lattice arrays
but NOT the object-mask arrays. Confirm:

```powershell
uv run python -c "import zarr; g=zarr.open_group(r'../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v3.zarr', mode='r'); print(sorted(g.array_keys()))"
```

- `wdl_outer_17`, `wdl_inner_16`, `wdl_outer_present`, `wdl_inner_present` → present (good).
- `object_geometry_visible_mask_257` → absent in v50.1 (expected; comes with the v50.2 rebuild).

**Decision**: you can start Stage A training TODAY on v50.1 (unweighted, Phase 2). The v50.2
rebuild (Phase 1) is only required for the object-mask-weighted comparison runs (Phase 4).

---

## Phase 1 — Build the v50.2 store (user-run, ~10–20 min per map; only needed for mask-weighted runs)

Same Spec 109 build command, but `--stream-profile full` (the v22 profile OMITS the strict
object arrays — this is the one flag that matters). Per map (Kalimdor, then Azeroth):

Dry-run first (prints plan, writes nothing):

```powershell
uv run python scripts/v50_build_dataset.py build --harvest-project ../tools/harvest/WowViewer.Tool.Harvest --clients-root H:\CLIENTS --map Kalimdor --stream-profile full --signals-config ./v50_configs/v50-signals-0_5_3_3368.json --manifest-template ./v50_configs/v50-manifest-template-0_5_3_3368.json --report ../output/reports/v50/v50.2/build-0_5_3_3368-Kalimdor.json --write-store ../output/datasets/v50/v50.2/0_5_3_3368-Kalimdor.zarr
```

Then the real extraction (append `--confirm-run` + `--write-manifest`):

```powershell
uv run python scripts/v50_build_dataset.py build --harvest-project ../tools/harvest/WowViewer.Tool.Harvest --clients-root H:\CLIENTS --map Kalimdor --stream-profile full --signals-config ./v50_configs/v50-signals-0_5_3_3368.json --manifest-template ./v50_configs/v50-manifest-template-0_5_3_3368.json --report ../output/reports/v50/v50.2/build-0_5_3_3368-Kalimdor.json --write-store ../output/datasets/v50/v50.2/0_5_3_3368-Kalimdor.zarr --write-manifest ../output/reports/v50/v50.2/build-manifest-0_5_3_3368-Kalimdor.json --confirm-run
```

Finalize + verify (repeat for Azeroth with `--map Azeroth` and matching paths):

```powershell
uv run python scripts/v50_build_dataset.py finalize --store ../output/datasets/v50/v50.2/0_5_3_3368-Kalimdor.zarr --manifest ../output/reports/v50/v50.2/build-manifest-0_5_3_3368-Kalimdor.json --row-lineages ../output/reports/v50/v50.2/build-0_5_3_3368-Kalimdor.json
```

Merge into the dual curriculum store (same pattern that produced dual_v3):

```powershell
uv run python scripts/v50_build_training_curriculum.py --store ../output/datasets/v50/v50.2/0_5_3_3368-Kalimdor.zarr --store ../output/datasets/v50/v50.2/0_5_3_3368-Azeroth.zarr --curation-manifest ../output/datasets/v50/v50.1/curation-0_5_3_3368-Kalimdor-terrain-v2 --curation-manifest ../output/datasets/v50/v50.1/curation-0_5_3_3368-Azeroth-terrain-v2 --output ../output/datasets/v50/v50.2/curriculum-0_5_3_3368-dual_v502.zarr --val-fraction 0.15
```

Build a fresh Spec 116 split against the new store (row identities differ from dual_v3 — do NOT
reuse the old split), dry-run then `--write`:

```powershell
uv run python scripts/spec116_build_held_out_split.py --store ../output/datasets/v50/v50.2/curriculum-0_5_3_3368-dual_v502.zarr
uv run python scripts/spec116_build_held_out_split.py --store ../output/datasets/v50/v50.2/curriculum-0_5_3_3368-dual_v502.zarr --output ../output/datasets/spec116/spec116-held-out-0_5_3_3368-dual_v502 --write
```

Verify the split manifest reports `verified_violation_count: 0`.

---

## Phase 2 — Stage A training (user-run; works TODAY on v50.1)

Dry-run first (prints plan incl. param count, split counts, baselines; trains nothing):

```powershell
uv run python scripts/spec121_train_lattice_prior.py --store ../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v3.zarr --held-out-split ../output/datasets/spec116/spec116-held-out-0_5_3_3368-dual_v2 --output ../output/runs/lattice-mit_b0-v1 --run-id lattice-mit_b0-v1 --source authored --release v50.1 --architecture mit_b0_lattice
```

Real training (~679 train rows, 43 steps/epoch, up to 100 epochs, warmup-aware early stop;
`--pretrained` pulls the `nvidia/mit-b0` encoder once from HuggingFace):

```powershell
uv run python scripts/spec121_train_lattice_prior.py --store ../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v3.zarr --held-out-split ../output/datasets/spec116/spec116-held-out-0_5_3_3368-dual_v2 --output ../output/runs/lattice-mit_b0-v1 --run-id lattice-mit_b0-v1 --source authored --release v50.1 --architecture mit_b0_lattice --pretrained --lr-schedule onecycle --pct-start 0.1 --confirm-run
```

Optional sanity rerun of the from-scratch fallback (cheap; Spec 117 plateau risk is known):

```powershell
uv run python scripts/spec121_train_lattice_prior.py --store ../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v3.zarr --held-out-split ../output/datasets/spec116/spec116-held-out-0_5_3_3368-dual_v2 --output ../output/runs/lattice-net-v5-rerun --run-id lattice-net-v5-rerun --source authored --release v50.1 --architecture lattice_net --base 64 --lr-schedule onecycle --pct-start 0.1 --confirm-run
```

### Phase 2 verdict (G1 gate, SC-001)

```powershell
uv run python -c "import json; m=json.load(open(r'../output/runs/lattice-mit_b0-v1/model_stage_run.json'))['metrics']; print('sc001_pass:', m['sc001_pass'], '| margin:', round(m['sc001_margin_vs_tile_mean'],4), '| best_val_mae:', round(m['best_val_mae'],6), '| tile_mean:', round(json.load(open(r'../output/runs/lattice-mit_b0-v1/model_stage_run.json'))['baselines']['tile_mean']['val_mae'],6))"
```

- `sc001_pass: True` (≥15% below tile-mean) → lane is alive. Tell the agent; Phase 3 (bridge +
  detailer, T014–T020) gets implemented.
- `sc001_pass: False` → open `../output/runs/lattice-mit_b0-v1/validation/final_best/fixed_rows.png`
  and `worst_cases.png`. Previews distinguish underfit (blurry but terrain-shaped) from no-signal
  (noise). Either way the negative gets recorded in `research.md` — that is a valid, cheap stop.

---

## Phase 3 — Bridge + Stage B (after G1 pass; agent implements T014–T020)

Not your commands yet. Once the bridge exists, this section will hold:

1. `spec121_bridge_prior_to_coarse.py` — frozen Stage A checkpoint → coarse store.
2. `v50_train_geometry_detailer.py --coarse-store <bridge output>` — the EXISTING detailer
   trainer, unchanged, with `--architecture detailer_mit_b0_v1` as the band-compliant trunk.
3. G2 gate: ≥9% below the prior-only baseline + ground-truth-prior ablation.

---

## Phase 4 — Object-mask-weighted comparison (needs the v50.2 store from Phase 1)

Same as Phase 2, against the v50.2 store + new split, plus `--object-mask-weight 1.0` and
`--release v50.2`:

```powershell
uv run python scripts/spec121_train_lattice_prior.py --store ../output/datasets/v50/v50.2/curriculum-0_5_3_3368-dual_v502.zarr --held-out-split ../output/datasets/spec116/spec116-held-out-0_5_3_3368-dual_v502 --output ../output/runs/lattice-mit_b0-masked-v1 --run-id lattice-mit_b0-masked-v1 --source authored --release v50.2 --architecture mit_b0_lattice --pretrained --lr-schedule onecycle --pct-start 0.1 --object-mask-weight 1.0 --confirm-run
```

Also run the unweighted twin on v50.2 (same command minus `--object-mask-weight`, different
`--run-id`/`--output`) so the pair differs ONLY in the weight. Compare
`metrics.object_touched_split` in both run records. Verdicts: helps / hurts / **null** — null
closes the question and is a valid outcome (SC-003).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `store is missing required arrays [...]` | store predates the lattice/object catalog amendments | Phase 1 rebuild (v50.2) |
| `[object-mask] store lacks object_geometry_visible_mask_257; disabling` | weighted run against v50.1 | expected; run Phase 1 first, or accept unweighted |
| `refusing --confirm-run ... outside the 3-30M band` | config drift changed param count | report to agent; do not bypass |
| HF download fails (offline) | no network | drop `--pretrained` (from-scratch B0; Spec 117 showed from-scratch is weaker — treat as diagnostic) |
| `verified_violation_count != 0` | leaky split | never override; rebuild split (Phase 1 step) |
| `CUDA is not available` | wrong machine/env | training is GPU-only by design |

## What gets written where

| Artifact | Path |
|---|---|
| Stage A run dir | `wow-viewer/output/runs/<run-id>/` (checkpoints, `model_stage_run.json`, preview sheets) |
| v50.2 stores | `wow-viewer/output/datasets/v50/v50.2/` |
| v50.2 split | `wow-viewer/output/datasets/spec116/spec116-held-out-0_5_3_3368-dual_v502/` |
| Verdict log | `wow-viewer/specs/121-v7-wdl-height/research.md` (agent updates after each gate) |
