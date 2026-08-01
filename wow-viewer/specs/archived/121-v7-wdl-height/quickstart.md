# Quickstart: V7-Style WDL-Prior Height Reconstruction (Small Model Lane)

**Created**: 2026-07-24
**Spec**: [spec.md](spec.md) · **Plan**: [plan.md](plan.md) · **CLIs**: [contracts/cli-contract.md](contracts/cli-contract.md)

All commands are PowerShell-ready, run from `i:\parp\parp-tools`. Training is user-run (RULE 0).
Substitute your store/split/output paths; nothing here hardcodes a client root.

## 0. Prerequisite (data, user-run)

This lane's substrate is the **v50.2** store release: the v50.1 signals PLUS the Spec 117 WDL
lattice arrays and the Spec 118 object-mask arrays. The store must contain `wdl_outer_17`,
`wdl_inner_16`, `wdl_outer_present`, `wdl_inner_present`, and
`object_geometry_visible_mask_257`. Verify (read-only):

```powershell
cd wow-viewer/data-harvester; uv run python -c "import zarr; g=zarr.open_group(r'PATH/TO/v50_store.zarr', mode='r'); print(sorted(g.array_keys()))"
```

If the Spec 118 arrays are missing, rebuild the store (same Spec 109 build command, Full profile)
before any weighted run. Unweighted runs work without them.

## 1. Stage A — minimap → WDL lattice prior (user-run)

Dry-run first (prints params, split counts, baseline, schedule; exits without training):

```powershell
cd wow-viewer/data-harvester; uv run python scripts/spec121_train_lattice_prior.py --store PATH/TO/v50_store.zarr --held-out-split PATH/TO/split.json --output PATH/TO/runs/lattice-mit_b0-v1 --run-id lattice-mit_b0-v1 --architecture mit_b0_lattice
```

Train (user presses go):

```powershell
cd wow-viewer/data-harvester; uv run python scripts/spec121_train_lattice_prior.py --store PATH/TO/v50_store.zarr --held-out-split PATH/TO/split.json --output PATH/TO/runs/lattice-mit_b0-v1 --run-id lattice-mit_b0-v1 --architecture mit_b0_lattice --confirm-run
```

**Gate G1 (SC-001)**: held-out lattice MAE ≥ 15% below tile-mean baseline in the run record.
Fail → stop; record the negative result; do not train Stage B. Check the per-epoch previews to
classify underfit vs no-signal.

From-scratch fallback (known Spec 117 plateau risk, cheap sanity check):

```powershell
cd wow-viewer/data-harvester; uv run python scripts/spec121_train_lattice_prior.py --store PATH/TO/v50_store.zarr --held-out-split PATH/TO/split.json --output PATH/TO/runs/lattice-net-v2-rerun --run-id lattice-net-v2-rerun --architecture lattice_net_v2 --confirm-run
```

## 2. Bridge — predicted prior → coarse store

```powershell
cd wow-viewer/data-harvester; uv run python scripts/spec121_bridge_prior_to_coarse.py --store PATH/TO/v50_store.zarr --checkpoint PATH/TO/runs/lattice-mit_b0-v1/lattice_prior.pt --output PATH/TO/runs/prior_coarse_store --write
```

## 3. Stage B — residual detailer over the prior (user-run)

Dry-run, then train (existing trainer, zero code-path surprises):

```powershell
cd wow-viewer/data-harvester; uv run python scripts/v50_train_geometry_detailer.py --store PATH/TO/v50_store.zarr --coarse-store PATH/TO/runs/prior_coarse_store --held-out-split PATH/TO/split.json --output PATH/TO/runs/detailer-prior-v1 --run-id detailer-prior-v1 --architecture detailer_mit_b0_v1 --confirm-run
```

**Gate G2 (SC-002)**: held-out MAE ≥ 9% below the prior-only baseline (upsampled prior, no
detailer); GT-prior ablation reported as upper bound.

## 4. Paired object-mask-loss comparison (user-run, SC-003)

Same commands as §1 and §3 with `--object-mask-weight 1.0` added, distinct `--run-id` /
`--output`. Compare run records: object-touched vs untouched MAE. Verdict options: helps /
hurts / null — null closes the question and is a valid outcome.

## 5. End-to-end chain + visual gate (SC-005)

```powershell
cd wow-viewer/data-harvester; uv run python scripts/spec121_materialize_chain.py --stage-a-checkpoint PATH/TO/runs/lattice-mit_b0-v1/lattice_prior.pt --stage-b-checkpoint PATH/TO/runs/detailer-prior-v1/geometry_detailer.pt --store PATH/TO/v50_store.zarr --output PATH/TO/runs/chain_sheets --write
```

OOD hand-painted tile:

```powershell
cd wow-viewer/data-harvester; uv run python scripts/spec121_materialize_chain.py --stage-a-checkpoint PATH/TO/runs/lattice-mit_b0-v1/lattice_prior.pt --stage-b-checkpoint PATH/TO/runs/detailer-prior-v1/geometry_detailer.pt --inputs PATH/TO/hand_painted.png --output PATH/TO/runs/chain_ood --write
```

Review fixed-row + worst-case sheets; issue the visual verdict; only then flip
`promotion_verdict` in the run records.

## 6. Verdict recording

Append outcomes to `research.md` (R-1 etc.) and the run-record JSONs. Negative results are
recorded, not hidden — this lane exists because unrecorded negatives (119/120) wasted a month.
