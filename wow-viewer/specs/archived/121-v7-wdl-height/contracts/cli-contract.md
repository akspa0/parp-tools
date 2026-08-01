# CLI Contract: V7-Style WDL-Prior Height Reconstruction (Small Model Lane)

**Created**: 2026-07-24
**Spec**: [spec.md](../spec.md) · **Plan**: [plan.md](../plan.md)

All commands run from `wow-viewer/data-harvester` via `uv run python <script>`. Every CLI is
dry-run-first: without the explicit run flag it prints the full plan and exits 0 without training
or writing. All training launches are user-run (RULE 0).

## New CLIs

### `scripts/spec121_train_lattice_prior.py` — Stage A trainer (FR-001/003/004/005/007/008/009/011/012)

| Flag | Required | Default | Notes |
|---|---|---|---|
| `--store PATH` | yes | — | v50 Full-profile Zarr store |
| `--held-out-split PATH` | yes | — | Spec 116 frozen split; no fallback |
| `--output PATH` | yes | — | run directory |
| `--run-id STR` | yes | — | immutable run identity, e.g. `lattice-mit_b0-v1` |
| `--architecture {lattice_net_v2,mit_b0_lattice}` | yes | — | fallback vs backbone |
| `--base INT` | no | arch default | from-scratch width (`lattice_net_v2`) |
| `--pretrained-hub-id STR` | no | `nvidia/mit-b0` | `mit_b0_lattice` only |
| `--pretrained-revision STR` | no | `None` | recorded in run record |
| `--no-pretrained` | no | off | from-scratch B0 encoder |
| `--object-mask-weight FLOAT` | no | `0.0` | 0 = parity; tile-coverage scale (D-05) |
| `--gradient-weight FLOAT` | no | `0.0` | V7 2D gradient term (Spec 117 port) |
| `--pct-start FLOAT` | no | `0.1` | onecycle warmup fraction |
| `--epochs INT` | no | `100` | early stop via warmup-aware stale counter |
| `--batch-size INT` | no | `16` | |
| `--confirm-run` | no | off | REQUIRED to actually train |

Run record: `v50-model-stage-run-v1`, stage `"lattice_prior"`, includes tile-mean baseline MAE,
per-class (outer/inner) MAE, `object_mask_weight`, `object_mask_signal_present`, backbone hub id +
revision (or null), `promotion_verdict="pending"`.

### `scripts/spec121_bridge_prior_to_coarse.py` — Stage A → coarse-store bridge (FR-010/014)

| Flag | Required | Default | Notes |
|---|---|---|---|
| `--store PATH` | yes | — | v50 store (row alignment source) |
| `--checkpoint PATH` | yes | — | frozen Stage A checkpoint |
| `--output PATH` | yes | — | new coarse store directory |
| `--write` | no | off | REQUIRED to write; default is dry-run report |

Output: materialized coarse store byte-compatible with
`geometry_detailer_train.validate_coarse_store`; attrs carry the Stage A checkpoint sha256.

### `scripts/spec121_materialize_chain.py` — deployment chain (FR-014)

| Flag | Required | Default | Notes |
|---|---|---|---|
| `--stage-a-checkpoint PATH` | yes | — | frozen Stage A |
| `--stage-b-checkpoint PATH` | yes | — | frozen Stage B |
| `--store PATH` | no | — | batch mode over store rows |
| `--inputs PATH [PATH ...]` | no | — | loose minimap images (OOD mode; mutually exclusive with `--store`) |
| `--output PATH` | yes | — | sheets + audit JSON |
| `--write` | no | off | dry-run prints inference plan |

Guarantee: reads no ground-truth WDL/height/mask array in either mode; audit JSON names both
checkpoint sha256s.

## Reused CLIs (unchanged surface except where noted)

### `scripts/v50_train_geometry_detailer.py` — Stage B trainer

Consumed exactly as today, with `--coarse-store` pointing at the bridge output. ONE bounded
addition: `--architecture {detailer_unet_v1,detailer_mit_b0_v1}` (default `detailer_unet_v1` =
parity). Existing flags already satisfy the lane: `--object-mask-weight` (Spec 118),
`--feature-store` (repeatable), `--held-out-split`, spectral loss flags (default 0), `--pct-start`.

### `scripts/v50_materialize_coarse_relief.py`

Unchanged; remains the coarse-only baseline producer for the SC-002 comparison (prior-only
baseline comes from the bridge store directly).

## Exit / Refusal Rules

- Missing `--held-out-split` → exit 2 with message (all trainers).
- Split leakage check failure → exit 2 (`verified_violation_count != 0`).
- Missing `object_geometry_visible_mask_257` with `--object-mask-weight > 0` → warn, disable
  weighting, record `object_mask_signal_present=false`, continue (never crash).
- Unknown `--architecture` / config that hashes to a different identity than the checkpoint at
  load time → exit 2.
- Param count outside 3–30M for `mit_b0_lattice` / `detailer_mit_b0_v1` → dry-run plan flags the
  violation; `--confirm-run` refuses.
