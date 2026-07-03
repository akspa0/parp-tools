# Contract: V23 RunPod Bundle

**Phase 1 contract. Companion to `plan.md`, `research.md`, and `data-model.md`.**

This contract defines the V23-specific bundle boundary on top of Spec 079's generalized RunPod workflow.

## Required Contents

- `src/harvester/v23/`
- `scripts/train_v23_height.py`
- `scripts/infer_v23_height.py`
- `scripts/package_v23_runpod.py`
- `runpod/v23/install_deps.sh`
- `runpod/v23/verify_bundle.sh`
- `runpod/v23/smoke.sh`
- `runpod/v23/train.sh`
- `pyproject.toml`
- `uv.lock`
- bounded V22 subset
- `manifest.json`

## Manifest Fields

| Field | Required | Meaning |
|---|---|---|
| `bundle_name` | yes | archive identifier |
| `source_spec` | yes | `089-dav2-height-predictor` |
| `contains_game_client_files` | yes | must be `false` |
| `dataset_subset` | yes | included builds / tile counts |
| `tree_hash` | yes | bundle content hash |
| `created_at` | yes | bundle timestamp |

## Forbidden Contents

- any staged client root
- any WoWArchive mount content
- any path under `output/tmp/wowarchive-clients/`
- raw game client files

## Producer

- `scripts/package_v23_runpod.py`

## Consumer

- Spec 079 Pod/bootstrap flow
- `runpod/v23/install_deps.sh`
- `runpod/v23/verify_bundle.sh`
- `runpod/v23/smoke.sh`
- `runpod/v23/train.sh`
