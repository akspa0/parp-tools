# Implementation Plan: RunPod Integration Guide

**Branch**: `079-runpod-integration-guide` | **Date**: 2026-06-29 | **Spec**: `wow-viewer/specs/079-runpod-integration-guide/spec.md`

**Input**: Feature specification from `/specs/079-runpod-integration-guide/spec.md`

## Summary

Package the hard-won RunPod integration lessons from Spec 077 into a reusable template and companion spec so future projects can bootstrap cloud GPU training without repeating the same discovery cycle. The plan covers packager design, setup helper architecture, GPU availability/filtering, data transfer bootstrap, SSH setup, cost targeting, and Serverless migration path.

## Technical Context

**Language/Version**: Python 3.11+ / `uv` for the packager and setup helper

**Primary Dependencies**:
- `runpodctl` v2.x (downloaded separately, not pip)
- `RUNPOD_API_KEY` environment variable
- RunPod REST API at `https://rest.runpod.io/v1`

**Storage**:
- Network volumes for persistent training artifacts
- Pod-local volumes as fallback

**Testing**:
- `pytest` for packager/setup helper unit tests
- Focused `--validate-only` and `--dry-run` modes for safe smoke testing

**Target Platform**:
- Windows workstation (setup/packaging host)
- Linux CUDA (RunPod Pod, training host)

## Constitution Check

- Repo independence: pass. All code lives in `wow-viewer/data-harvester/scripts/`.
- Library-first: pass. The packager and setup helper are scripts, not deep library modules; spec captures cross-project knowledge.
- Real-data validation: pass. Spec 077 real RunPod Pod creation validated the approach.
- No untrusted client paths: pass. Packager explicitly excludes game-client data.

## Phases

### Phase 1 - Spec and Knowledge Capture (Done)

1. Write `spec.md` documenting all 10 RunPod lessons learned from Spec 077.
2. Document 6 user stories with acceptance scenarios (package, create, transfer, bootstrap, availability query, Serverless migration).
3. Define 19 functional requirements (FR-001 through FR-019).
4. Provide a reusable integration checklist for any project.

**Validation**: Spec exists with all lessons, user stories, FRs, and checklist. Done.

### Phase 2 - Generalized Packager Template

1. Extract the common packager logic from `package_spec077_runpod.py` into a reusable module under `src/harvester/`.
2. Define a `PackagerConfig` dataclass with: source paths, exclude patterns, archive format, output root, run name.
3. Implement `build_bundle(config)` that copies only derived artifacts, writes `manifest.json`, `README.md`, `requirements.txt`, and pod-side helpers (`install_deps.sh`, `verify_bundle.sh`, `smoke.sh`, `train.sh`).
4. Implement `validate_bundle(bundle_dir)` that checks for game-client file inclusion.
5. Add pytest coverage for the generalized packager.

**Validation**: The generalized packager can replace `package_spec077_runpod.py` for Spec 077 and a different project with a different config.

### Phase 3 - Generalized Setup Helper

1. Extract common setup logic from `setup_spec077_runpod.py` into a reusable module.
2. Define `SetupConfig` with: GPU target (type list or cost-target params), cloud type, RAM/vCPU/storage, network volume settings, transfer method, auto-start flag.
3. Implement GPU availability query (`runpodctl gpu list` with hardcoded pricing fallback).
4. Implement cost-target filtering (VRAM min, RAM min, cost max, datacenter-GPU exclusion).
5. Implement Pod creation with minimal REST payload (proven pattern from Spec 077).
6. Implement `--manual-pod` mode (print web UI URL + settings).
7. Implement network-volume creation with concrete datacenter resolution.
8. Implement cleanup of orphaned volumes after failed Pod creation.
9. Add pytest coverage for setup config, GPU filtering, cost targeting, and error handling.

**Validation**: The generalized setup helper can create a RunPod Pod with either GPU-type or cost-target params, with network volume or Pod-local storage.

### Phase 4 - Data Transfer and Bootstrap

1. Implement `runpodctl send/receive` bootstrap that starts Pod-side `receive` before local `send`.
2. Implement `scp`/`rsync` fallback instruction printing when `runpodctl` is not on PATH.
3. Implement SSH key setup documentation (printed after Pod creation).
4. Implement auto-start training sequence: `install_deps.sh` → `verify_bundle.sh` → `smoke.sh` → `train.sh`.
5. Implement `--no-auto-transfer`, `--no-auto-start-training` flags.
6. Add bootstrap script templates with `set -euo pipefail` and graceful failure paths.

**Validation**: A Pod created with the setup helper receives the bundle and starts training automatically (or prints instructions for manual steps).

### Phase 5 - Serverless Migration Guide (P3)

1. Document how to wrap a stable training command in a Serverless handler.
2. Document network-volume attachment for Serverless endpoints.
3. Document Flash (`@Endpoint`) integration for local-style function calls.
4. Add migration checklist to `spec.md`.

**Validation**: A developer can follow the guide to migrate a Pod-based training command to Serverless in under 15 minutes.

### Phase 6 - Docs and Handoff (P2)

1. Write `quickstart.md` with a minimal "copy this template, change 4 paths, run 2 commands" flow.
2. Update `plan.md` and `tasks.md` with final phase counts and status.
3. Create a reusable template directory under `wow-viewer/data-harvester/templates/runpod/` with placeholder config files.

**Validation**: A new project operator can get training running on RunPod in under 30 minutes using the template.

## Complexity Tracking

No constitution violations. The spec captures cross-project knowledge without creating repo-dependency issues. The packager and setup helper are extracted from working Spec 077 code and made generic.
