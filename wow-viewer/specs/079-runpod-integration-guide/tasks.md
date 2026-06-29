# Tasks: RunPod Integration Guide

**Input**: Design documents from `/specs/079-runpod-integration-guide/`

**Prerequisites**: `spec.md`, `plan.md`

**Tests**: Include bounded pytest for packager, setup helper, GPU filtering, cost targeting, error handling.

---

## Phase 1: Spec and Knowledge Capture (Done)

- [x] R001 [US1-US6] Write `spec.md` with 10 RunPod lessons, 6 user stories, 19 functional requirements, integration checklist.

---

## Phase 2: Generalized Packager Template

**Goal**: Extract reusable packager from Spec 077's `package_spec077_runpod.py` into a configurable template.

- [ ] R002 [US1] Define `PackagerConfig` dataclass with source paths, exclude patterns, archive format, output root.
- [ ] R003 [US1] Implement `build_bundle(config)` in `src/harvester/runpod_packager.py`.
- [ ] R004 [US1] Implement `validate_bundle(bundle_dir)` that checks for game-client file inclusion.
- [ ] R005 [US1] Generate template pod-side helpers: `install_deps.sh`, `verify_bundle.sh`, `smoke.sh`, `train.sh`.
- [ ] R006 [US1] Add pytest coverage for packager config, build, and validation.

---

## Phase 3: Generalized Setup Helper

**Goal**: Extract reusable setup helper from Spec 077's `setup_spec077_runpod.py`.

- [ ] R007 [US2, US5] Define `SetupConfig` dataclass: GPU target, cloud type, RAM/vCPU/storage, network volume, transfer method.
- [ ] R008 [US5] Implement `query_gpu_availability()` using `runpodctl gpu list` with hardcoded pricing fallback.
- [ ] R009 [US5] Implement `filter_gpus_by_cost_target(config)` — VRAM min, RAM min, cost max, datacenter-GPU exclusion, sort cheapest first.
- [ ] R010 [US2] Implement `create_pod(config)` with minimal REST payload (gpuTypeIds, gpuCount, imageName, containerDiskInGb, volumeInGb, volumeMountPath, ports, supportPublicIp).
- [ ] R011 [US2] Implement `--manual-pod` mode printing web UI URL + recommended settings.
- [ ] R012 [US2] Implement network-volume creation with concrete datacenter resolution and orphan volume cleanup on failure.
- [ ] R013 [US2] Implement error classification: `_is_retryable_error()` (500-level, "not found", "no instances") vs fatal.
- [ ] R014 [US2] Add pytest coverage for GPU filtering, cost targeting, payload construction, error handling.

---

## Phase 4: Data Transfer and Bootstrap

**Goal**: Implement automatic and manual data transfer with bootstrap training.

- [ ] R015 [US3] Implement `runpodctl send/receive` bootstrap handoff.
- [ ] R016 [US3] Implement `scp`/`rsync` fallback instruction printing.
- [ ] R017 [US4] Implement auto-start training sequence in `install_deps.sh` → `verify_bundle.sh` → `smoke.sh` → `train.sh`.
- [ ] R018 [US4] Implement `--no-auto-transfer` and `--no-auto-start-training` flags.
- [ ] R019 [US3] Print SSH connection string and `scp` commands after Pod creation.
- [ ] R020 [US3] Document SSH key setup (my.runpod.io → Settings → SSH Keys).

---

## Phase 5: Serverless Migration Guide (P3)

**Goal**: Document Serverless migration path for stable training commands.

- [ ] R021 [US6] Write Serverless handler template wrapping the training script.
- [ ] R022 [US6] Document network-volume attachment for Serverless endpoints.
- [ ] R023 [US6] Document Flash (`@Endpoint`) integration.
- [ ] R024 [US6] Add Serverless migration checklist to `spec.md`.

---

## Phase 6: Docs and Handoff

**Goal**: Make the integration easy for a new project to adopt.

- [ ] R025 [US1-US6] Write `quickstart.md` with minimal "copy template, change 4 paths, run 2 commands" flow.
- [ ] R026 [US1-US6] Create template directory `wow-viewer/data-harvester/templates/runpod/` with placeholder config files.
- [ ] R027 [US1-US6] Update `plan.md` and `tasks.md` with final phase counts and status.

---

## Dependencies

- Phase 1 (spec) blocks all implementation phases.
- Phase 2 (packager) can parallel with Phase 3 (setup helper) but both share error handling patterns.
- Phase 4 (transfer/bootstrap) depends on Phase 3 (Pod creation).
- Phase 5 (Serverless) depends on Phase 2-4 code existing.
- Phase 6 (docs) is last.
