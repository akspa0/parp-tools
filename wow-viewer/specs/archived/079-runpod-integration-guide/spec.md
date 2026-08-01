# Feature Specification: RunPod Integration Guide

**Feature Branch**: `079-runpod-integration-guide`

**Created**: 2026-06-29

**Status**: Draft

**Input**: User description: "Write a generalized spec for RunPod integration patterns so future projects can bootstrap cloud GPU training without reinventing the wheel. Covers Pod creation, data transfer, SSH setup, bootstrap scripts, cost targeting, and Serverless migration path."

## Problem Statement

Every project that needs cloud GPU training reinvents the same RunPod integration: packaging derived data, creating Pods, transferring archives, setting up SSH, writing bootstrap scripts, and debugging availability. This spec captures the hard-won lessons from Spec 077's RunPod integration so the next project starts from a proven baseline instead of repeating the same mistakes.

## Key Lessons Learned (from Spec 077)

1. **`runpodctl gpu list` does NOT expose pricing.** The output has `memoryInGb`, `available`, `communityCloud`, `secureCloud`, `stockStatus` — but no price fields. Cost filtering must use hardcoded approximate prices or the RunPod web UI.

2. **`runpodctl datacenter list` returns datacenters that do NOT support network volumes.** Not all datacenters that list GPU availability can create network volumes. The valid set must be filtered against the RunPod API error message or a known-good set.

3. **`runpodctl send/receive` relay is fragile.** If the local `send` doesn't start (e.g., `runpodctl` not on PATH), the Pod-side `receive` gets a "Malformed relay" error and the bootstrap script exits. The Pod container may restart but the relay won't reconnect.

4. **SSH requires key registration via the RunPod web console.** `runpodctl ssh info <pod-id>` shows SSH info but does NOT set up keys. Keys must be added at my.runpod.io → Settings → SSH Keys before SSH works.

5. **Pod creation via REST API can fail even when the web UI shows availability.** The REST API `POST /pods` may return "no instances currently available" for GPU/datacenter combos that the web UI shows as available. This may be due to stale `runpodctl datacenter list` data, cloud type differences, or payload field issues.

6. **Network volume + Pod must share the same datacenter.** Creating a network volume in one datacenter and a Pod in another fails. The script must resolve a concrete datacenter before creating either resource.

7. **`runpodctl` v2.x uses positional args, not flags.** `runpodctl ssh info <pod-id>`, not `runpodctl ssh --pod <pod-id>`. Legacy flag-based commands are deprecated.

8. **COMMUNITY cloud has more availability and lower prices than SECURE cloud.** Default to COMMUNITY unless reliability is critical.

9. **The web UI is the most reliable Pod creation path.** When the REST API fails, creating a Pod from my.runpod.io always works. The integration should support a "manual Pod" fallback.

10. **`--include-unavailable` flag may not exist in runpodctl v2.x.** Calling `_runpodctl_json(["gpu", "list", "--include-unavailable"])` may fail silently. Use `["gpu", "list"]` without extra flags.

## User Scenarios & Testing

### User Story 1 - Package Derived Data for Cloud Training (Priority: P1)

As a project operator, I want to package only derived training artifacts (no source data, no game clients, no raw assets) into a transferable archive, so I can train on RunPod without redistributing proprietary data.

**Acceptance Scenarios**:

1. **Given** a project with local training code and derived datasets, **When** the packager runs, **Then** it produces a `.tar` archive containing only Python code, Zarr/NPZ datasets, manifests, and pod-side helper scripts.
2. **Given** the produced archive, **When** an auditor checks its manifest, **Then** `contains_game_client_files` is `false` and no path under the bundle resolves to source/client data.
3. **Given** the archive, **When** it is unpacked on a Pod, **Then** `install_deps.sh`, `verify_bundle.sh`, `smoke.sh`, and `train.sh` exist and are executable.

### User Story 2 - Create a RunPod Pod (Priority: P1)

As a project operator, I want to create a RunPod Pod with the right GPU/RAM/storage for my training job, so I can run training in the cloud.

**Acceptance Scenarios**:

1. **Given** `RUNPOD_API_KEY` is set, **When** the setup helper runs, **Then** it creates a Pod with the requested GPU type, RAM, vCPU, and storage.
2. **Given** the web UI shows GPU availability but the REST API fails, **When** the operator passes `--manual-pod`, **Then** the helper prints the exact web UI URL and recommended Pod settings instead of calling the API.
3. **Given** a Pod is created, **When** the operator queries SSH info, **Then** the helper waits for a public IP and prints the SSH connection string.

### User Story 3 - Transfer Data to the Pod (Priority: P1)

As a project operator, I want to transfer the training archive to the Pod with minimal friction, so I can start training quickly.

**Acceptance Scenarios**:

1. **Given** `runpodctl` is installed locally and the Pod is running, **When** the setup helper runs with auto-transfer, **Then** the Pod-side bootstrap runs `runpodctl receive <code>` and the local helper runs `runpodctl send <archive> --code <code>`.
2. **Given** `runpodctl` is NOT installed locally, **When** the operator needs to transfer, **Then** the helper prints `scp` and `rsync` commands with the Pod's IP and port pre-filled.
3. **Given** SSH is configured, **When** the operator runs `scp -P <port> <archive> root@<ip>:/workspace/`, **Then** the archive transfers successfully.

### User Story 4 - Bootstrap Training on the Pod (Priority: P1)

As a project operator, I want the Pod to automatically install dependencies, verify the bundle, and start training after the archive is unpacked, so I don't have to SSH in and run commands manually.

**Acceptance Scenarios**:

1. **Given** auto-start is enabled and the archive is unpacked, **When** the bootstrap script runs, **Then** it executes `install_deps.sh`, `verify_bundle.sh`, `smoke.sh`, and `train.sh` in sequence.
2. **Given** auto-start is disabled, **When** the bootstrap script runs, **Then** it stops after verify and prints instructions for manual training start.

### User Story 5 - Query GPU Availability and Filter by Criteria (Priority: P2)

As a project operator, I want to see which GPUs are available and filter by VRAM, cloud type, and cost, so I can pick the cheapest suitable GPU.

**Acceptance Scenarios**:

1. **Given** `runpodctl` is installed, **When** the helper queries `runpodctl gpu list`, **Then** it displays each GPU with VRAM, community/secure cloud support, and stock status.
2. **Given** a cost target and minimum VRAM, **When** the helper filters GPUs, **Then** it excludes datacenter cards (A100, H100, H200, B200, L4, L40, A40, Tesla, RTX PRO) and sorts remaining GPUs by approximate price.
3. **Given** `runpodctl` is NOT installed, **When** the helper needs GPU info, **Then** it falls back to a hardcoded GPU list with approximate pricing.

### User Story 6 - Migrate to Serverless (Priority: P3)

As a project operator, I want to migrate from Pod-based training to Serverless endpoints once the training command is stable, so I can automate repeatable training/inference jobs.

**Acceptance Scenarios**:

1. **Given** a stable training command on a Pod, **When** the operator wraps it in a Serverless handler, **Then** the handler reads the unpacked bundle from a network volume at `/runpod-volume/` and runs the same training script.
2. **Given** a Serverless endpoint, **When** the operator triggers it, **Then** it runs training/inference and writes outputs to the network volume.

## Requirements

### Functional Requirements

- **FR-001**: The integration MUST provide a packager that copies only derived training artifacts and Python code, excluding all source/client/raw data.
- **FR-002**: The packager MUST emit `manifest.json`, `README.md`, `requirements.txt`, and pod-side helper scripts (`install_deps.sh`, `verify_bundle.sh`, `smoke.sh`, `train.sh`).
- **FR-003**: The setup helper MUST accept `RUNPOD_API_KEY` for Pod/network-volume creation and MUST NOT store the key in any manifest or bundle.
- **FR-004**: The setup helper MUST support a `--manual-pod` mode that prints web UI URL and recommended settings without calling the REST API.
- **FR-005**: The setup helper MUST query `runpodctl gpu list` (without `--include-unavailable`) and parse `gpuId`, `memoryInGb`, `available`, `communityCloud`, `stockStatus`.
- **FR-006**: The setup helper MUST filter datacenters against a known-valid set for network volumes, skipping datacenters that will fail with "not found or does not support network volumes".
- **FR-007**: The setup helper MUST default to COMMUNITY cloud type for lower cost and higher availability.
- **FR-008**: The setup helper MUST exclude datacenter GPU types (A100, H100, H200, B200, L4, L40, A40, Tesla, RTX PRO, AMD) by default.
- **FR-009**: The setup helper MUST support `--no-network-volume` for Pod-local persistent storage as a fallback.
- **FR-010**: The setup helper MUST support `--no-auto-transfer` to skip `runpodctl send/receive` bootstrap and rely on manual `scp`/`rsync`.
- **FR-011**: The setup helper MUST support `--no-auto-start-training` to stop after verify, letting the operator start training manually.
- **FR-012**: The setup helper MUST print `scp` and `ssh` commands with the Pod's real IP and port after creation.
- **FR-013**: The bootstrap script MUST use `set -euo pipefail` and log to `/workspace/bootstrap.log`.
- **FR-014**: The bootstrap script MUST handle `runpodctl receive` failure gracefully by printing manual `scp` instructions instead of exiting silently.
- **FR-015**: All errors from RunPod API calls MUST be classified as retryable (500-level, "not found", "no instances") or fatal (400/401/403/404).
- **FR-016**: The setup helper MUST clean up auto-created network volumes when a Pod creation attempt fails with a retryable error.
- **FR-017**: The integration MUST document SSH key setup via the RunPod web console (my.runpod.io → Settings → SSH Keys) as the primary SSH configuration path.
- **FR-018**: The integration MUST document `runpodctl` v2.x positional-arg syntax (`runpodctl ssh info <pod-id>`, `runpodctl pod list`, `runpodctl pod delete <id>`).
- **FR-019**: The integration MUST provide a Serverless migration guide describing how to wrap the stable training command in a handler that reads from `/runpod-volume/`.

### Key Entities

- **PackageResult**: Name, bundle directory, archive path.
- **RunPodApiError**: Method, path, HTTP status, detail text. Classifies as retryable or fatal.
- **AvailabilityCandidate**: (GPU type, datacenter ID) pair filtered by VRAM, cloud type, and valid-network-volume datacenter set.
- **SetupManifest**: JSON file recording package name, Pod payload, Pod info, network volume info, transfer code, availability attempts, and API key absence.

### Success Criteria

- **SC-001**: A new project can bootstrap RunPod training by copying the packager template, adjusting 4 paths, and running 2 commands.
- **SC-002**: The packager never includes source/client/raw data in the bundle.
- **SC-003**: The setup helper creates a Pod via REST API OR prints manual web UI instructions as fallback.
- **SC-004**: Data transfer succeeds via `runpodctl send/receive` OR `scp` OR `rsync`.
- **SC-005**: Training starts automatically on the Pod after archive unpack, OR the operator can start it manually with one command.

## Assumptions

- `runpodctl` v2.x is installed locally (download `runpodctl-windows-amd64.exe` into the project `.venv/Scripts/` or PATH).
- `RUNPOD_API_KEY` is set in the environment and has Pod/volume creation permissions.
- The RunPod REST API base is `https://rest.runpod.io/v1`.
- Network volumes require a concrete datacenter ID (never `"auto"`).
- The RunPod web UI (my.runpod.io) is the most reliable Pod creation path and serves as the fallback when the REST API fails.
- Pricing is NOT available through `runpodctl` or the REST API; approximate hardcoded prices are used for sorting only.

## RunPod Integration Checklist (Copy Per Project)

1. **Install runpodctl**: Download `runpodctl-windows-amd64.exe` into `.venv/Scripts/runpodctl.exe`. Verify with `uv run runpodctl version`.
2. **Configure API key**: `uv run runpodctl config --apiKey $env:RUNPOD_API_KEY`.
3. **Set up SSH keys**: Generate `ssh-keygen -t ed25519`, add `.pub` to my.runpod.io → Settings → SSH Keys.
4. **Package derived data**: Run packager with `--archive-format tar`. Verify `manifest.json` shows `contains_game_client_files: false`.
5. **Create Pod**: Run setup helper with `--no-auto-transfer --no-auto-start-training`. If REST API fails, create Pod from web UI manually.
6. **Get SSH info**: `uv run runpodctl ssh info <pod-id>`. Wait for public IP.
7. **Transfer archive**: `scp -P <port> <archive.tar> root@<ip>:/workspace/`.
8. **SSH in and run setup**: `ssh -p <port> root@<ip>`, then `tar -xf <archive>`, `cd <bundle>`, `bash runpod/install_deps.sh`, `bash runpod/verify_bundle.sh`, `bash runpod/smoke.sh`, `bash runpod/train.sh`.
9. **Monitor training**: SSH in periodically, check `*_metrics.json`, `*_preview.png`, and `*_validation_previews/`.
10. **Download results**: `scp -P <port> root@<ip>:/workspace/<bundle>/models/<run>/*_{metrics.json,best.pt,latest.pt,preview.png} ./`.
11. **Clean up**: `uv run runpodctl pod stop <pod-id>` or `uv run runpodctl pod delete <pod-id>`. Delete orphaned network volumes.

## Serverless Migration Path

Once the training command is stable on a Pod:

1. Keep the bundle on a network volume attached to `/runpod-volume/`.
2. Write a Python handler that reads the bundle from `/runpod-volume/<bundle-dir>/` and runs the same training script.
3. Create a Serverless endpoint with the handler image.
4. Attach the network volume to the endpoint.
5. Trigger the endpoint to run training/inference jobs.
6. Flash (`@Endpoint`) can also attach network volumes for local-style function calls on Serverless workers.

## Relationship to Other Specs

- **Informs**: Spec 077 (Minimap Deconstruction Engine) — the source of all lessons learned here.
- **Informs**: Any future project that needs cloud GPU training.
- **References**: RunPod docs at https://docs.runpod.io, runpodctl at https://github.com/runpod/runpodctl.
