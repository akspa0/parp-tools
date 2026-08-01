# Data Paths

Where wow-viewer finds and writes data, and how to point it at a different working tree.

This document is the authoritative reference for any path that appears in wow-viewer code, scripts, tests, or documentation. If you find a path that is not described here, treat it as a bug and file an issue.

## Quick Reference

| What | Default location | Override |
|------|------------------|----------|
| Workspace root | `<repo>` (wherever you cloned `parp-tools`) | `WOWVIEWER_WORKSPACE` |
| Configured game-client library | runtime CLI/config; `H:\CLIENTS` is approved on this machine | explicit CLI argument |
| Optional staged game clients | `output/tmp/wowarchive-clients/` (under the workspace) | `WOWVIEWER_STAGED_CLIENTS` |
| WoWArchive mount | `G:\WoW\WoWArchive-0.X-3.X\Mount` (Windows default) | `WOWARCHIVE_MOUNT` |
| Build/test data | `test_data/` (under the workspace) | `WOWVIEWER_TEST_DATA` |
| Caches (listfiles, PM4 overlays) | `wow-viewer/output/cache/` | `WOWVIEWER_CACHE` |
| Datasets (Zarr stores) | `wow-viewer/output/datasets/` | `WOWVIEWER_DATASETS` |
| Temp / scratch | `output/tmp/` | `WOWVIEWER_TMP` |

All default paths are relative to the workspace root unless stated otherwise. The `wow-viewer/` prefix on a default indicates the path is under the `wow-viewer` subdirectory of the monorepo.

All paths are normalized to forward slashes in code, manifests, and reports. Windows backslashes appear only in this document when showing a literal example.

## Game Client Access

### Configured client roots

Pass the client-library root at runtime; do not bake a machine-local path into source or portable
configuration. `H:\CLIENTS` is the current user-curated, known-good fast SSD library and is approved
for validation, extraction, inspection, and harvesting. Trust in the library does not replace
per-build fingerprinting: every report must still identify the exact configured root, build, and
fingerprint used.

The optional project-local staging root is:

```
<workspace>/output/tmp/wowarchive-clients/
```

Each client build lives in its own subdirectory, named by build identifier. Examples:

```
output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft
output/tmp/wowarchive-clients/0_5_3_3368/World of Warcraft
output/tmp/wowarchive-clients/3_0_1_8303/World of Warcraft
```

When reporting validation, always name the staged client root you used.

### How to stage a client when the build is not in the approved library

1. If you have a `WoWArchive` mount available, locate the build under the mount.
2. Copy the required build folder into `output/tmp/wowarchive-clients/`.
3. Process the staged copy. Do not stream from the mount for repeated or wide reads.
4. Delete the staged copy when the task is done so the temp area does not grow without bound.

A local staged copy is roughly five times faster than reading through the archive mount, even before factoring SSD vs. network-attached storage. Stage first, work second.

### `H:\CLIENTS` policy

`H:\CLIENTS` is explicitly approved by the user as the current known-good client library. It is a
runtime input, never a source-code default. The older "forbidden/untrusted" wording is retired;
project-local staging remains optional for builds absent from the approved library or for bounded
scratch copies.

### WoWArchive (source-of-truth bundle)

The deduplicated WoWArchive bundle contains the canonical 0.x through 3.x client data. On the default Windows setup it lives at:

```
G:\WoW\WoWArchive-0.X-3.X\
```

with the mount entrypoint at `MountAll.bat` and the readme at `Readme.txt`. Running `MountAll.bat` mounts the bundle read-only (via `rman-mount` / WinFsp) into `G:\WoW\WoWArchive-0.X-3.X\Mount`.

The mount is intended for discovery and one-off copies, not as a working root. Stage what you need, then work from the staged copy.

If your WoWArchive lives elsewhere, set `WOWARCHIVE_MOUNT` to its mount point. On non-Windows hosts, `WOWARCHIVE_MOUNT` is the path where you have bound the deduplicated bundle (for example, via a loop mount or a network share).

## Test Data

Under the workspace, `test_data/` ships with the repository and provides bounded reference assets for unit tests, integration tests, and reproducible proofs.

```
<workspace>/test_data/development/World/Maps/development
```

This directory contains the development map corpus: split Cataclysm ADTs (root plus `_obj0` and `_tex0` variants) and 616 PM4 pathfinding files. It is the primary test fixture for `WowViewer.Core.Tests`.

```
<workspace>/test_data/0.5.3/
```

Alpha 0.5.3 reference assets (MDX, BLP, DBC) for era-aware reader tests.

```
<workspace>/test_data/minimaps/development
```

Development map minimap PNGs. Used by training-curation and validation-capture tests.

To relocate the entire test-data tree (for example, onto a fast scratch disk), set `WOWVIEWER_TEST_DATA` to the new root.

## Output Roots

| Purpose | Default | Override |
|---------|---------|----------|
| Caches (listfiles, PM4 overlays) | `wow-viewer/output/cache/` | `WOWVIEWER_CACHE` |
| Datasets (Zarr stores) | `wow-viewer/output/datasets/` | `WOWVIEWER_DATASETS` |
| Smoke / scratch reports | `wow-viewer/output/tmp/` | `WOWVIEWER_TMP` |
| Optional staged clients | `output/tmp/wowarchive-clients/` | `WOWVIEWER_STAGED_CLIENTS` |

Output directories under `wow-viewer/output/` are gitignored. Datasets and caches are large; they live under `wow-viewer/output/` rather than the repo-root `output/` so they stay inside the active development target.

For throwaway smoke runs, use `wow-viewer/output/tmp/` rather than the repo-root `output/tmp/` unless you have a specific reason to share scratch space with the legacy tool.

## How Overrides Resolve

When a wow-viewer tool or library needs a path, it resolves in this order:

1. Explicit CLI argument (if the tool accepts one).
2. Environment variable (the names listed in the tables above).
3. Convention-based default relative to the workspace root.

The workspace root is the directory containing the `wow-viewer/` subdirectory. The default is the current working directory if a `wow-viewer/` directory is present there; otherwise the parent of the `wow-viewer` binary's location. Set `WOWVIEWER_WORKSPACE` to override.

There is no automatic fallback to a hardcoded absolute path. If an override is unset and the default is missing, the operation fails with a clear "path not found" error rather than silently substituting a stale root.

## Path Conventions

- **Forward slashes everywhere in code.** `Path.Combine` is fine for filesystem I/O; manifest and report output use forward slashes so cross-platform consumers do not have to translate.
- **Lowercase paths in manifests and reports.** `WowViewer.Core.Anim.PathNormalizer.NormalizeForOutput` enforces this for anim-farm manifests; the same convention applies to PM4 overlay reports, dataset manifests, and any other machine-readable output.
- **Build identifiers in directory names.** Use the canonical underscore form (`3_3_5_12340`, `0_5_3_3368`, `3_0_1_8303`) so build-aware asset keys are stable across machines.
- **Workspace-relative paths in code and tests.** Absolute paths in code are a smell. The only acceptable absolute path is one resolved at runtime from an env var or a CLI arg.
- **Staging directories are per-task.** When you stage a client for a one-off run, name the directory by build identifier and delete it when the run is done.

## See Also

- `wow-viewer/README.md` — workspace overview and quick start
- `wow-viewer/AGENTS.md` — repository guardrails, including the staged-client-only rule
- `wow-viewer/memory-bank/coding_standards.md` — how to write code that respects these paths
- `wow-viewer/docs/CLI-TOOLS.md` — how each tool resolves its inputs and outputs
- `wow-viewer/docs/architecture/` — per-format and per-feature architecture notes
