# wow-viewer Constitution

## Core Principles

### I. Repo Independence

`wow-viewer` must be extractable as its own standalone repository. No source file may reference a path outside `wow-viewer/` (except game client paths on disk). No project file may reference a `.csproj` outside `wow-viewer/`. All shared code lives inside `wow-viewer/src/core/` or `wow-viewer/data-harvester/src/`.

### II. Library-First

Every capability starts as a shared library in `WowViewer.Core`, `WowViewer.Core.IO`, `WowViewer.Core.PM4`, or `WowViewer.Core.Runtime`. CLI tools are thin wrappers. Format readers/writers are never duplicated across tools. One canonical owner per format surface.

### III. Real-Data Validation

Every format, converter, and dataset claim must be validated against real staged game client data from `output/tmp/wowarchive-clients/`. Mock assets are not sufficient for signoff. Validation evidence (commands, outputs, hashes) must be reproducible.

### IV. Residual Model Chain

Every V14+ terrain model predicts ONE residual signal. Models chain together — each model's output becomes input to downstream models. No monolithic models. No multi-task training. No shared weights between models. Each model trains independently with its own checkpoint.

### V. Streaming-First Dataset Pipeline

Data flows from the C# harvester through a length-prefixed binary protocol over stdout directly into the Python Zarr writer. No intermediate NPZ files on disk. The Zarr store is the only on-disk artifact. NPZ shard format is the contract between C# and Python; both sides must agree on array names, shapes, and dtypes.

### VI. No Game Client Path Assumptions

Client data locations are **configuration, never assumptions baked into code**. No source file, script, or doc may hardcode a client root. Validation and harvesting read from a configured clients folder; the current default is `I:\parp\parp-tools\output\tmp\wowarchive-clients\`.

**Storage layout (as of 2026-07-15):** the authoritative corpus is **WoWArchive** (~150 GB, cold HDD storage). `H:\CLIENTS` is a **curated temporary SSD staging area** the user copies individual builds into for working sessions. Both are legitimate sources.

**Amended 2026-07-15 (v1.1.0).** This principle previously read *"Never use `H:\CLIENTS` for anything. Those paths are untrusted."*

- **Rationale for the amendment**: the original prohibition was written against a specific hazard — broken clients of unknown origin that the user did not trust. The user has since cleaned that folder out; the hazard no longer exists, and the folder is now curated for this project's needs. The rule outlived its reason and had started producing false conflicts (e.g. the 1.0.0 Ghidra evidence underpinning spec 105 is derived from a binary imported from that path, which the old wording nominally forbade while every prior session relied on it).
- **Approved by**: the user, 2026-07-15, in session.
- **Migration**: one enforcement point exists and now **contradicts** this principle — `WowViewer.Core.Anim/PathNormalizer.cs` (`StaleClientsRoot`) **throws `InvalidOperationException`** on any path containing `H:\CLIENTS`, with `PathNormalizerTests` pinning that behaviour. Under the amended principle the pose-farm library would refuse a legitimate staging path. **This is a tracked follow-up, deliberately not bundled into the amendment commit** (it is a code change to spec 053's library, outside the scope of the session that raised it). Until it is removed or retargeted, `Core.Anim` consumers must continue to pass staged-client paths. Documentation and memory-bank text asserting a blanket prohibition is superseded by this principle. Static RE evidence derived from a staged binary is explicitly permitted and should cite the build it came from.

**What has NOT changed**: hardcoding *any* client path is still forbidden, `output/tmp/wowarchive-clients/` remains the default for automated validation, and the Data Policy below still governs distribution.

## Safety Constraints

### Read-Only Reference Codebase

`gillijimproject_refactor` is READ-ONLY. No new code, no new features, no refactoring, no rewrites. It exists as a reference implementation and compatibility validation host only. The only valid writes are bounded hotfixes explicitly requested by the user.

### Format Reader/Writer Ownership

The tooling for reading game client files is COMPLETE and lives in `wow-viewer/src/core/WowViewer.Core.IO/`. Do not rewrite parsers that already exist and work. If a new format needs a reader, check if one already exists before writing a new one.

### Terrain Alpha Risk Area

Any change to MCAL decode, edge-fix behavior, `_tex0.adt` texture sourcing, alpha packing, or shader blending must be checked against both Alpha-era terrain and LK 3.3.5 terrain. The pre-regression baseline is commit `343dadfa27df08d384614737b6c5921efe6409c8`.

### AlphaWdtWriter is Frozen

`AlphaWdtWriter.cs` is COMPLETE for current project needs. Do not touch it unless: (1) user explicitly reopens alphaWDT writer work, (2) a focused regression fix is required, or (3) a bounded compatibility change with round-trip proof.

## Development Workflow

### One Phase at a Time

Cannot work on Phase N+1 until Phase N is done. Done means validated, not coded. Every phase ends with validation against ground truth (MdxViewer renders, raw game file data). If validation fails, the phase is not done. Do not move on.

### Spec Docs Are Source of Truth

Every model, dataset, training run, pipeline, and public interface has a spec doc in `wow-viewer/docs/architecture/`. When code changes that a spec describes, update the spec in the same commit. If no spec exists, create one before implementing.

### Training Script Changes Require Validation

Every change to a training script must have a documented reason and a validation path. Do not change input channel counts without updating the model spec. Do not add new loss terms without documenting their weight and purpose.

### Memory Bank Discipline

At the end of every non-trivial session, update `activeContext.md` and `progress.md`. Compress aggressively — prefer a 20-line accurate summary over a 200-line log. The memory bank does NOT auto-update when code edits are made; this is a manual discipline.

### Bite-Sized Plans

Every plan must be decomposed into steps small enough for ANY LLM model to implement in a single focused pass. One concern per step, independently validatable, max 10 steps per phase.

## Technology Stack

- **C# / .NET 10**: Core libraries, tools, converters, inspectors
- **Python 3.11+ / uv**: Dataset building, training, inference, validation
- **Zarr v3**: Dataset storage (one store per client build)
- **Parquet**: Index and metadata files
- **Silk.NET.OpenGL**: Current rendering backend (Vulkan/WebGL are future)
- **PyTorch**: Model training and inference
- **Blosc (lz4/zstd)**: Array compression in Zarr stores

## Data Policy

Bring Your Own Data. Do not distribute proprietary client data, harvested corpora, or derived outputs from copyrighted game assets. All dataset work assumes lawful access to game client files.

## Governance

This constitution supersedes all other development practices when conflicts arise. Amendments require: (1) documented rationale, (2) user approval, (3) migration plan for affected code. The workspace `AGENTS.md` at repo root is the authoritative policy source for scope, safety, and repo boundaries.

**Version**: 1.1.0 | **Ratified**: 2026-05-18 | **Last Amended**: 2026-07-15

### Amendment log

- **1.1.0** (2026-07-15) — Principle VI rewritten. The blanket `H:\CLIENTS` prohibition was retired: it was written against untrusted broken clients that the user has since removed, and the folder is now a curated SSD staging area fed from WoWArchive (~150 GB, cold HDD). The principle's durable intent — never hardcode a client root — is preserved and strengthened. Rationale, approval, and migration recorded inline. Requested and approved by the user in session.
