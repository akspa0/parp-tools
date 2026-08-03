# wow-viewer Constitution

## Core Principles

### I. Repo Independence

`wow-viewer` must be extractable as its own standalone repository. No source file may reference a path outside `wow-viewer/` (except game client paths on disk). No project file may reference a `.csproj` outside `wow-viewer/`. All shared code lives inside `wow-viewer/src/core/` or `wow-viewer/data-harvester/src/`.

### II. Library-First

Every capability starts as a shared library in `WowViewer.Core`, `WowViewer.Core.IO`, `WowViewer.Core.PM4`, or `WowViewer.Core.Runtime`. CLI tools are thin wrappers. Format readers/writers are never duplicated across tools. One canonical owner per format surface.

### III. Real-Data Validation

Every format, converter, and dataset claim must be validated against real data from an approved,
configured game-client library. `H:\CLIENTS` is the current preferred fast SSD library of known-good
builds; `output/tmp/wowarchive-clients/` remains optional staging. Mock assets are not sufficient for
signoff. Validation evidence must record commands, configured root, build identity, and hashes.

### IV. Model Architecture Is an Engineering Choice, Evidence Is Not

Model topology — single-output, multi-head, shared trunk, monolithic, chained — is chosen per spec
on technical merit. **Multi-task training and shared weights are explicitly permitted.** A spec must
state which topology it uses and why; it does not need to justify itself against a prohibition.

What is NOT negotiable is that a model cannot hide a dead signal inside an aggregate win:

- A model producing N signals MUST report per-signal validation metrics, each against that signal's
  own trivial baseline. An aggregate loss that improves while one output stays at baseline is a
  **partial failure and must be reported as one**.
- Every output must be independently ablatable — removing or freezing one head must be possible
  without retraining the others from scratch, so a signal that turns out not to be recoverable can
  be dropped rather than silently carried.
- Whether a signal is recoverable AT ALL is an empirical question answered per signal, never assumed
  from a shared trunk's overall performance. The target must be visible in the input.

**Amended 2026-08-02 (v2.0.0).** This principle previously read *"Every V14+ terrain model predicts
ONE residual signal... No monolithic models. No multi-task training. No shared weights between
models."*

- **Rationale for the amendment**: the V14 residual-chain architecture was tried extensively and did
  not produce the wins it promised — the tile-mean baseline remains unbeaten across much of that
  lane. The prohibition had stopped describing a working method and had become a blocker: Spec 125
  established that we now hold a complete, known forward model for minimap generation, which makes
  "one minimap tile in, every signal out" a reasonable architecture to attempt. The old rule forbade
  attempting it. A constitution should encode what works, not what was hoped for in May 2026.
- **Approved by**: the user, 2026-08-02, in session, explicitly and after the conflict was raised.
- **Migration**: no code change is required — every existing model remains valid under the amended
  principle, since single-output specialists are still an allowed topology. Prior specs (117, 118,
  123, 125) and archived specs (119) cite this principle by its old wording, often as a "PASS"
  compliance gate; **those are historical records of decisions made under the old rule and are left
  as written**. New specs must not cite the old prohibition. Conflicting policy text in the root
  `AGENTS.md` (RULE 7) and `wow-viewer/memory-bank/coding_standards.md` was updated in this same
  pass. Model docstrings referencing "constitution IV" for single-output justification are stale but
  harmless; correct them opportunistically, not in a sweep.

**What has NOT changed**: models are still validated against real data (Principle III), training
script changes still require a documented reason and validation path, and a claimed win still needs
a baseline it actually beats.

### V. Streaming-First Dataset Pipeline

Data flows from the C# harvester through a length-prefixed binary protocol over stdout directly into the Python Zarr writer. No intermediate NPZ files on disk. The Zarr store is the only on-disk artifact. NPZ shard format is the contract between C# and Python; both sides must agree on array names, shapes, and dtypes.

### VI. No Game Client Path Assumptions

Client data locations are **configuration, never assumptions baked into code**. No source file or
portable config may hardcode a client root. Validation and harvesting read from a configured clients
folder; the current operator-approved preferred root is `H:\CLIENTS`.

**Storage layout (as of 2026-07-15):** the authoritative cold corpus is **WoWArchive** (~150 GB,
cold HDD storage). `H:\CLIENTS` is a **user-curated, known-good SSD client library** with broader,
faster build coverage. Both are legitimate sources; v50 work should prefer the SSD library.

**Amended 2026-07-15 (v1.1.0).** This principle previously read *"Never use `H:\CLIENTS` for anything. Those paths are untrusted."*

- **Rationale for the amendment**: the original prohibition was written against a specific hazard — broken clients of unknown origin that the user did not trust. The user has since cleaned that folder out; the hazard no longer exists, and the folder is now curated for this project's needs. The rule outlived its reason and had started producing false conflicts (e.g. the 1.0.0 Ghidra evidence underpinning spec 105 is derived from a binary imported from that path, which the old wording nominally forbade while every prior session relied on it).
- **Approved by**: the user, 2026-07-15, in session.
- **Migration**: one enforcement point exists and now **contradicts** this principle — `WowViewer.Core.Anim/PathNormalizer.cs` (`StaleClientsRoot`) **throws `InvalidOperationException`** on any path containing `H:\CLIENTS`, with `PathNormalizerTests` pinning that behaviour. Under the amended principle the pose-farm library would refuse a legitimate staging path. **This is a tracked follow-up, deliberately not bundled into the amendment commit** (it is a code change to spec 053's library, outside the scope of the session that raised it). Until it is removed or retargeted, `Core.Anim` consumers must continue to pass staged-client paths. Documentation and memory-bank text asserting a blanket prohibition is superseded by this principle. Static RE evidence derived from a staged binary is explicitly permitted and should cite the build it came from.

**What has NOT changed**: hardcoding a machine-local client path in source is still forbidden, every
build must be fingerprinted, and the Data Policy below still governs distribution.

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

**Version**: 2.0.0 | **Ratified**: 2026-05-18 | **Last Amended**: 2026-08-02

### Amendment log

- **2.0.0** (2026-08-02) — Principle IV replaced. The V14 residual-model-chain prohibition ("no
  monolithic models, no multi-task training, no shared weights") is **retired**: the architecture it
  mandated was tried at length and did not beat its baselines, while the rule itself had begun
  blocking work the project now has the forward model to attempt. Model topology is now an
  engineering choice made per spec. MAJOR rather than MINOR because active specs cite the removed
  prohibition as a compliance gate, so their "PASS" claims against it no longer carry meaning. The
  durable intent — a strong signal must never mask a dead one — is preserved and strengthened as a
  per-signal reporting requirement. Requested and approved by the user in session after the conflict
  was surfaced. Conflicting `AGENTS.md` RULE 7 and `coding_standards.md` text updated in the same
  pass.
- **1.2.0** (2026-07-15) — User confirmed `H:\CLIENTS` is the current known-good, faster SSD
  client library. It is now the preferred configured root for v50 work; project-local staging is
  optional. Per-build fingerprints and runtime configuration remain mandatory. Conflicting AGENTS
  text was updated in the same pass. The existing Core.Anim stale-root throw remains a tracked
  compatibility fix and is not needed by the v50 dataset path.
- **1.1.0** (2026-07-15) — Principle VI rewritten. The blanket `H:\CLIENTS` prohibition was retired: it was written against untrusted broken clients that the user has since removed, and the folder is now a curated SSD staging area fed from WoWArchive (~150 GB, cold HDD). The principle's durable intent — never hardcode a client root — is preserved and strengthened. Rationale, approval, and migration recorded inline. Requested and approved by the user in session.
