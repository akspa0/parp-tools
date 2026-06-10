# Feature Specification: M2/MDX Animation Pose Farm

**Feature Branch**: `053-m2-animation-pose-farm`
**Created**: 2026-06-09
**Status**: Draft
**Input**: User description: "build a tool to farm all the animations from MDX/M2 models, for use as an animation pose library, perhaps focusing on a modern pose format for use in mocap or image gan controlnet use"

## Context

The wow-viewer toolchain already reads M2/MDX (and their external `.anim` companions), resolves sequence names, and can evaluate per-bone TRS at any timestamp. The pieces needed to "farm animations" are essentially:

- `M2ModelReaderDispatcher` — open M2/MDX/classic/1121-era models uniformly
- `M2ExternalAnimationRuntime.Choose` / `Load` — resolve which `.anim` file backs a sequence
- `M2TrackSampler` — read keyframes with the correct interpolation mode
- `M2BonePoseEvaluator` — compute world transforms per bone
- `M2AnimationNameResolver` — map `(animationId, variationIndex)` to a human name
- `MdlWriter` — canonical M2 description writer (used as round-trip / debug path)
- **`WoWViewer` (the viewer app under `src/viewer/`)** — a real, working OpenGL renderer for the same M2/MDX data, with a headless capture surface (`ViewerApp_CaptureAutomation`, `ScreenshotRenderer`, `AssetExporter`). This is the canonical rasterizer in this repo and is what any future per-frame image pipeline (depth, normal, silhouette, OpenPose) must route through. **Do not build a separate rasterizer for the pose farm.**

What is missing is the glue that walks a model, enumerates its sequences, walks each sequence's keyframes, and exports the result in a pose-library friendly format. This spec defines that glue, plus a small BVH writer, an FBX fallback, a normalized pose-clip sidecar, and a tagged library index.

The ChatGPT-suggested pipeline (BVH/FBX extraction → skeleton normalization → per-frame depth/normal/silhouette/OpenPose rasterization → indexed pose library) maps cleanly onto this codebase:

- **v1 (this spec)**: BVH + FBX + pose clip + library index. No rasterization. Pure data, ML-ready keyframes.
- **v2 (future, separate spec)**: Per-frame conditioning. Routes through the existing `WoWViewer` capture surface, not a new renderer. Adds depth/normal/silhouette rasterization for ControlNet and OpenPose-style 2D keypoint extraction (via existing 3D-pose-projection or a small Python post-process).

## User Scenarios & Testing

### User Story 1 - Single model BVH + JSON dump (Priority: P1)

As a tool user, I can run `WowViewer.Tool.AnimFarm dump --input <path/to/model.m2>` and get a directory of BVH motion files (one per non-alias sequence) plus a JSON sidecar that records bone hierarchy, sequence metadata, and source identity.

**Why this priority**: The BVH + JSON pair is the smallest end-to-end proof that we can walk an M2, evaluate all sequences, and emit a pose-library artifact. Without it nothing else matters.

**Independent Test**: Run `dump` against one representative model from each era (classic, TBC, WotLK, Cata/MoP via chunked M2). Confirm BVH files are valid (parse with a known BVH reader) and the JSON sidecar contains the expected sequence list with correct durations.

**Acceptance Scenarios**:

1. **Given** a chunked M2 with 50 sequences, **When** the user runs `dump --input x.m2`, **Then** the tool writes one `.bvh` per resolved (non-alias) sequence, one `manifest.json`, and a `bones.json` sidecar.
2. **Given** a sequence backed by an external `.anim` file, **When** the tool is run with the .anim on the same `MPQ` search path, **Then** the BVH contains the keyframes from the external file (not just the inline fallback).
3. **Given** an alias sequence, **When** the tool resolves it, **Then** the alias is followed and the resolved sequence is exported; the alias is recorded in the manifest but does not produce a duplicate BVH.

---

### User Story 2 - Listfile-driven batch run (Priority: P1)

As a tool user, I can run `WowViewer.Tool.AnimFarm batch --client-root <staged-client>` and have the tool enumerate every M2/MDX in the client via the existing listfile cache, farm animations for each, and write results to a per-model subdirectory under `--output`.

**Why this priority**: The whole point of the feature is to mine a corpus. Single-file `dump` is a stepping stone; the batch mode is the deliverable.

**Independent Test**: Point at a staged 3.3.5 client, run `batch`, confirm the tool processes at least the creature and character subfolders and emits one BVH per (model, sequence) pair. Confirm a non-zero fraction of models fail predictably (corrupt MPQ entries, missing .anim, etc.) with a per-model error log rather than crashing the run.

**Acceptance Scenarios**:

1. **Given** a staged client root with a built listfile cache, **When** the user runs `batch --client-root <dir> --output <dir>`, **Then** the tool walks the listfile, filters to `*.m2` and `*.mdx`, and processes each model.
2. **Given** a model that fails to read, **When** the batch encounters it, **Then** the error is recorded in `errors.jsonl` and the run continues with the next model.
3. **Given** the user passes `--include` (regex) or `--exclude` (regex), **When** the batch runs, **Then** only matching model paths are processed.

---

### User Story 3 - FBX fallback exporter (Priority: P2)

As a tool user, I can request FBX output via `dump --format fbx` or `batch --format fbx` to get industry-standard skeleton files for use in Blender/Maya/mocap tools.

**Why this priority**: BVH is sufficient for proof and most pose uses, but FBX is the de-facto interchange format for mocap retargeting pipelines. Worth having but not blocking.

**Independent Test**: Convert one known-good BVH run to FBX and confirm the resulting file opens in a standard FBX reader and contains the expected bone count and frame count.

**Acceptance Scenarios**:

1. **Given** a successfully dumped model, **When** the user runs `dump --input x.m2 --format fbx`, **Then** the tool writes `.fbx` files instead of `.bvh` with the same per-sequence content.
2. **Given** a batch run with `--format fbx`, **When** the tool processes N models, **Then** it emits one `.fbx` per sequence per model.

---

### User Story 4 - Stable pose library indexing (Priority: P2)

As a downstream consumer (PyTorch loader, web viewer, dataset inspector), I can read a top-level `index.json` written by the batch run that lists every (model_path, sequence_index, bvh_file, duration_ms, frame_count, bone_count) tuple for fast lookup.

**Why this priority**: A flat `index.json` is what turns "a folder of files" into "a pose library". Important for ML/ControlNet consumption but the per-file outputs are still usable without it.

**Independent Test**: After a batch run, parse `index.json` and confirm one entry per (model, sequence) with all fields populated and consistent with the per-file BVH header.

**Acceptance Scenarios**:

1. **Given** a completed batch run, **When** the user opens `index.json`, **Then** it contains a stable, sorted list of every exported sequence with model identity, sequence index, duration, frame count, bone count, and BVH/FBX filename.
2. **Given** two batch runs over the same client with the same config, **When** the user diffs the resulting `index.json` files, **Then** they are byte-identical (deterministic ordering and content).

---

### User Story 5 - Pose clip sidecar with normalized skeleton (Priority: P1)

As a downstream consumer (UniRig retargeting, ControlNet conditioning pipeline, ML dataset loader), I get a `.poseclip.json` per exported sequence containing the keyframes in a normalized, retargetable form — bones mapped to the Mixamo humanoid skeleton (the de-facto standard that Unreal Engine, Blender Rigify, and most retargeting tools understand), with the original WoW bone names preserved alongside in an `extras` map for any unmatched bone.

**Why this priority**: The BVH/FBX outputs are skeleton-bound. For retargeting, ControlNet conditioning, and any "pose library" that mixes models, you need a normalized representation. The pose clip is what makes the tool useful as a dataset, not just a dump.

**Independent Test**: For a known humanoid model (e.g. `Creature/Orc/OrcMale.m2`), `dump` produces a `.poseclip.json` that:
1. Contains keyframes for the standard Mixamo slots (Hips, Spine, Spine1, Spine2, Neck, Head, LeftShoulder, LeftArm, LeftForeArm, LeftHand, RightShoulder, RightArm, RightForeArm, RightHand, LeftUpLeg, LeftLeg, LeftFoot, LeftToeBase, RightUpLeg, RightLeg, RightFoot, RightToeBase) when the source model has corresponding bones.
2. Original WoW bone indices and pivots are preserved in `extras` for traceability.
3. Each keyframe carries `(timestampMs, translation[3], rotation[4] as xyzw quaternion, scale[3])` in a flat array for ML-friendliness.
4. A `summary` block exposes `tags` (derived from `M2AnimationNameResolver` + the model identity's creature class), `frameCount`, `durationMs`, `rootMotionDelta` (XZ displacement across the clip), and `bounds` (XYZ AABB of all keyframe bone positions).

**Acceptance Scenarios**:

1. **Given** a humanoid M2 with a Spine bone, **When** the tool emits the pose clip, **Then** the `Spine` slot is populated and the source bone index is recorded in `extras.spine_src_bone_index`.
2. **Given** a creature with no humanoid bones (e.g. a fish), **When** the tool emits the pose clip, **Then** standard Mixamo slots are present but mostly empty (identity TRS), and the actual bones appear in `extras` only.
3. **Given** the same animation exported twice (e.g. once as BVH, once as the sidecar), **When** the user diffs timestamps and bone sets, **Then** they match exactly.
4. **Given** a pose clip, **When** a downstream Python script reads it via `json.load`, **Then** all numeric arrays parse as flat `float` lists with no nested objects, ready for `np.array` conversion.

---

### User Story 6 - Pose Library Index with tags and search (Priority: P2)

As a consumer building a "teleplay engine" or pose library browser, I get a single `library.index.json` at the batch output root that contains one row per (model, sequence) with derived tags, summary stats, and paths to the per-clip BVH/FBX/poseclip artifacts. This is the lookup table that turns the dump into a searchable pose library.

**Why this priority**: Without an index, the consumer has to walk the directory tree and parse every manifest. With it, they can `jq '.clips[] | select(.tags contains ["attack"])'` and get results.

**Independent Test**: After a batch run, `library.index.json` contains one entry per exported (model, sequence) with:
- `modelPath` (relative, forward-slash, lowercase)
- `sequenceName` (resolved display name)
- `animationId`, `variationIndex`, `durationMs`, `frameCount`
- `tags` array (e.g. `["humanoid", "attack", "1h"]` — derived from model class + `M2AnimationNameResolver` category)
- `bvhPath` and `poseClipPath` (relative to the library root)
- `rootMotionDelta`, `bounds` (from the pose clip summary)

**Acceptance Scenarios**:

1. **Given** a batch run that emits 1000 clips, **When** the user opens `library.index.json`, **Then** it is sorted by `(modelPath, sequenceIndex)` and is byte-deterministic across runs.
2. **Given** the `tags` array, **When** the user filters by `tags contains ["walk"]` and `tags contains ["humanoid"]`, **Then** only walking animations for humanoid models match (logical AND across the tag set).
3. **Given** two runs over the same client, **When** the user diffs the resulting `library.index.json` files, **Then** they are byte-identical (sort, formatting, float precision all fixed).

---

### User Story 7 - Skeleton introspection subcommand (Priority: P3)

As a tool user, I can run `WowViewer.Tool.AnimFarm skeleton --input <model>` to dump the bone hierarchy, parent indices, pivot points, and sequence list as JSON without evaluating any animation. Useful for building skeleton retarget maps upstream.

**Why this priority**: Helpful for skeleton retargeting and debugging, but not needed for the core pose-farming use case.

**Independent Test**: Run `skeleton` on a humanoid and confirm the output JSON contains bones in parent-then-child order with parent indices, pivots, and the full sequence list (with alias resolution and external-file flags).

**Acceptance Scenarios**:

1. **Given** any valid M2/MDX, **When** the user runs `skeleton --input x.m2`, **Then** a JSON document is written describing the bone hierarchy, pivot points, parent indices, and sequence metadata.

---

### Edge Cases

- **Alias loop**: a model whose alias chain loops. Tool must detect and skip the loop with a recorded error.
- **External .anim missing**: sequence claims external animation but the `.anim` file is not on the search path. Tool must record the sequence as "external-unresolved" in the manifest and continue (do not silently export empty keyframes).
- **Zero-length sequence**: a sequence with `Duration == 0` (idle/placeholder). Tool must skip it and not emit a degenerate BVH.
- **Compressed quaternions vs full quaternions**: tool must support both (`M2TrackSampler.SampleQuaternion` and `SampleCompressedQuaternion`) and record which was used per bone in the manifest.
- **Global sequence timing**: a bone track that references a global sequence (not the per-sequence duration). Tool must use the global loop duration when emitting timestamps.
- **TBC/WotLK chunked M2 vs classic M2**: both formats must be transparently supported; tool must not require the user to pick a reader.
- **Models with no bones** (pure particle/effect M2): must skip gracefully with a "no-bones" note in the manifest.
- **BVH channel order**: must emit in the standard Biovision order (position then rotation per joint). This is non-negotiable for BVH consumer compatibility.

## Requirements

### Functional Requirements

- **FR-001**: The system MUST provide a CLI tool `WowViewer.Tool.AnimFarm` with subcommands `dump`, `batch`, and `skeleton`.
- **FR-002**: The system MUST read M2 (classic and chunked), MDX, and era-1121 M2 via the existing `M2ModelReaderDispatcher` and MDX readers. No new format readers may be added.
- **FR-003**: The system MUST resolve external `.anim` companion files using the same search path rules already used by `M2ExternalAnimationRuntime` (model directory, MPQ listfile, etc.).
- **FR-004**: The system MUST follow alias chains to a terminal sequence and export only the terminal sequence. Aliases are recorded in the manifest.
- **FR-005**: The system MUST evaluate bone TRS at the native keyframe timestamps for each sequence (no resampling).
- **FR-006**: The system MUST support both compressed and full quaternion tracks transparently.
- **FR-007**: The system MUST emit BVH motion files using the Biovision channel convention (one HIERARCHY block, one MOTION block per sequence, Xposition Yposition Zposition followed by Zrotation Xrotation Yrotation per joint).
- **FR-008**: The system MUST write a `manifest.json` per model containing: model canonical path, source format (M2/MDX/chunked), bone list (index, parent, name, pivot), sequence list (index, name, animationId, variationIndex, duration, isAlias, resolvedSequenceIndex, source, frameCount), and a hash of the source file for provenance.
- **FR-009**: The system MUST support FBX export as a fallback format when `--format fbx` is passed. FBX is ASCII FBX 7.4+ (binary FBX is out of scope for v1; we ship a hand-rolled minimal ASCII writer because no FBX writer dependency is allowed in this tool's csproj).
- **FR-010**: The system MUST produce a deterministic `index.json` at the batch output root with one entry per exported (model, sequence) tuple. Sort order is `(modelPath, sequenceIndex)`.
- **FR-011**: The system MUST write an `errors.jsonl` file during batch runs capturing per-model failures (read error, alias loop, missing .anim, etc.) with stack-free error messages.
- **FR-012**: The system MUST support `--include` and `--exclude` regex filters on model path during batch runs.
- **FR-013**: The system MUST write all BVH/FBH/JSON output under a user-provided `--output` directory; never inside the client root.
- **FR-014**: The system MUST NOT use `H:\CLIENTS`. All client access goes through the staged path under `I:\parp\parp-tools\output\tmp\wowarchive-clients/`.
- **FR-015**: The system MUST live in a new library `WowViewer.Core.Anim` (under `wow-viewer/src/core/`) that exposes the pose-extraction pipeline so other tools (viewer, harvester) can consume it.
- **FR-016**: The system MUST be reachable from the new tool project `wow-viewer/tools/animfarm/WowViewer.Tool.AnimFarm/` and registered in `WowViewer.slnx` under `/tools/animfarm/`.
- **FR-017**: The system MUST emit a `clip.<sequenceName>.poseclip.json` sidecar per exported sequence containing keyframes in a normalized skeleton space.
- **FR-018**: The normalized skeleton target is the **Mixamo humanoid bone layout** (Hips, Spine, Spine1, Spine2, Neck, Head, LeftShoulder, LeftArm, LeftForeArm, LeftHand, RightShoulder, RightArm, RightForeArm, RightHand, LeftUpLeg, LeftLeg, LeftFoot, LeftToeBase, RightUpLeg, RightLeg, RightFoot, RightToeBase). This is the de-facto retargeting standard for Unreal Engine, Blender Rigify, and most consumer-grade pose/ML pipelines.
- **FR-019**: The pose clip schema MUST be:
  - top-level fields: `schemaVersion: 1`, `modelPath`, `sequenceIndex`, `sequenceName`, `animationId`, `variationIndex`, `durationMs`, `frameCount`, `skeletonTarget: "mixamo"`, `boneCount`, `sourceHash`, `tags` (array of strings), `summary` (object with `rootMotionDelta` [x,y,z], `boundsMin` [x,y,z], `boundsMax` [x,y,z]).
  - `keyframes` (array, one per native keyframe timestamp): each entry has `tMs` and a flat `bones` array of `boneCount * 10` floats in the order `[tx, ty, tz, qx, qy, qz, qw, sx, sy, sz]` per bone.
  - `boneOrder` (array of strings, length `boneCount`): the Mixamo slot name per index.
  - `extras` (object): map from Mixamo slot name → source info `{srcBoneIndex, srcBoneName, pivot: [x,y,z]}`.
- **FR-020**: The pose clip MUST be parseable as a single JSON object via `json.load` with no streaming or external references. All arrays are flat `float` lists for direct `np.array` conversion.
- **FR-021**: The system MUST emit a `library.index.json` at the batch output root. Each row contains: `modelPath`, `sequenceIndex`, `sequenceName`, `animationId`, `variationIndex`, `durationMs`, `frameCount`, `tags` (array), `bvhPath` (relative), `poseClipPath` (relative), `rootMotionDelta`, `boundsMin`, `boundsMax`.
- **FR-022**: Tags MUST be derived deterministically from `(model class, animationId)`:
  - category tag from `M2AnimationNameResolver` family (e.g. `walk`, `run`, `attack`, `spell`, `death`, `idle`, `swim`, `jump`)
  - weapon/variation tag (e.g. `1h`, `2h`, `bow`, `unarmed`) when the resolved name carries one
  - rig class tag (`humanoid` | `quadruped` | `creature` | `inanimate`) derived from the model's bone topology (a simple heuristic: humanoid if Hips + LeftUpLeg + RightUpLeg exist; quadruped if any bone suggests front/hind legs; otherwise `creature` or `inanimate`).
- **FR-023**: The `library.index.json` MUST be sorted by `(modelPath, sequenceIndex)`, formatted with `en-US` culture, and byte-identical across runs over the same inputs.
- **FR-024**: Both `dump` and `batch` MUST accept `--with-pose-clip` (default `true`) and `--with-bvh` (default `true`) so users can opt out of either output. `library.index.json` is only written by `batch` and only when at least one pose clip is exported.

### Non-Functional Requirements

- **NFR-001**: BVH writer must round-trip: a BVH written by the tool must be readable by the standard `bvh-python` parser and the in-tool `BVHReader` round-trip test in `WowViewer.Core.Anim.Tests`.
- **NFR-002**: The tool must be deterministic. Two runs over the same inputs must produce byte-identical outputs (timestamps, ordering, formatting). Use a fixed culture (`en-US`) and sort all collections.
- **NFR-003**: The tool must never throw on a single bad model during batch. Errors are recorded, the run continues.
- **NFR-004**: The library `WowViewer.Core.Anim` must have unit tests in `wow-viewer/tests/WowViewer.Core.Anim.Tests/` covering: alias resolution, BVH header generation, keyframe extraction, manifest schema stability, determinism.
- **NFR-005**: All paths in the manifest must be normalized (forward slashes, lowercased, no `H:\CLIENTS`).

### Key Entities

- **`M2AnimationPoseSource`**: identifies a model + optional external animation. Holds the loaded `M2ModelDocument` (or MDX equivalent), resolved sequence metadata, and the list of external `.anim` payloads. Built by `M2PoseSourceLoader`.
- **`M2BoneTrackStream`**: per-bone stream of `(timestampMs, translation, rotation, scale)` for a single sequence. Produced by walking the M2 track definitions via `M2TrackSampler`.
- **`BvhDocument`**: in-memory representation of a Biovision hierarchy + one or more motions. Serializer: `BvhDocumentWriter`.
- **`FbxAsciiDocument`**: minimal ASCII FBX 7.4 representation (limb-node graph + animation stack). Serializer: `FbxAsciiDocumentWriter`.
- **`PoseManifest`**: per-model manifest with bone list, sequence list, and source hash. JSON-serializable.
- **`PoseLibraryIndex`**: batch-run top-level index. One row per (model, sequence).

## Success Criteria

- **SC-001**: Given a staged 3.3.5 client root, `batch` processes at least 95% of M2/MDX entries without crashing and emits one BVH per non-alias sequence into the output tree.
- **SC-002**: At least one BVH produced by `dump --input <known-good-humanoid>` round-trips through `BVHReader` and reports the same bone count and frame count as the source M2 sequence keyframes.
- **SC-003**: For an M2 with N sequences and M external `.anim` companions, the resulting `index.json` has N rows and each row's `frameCount` matches the count of keyframes in the source (or external) track definitions.
- **SC-004**: The full unit test suite (`WowViewer.Core.Anim.Tests`) passes on Windows with .NET 10.
- **SC-005**: `WowViewer.Tool.AnimFarm --help` and `dump --help` print usage and exit 0.
- **SC-006**: A real end-to-end demo run is committed to the architecture doc (`docs/architecture/m2-anim-pose-farm-2026-06-09.md`) with a small model (e.g. `Creature/Orc/Orc.m2` from 3.3.5) showing one BVH and its keyframe counts.

## Assumptions

- The existing `M2TrackSampler`, `M2BonePoseEvaluator`, `M2ExternalAnimationRuntime`, and `MdlWriter` are correct and complete. We do not modify them; we consume them.
- The existing listfile cache (`WowViewer.Tool.Inspect archive build-listfile-cache`) is the canonical way to enumerate models in a client.
- BVH is the primary output; FBX is provided for users who specifically need it. No GLB, no USD, no Collada in v1.
- The "modern pose format for mocap / ControlNet" goal is met by BVH (mocap standard) + JSON (programmatic, ML-friendly). We do not invent a new pose format.
- Mixamo humanoid is the canonical target skeleton. It is the most widely supported retargeting target (Unreal Engine, Blender Rigify, Cascadeur, most ML pose datasets) and is the closest existing "industry standard" to a generic humanoid pose library. UniRig, if it appears, is a downstream consumer; the tool does not couple to it.
- The tool does not need to be fast. A 3.3.5 client with ~30k models can take hours. We do not parallelize within v1; the user can shard across client regions.
- All paths in the manifest are relative to the output root. The tool does not embed absolute paths from the client.
- The `WoWViewer` viewer app under `src/viewer/` is the canonical rasterizer in this repo. Any future per-frame image pipeline (v2) will route through its existing headless capture surface (`ViewerApp_CaptureAutomation`, `ScreenshotRenderer`, `AssetExporter`), not a new renderer.

## Explicitly Out of Scope (v1)

- GLB / mesh reskinning
- USD, Collada, BVH-with-scale exports
- Skin/attachment animation (only bone tracks; texture/visibility tracks are recorded in manifest but not exported)
- Camera animation tracks
- Particle / ribbon / light animation
- **Per-frame rasterization (depth, normal, silhouette, UV-position maps)** — the viewer (`WoWViewer` under `src/viewer/`) already has a working renderer and headless capture surface (`ViewerApp_CaptureAutomation`, `ScreenshotRenderer`, `AssetExporter`). Any image pipeline must route through that surface, not a new one. This is a v2 concern and lives in a separate spec once v1 lands.
- **2D keypoint extraction (OpenPose / COCO format)** — requires either a 3D→2D projection (cheap, in-tool) or a learned 2D pose estimator (heavy, Python). The 3D→2D projection path is the natural v2 add-on once per-frame image capture is in place; the learned path is out of scope indefinitely.
- **UniRig-specific skeleton binding** — the pose clip is exported in Mixamo-normalized space. UniRig (if/when it shows up) is a downstream consumer that maps Mixamo → UniRig. The tool does not own UniRig coupling.
- Multi-model retargeting (taking an animation from model A and applying it to model B with a different skeleton)
- GPU acceleration
- Web UI / viewer integration (the core library is exposed; the viewer can consume it later)
- Real-time streaming of poses
- Embedding original M2/MDX in the output (we only export derived data)

### Future (v2) hook — Per-frame conditioning via the existing viewer

The ChatGPT-style "render every frame into depth/normal/silhouette/OpenPose" pipeline is intentionally deferred to v2 because:

1. It requires per-frame animation playback in a headless OpenGL context. The viewer already has this (`ViewerApp_CaptureAutomation`), but reusing it from a CLI tool means wiring a new headless capture entrypoint.
2. OpenPose / COCO keypoints require either a 2D pose estimator (Python, GPU) or a 3D-skeleton→2D projection (cheap, deterministic, but only valid for orthographic / known-camera cases).
3. The output (PNG sequences + BVH + manifest) is large; v1's pose clip is the small, ML-loadable surrogate that proves the upstream pipeline works.

When v2 lands, it will consume v1's pose clip + BVH directly. The v1 outputs are intentionally designed to be the **input** to v2's rasterizer, not a parallel pipeline.

## Open Questions

- **OQ-1**: Should the JSON sidecar be schema-versioned (`schemaVersion: 1`) to allow future evolution? **Default yes** unless the user objects.
- **OQ-2**: BVH traditionally has no scale channel. We record scale in JSON but omit it from BVH. Confirm acceptable.
- **OQ-3**: Should `dump` accept MDX directly, or should it convert MDX to a normalized M2 form first? **Default: accept MDX directly, treat bones/sequences the same as M2.** MDX uses the same sequence table layout.
- **OQ-4 (v2)**: When per-frame rasterization lands, where does the headless capture entrypoint live? Three options: (a) a new `WowViewer.Tool.AnimFarm render` subcommand that boots a headless GL context and reuses `ScreenshotRenderer`; (b) a thin capture-only mode on the existing `WoWViewer` app; (c) a separate `WowViewer.Tool.Rasterize` tool. **Default for v2: option (a)** — keeps the anim farm CLI surface coherent and lets us compose `--format clip --with-bvh --with-depth --with-normal` cleanly. Defer until v1 ships.
- **OQ-5 (v2)**: OpenPose output format. COCO 17-keypoint, OpenPose BODY_25, or a custom WoW-adapted set (extra joints for weapons, off-hand, spine extras)? **Default: COCO 17** as the most common ControlNet input. Confirm when v2 starts.
