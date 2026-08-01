---
description: "Task list for M2/MDX Animation Pose Farm"
---

# Tasks: M2/MDX Animation Pose Farm

**Input**: Design documents from `/specs/053-m2-animation-pose-farm/`
**Prerequisites**: plan.md ✅, spec.md ✅

**Format**: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[USn]**: Which user story this task belongs to

**Path conventions**: `wow-viewer/` is the repo root for this feature (not the workspace root). All paths below are relative to `wow-viewer/`.

---

## Phase 0: Research & Decisions (no code)

**Purpose**: Lock down API contracts and technical choices before any code lands.

**Goal**: Produce `specs/053-m2-animation-pose-farm/research.md` with concrete answers to 7 questions.

- [x] T001 Read `src/core/WowViewer.Core.IO/M2/M2ModelReader.cs:55-120` and document the auto-detection logic (classic vs chunked vs era-1121), the signature sniff, and the exact exception type for bad headers. Record in research.md R-0.1.
- [x] T002 Read `src/core/WowViewer.Core.IO/Mdx/MdxFile.cs:122-2016` and confirm that `Load(path)` populates `Model.Sequences` and `Model.Bones` for a typical `.mdx`. Inspect `MdlBone` and `MdlAnimTrack<T>` to verify the per-bone TRS structure. Record in research.md R-0.2.
- [x] T003 Read `src/core/WowViewer.Core.IO/Files/ArchiveListfileCache.cs:31-100` and document the cache key format, default cache directory, and how `WowViewer.Tool.Inspect archive build-listfile-cache` populates it. Record in research.md R-0.3.
- [x] T004 Read `src/core/WowViewer.Core.Runtime/M2/M2ExternalAnimationRuntime.cs:1-111` and document the search path it uses for `.anim` companions, the `CompanionPath` it builds, and how to wire `Choose` → `Load` from a CLI tool. Record in research.md R-0.4.
- [x] T005 [P] Pick the JSON Schema strategy: ship `contracts/*.schema.json` (JSON Schema Draft 2020-12) + cross-reference from `data-model.md`. Document in research.md R-0.5.
- [x] T006 [P] Pick BVH version (Biovision Hierarchical 1.0) and confirm channel order (Zrotation Xrotation Yrotation after position, per joint). Document in research.md R-0.6.
- [x] T007 [P] Pick FBX 7.4 ASCII writer scope: minimum-viable = `FBXHeaderExtension` + `Objects` (GlobalSettings, Model nodes, Pose::BindPose, AnimationStack, AnimationCurveNode, AnimationCurve) + `Connections`. No blend shapes, no materials. Document in research.md R-0.7.
- [x] T008 Write `specs/053-m2-animation-pose-farm/research.md` summarizing all 7 decisions with citations to source files and line numbers.

**Checkpoint**: research.md exists with concrete answers to all 7 questions. No code yet.

---

## Phase 1: Library Skeleton + PoseSourceLoader (Foundational)

**Purpose**: Stand up the new `WowViewer.Core.Anim` library + test project, prove we can open M2/MDX from disk and from a virtual archive path, and establish path normalization. **Blocks all user stories.**

- [x] T009 [P] Create `src/core/WowViewer.Core.Anim/WowViewer.Core.Anim.csproj` (net10.0, OutputType=Library, ProjectReferences to `WowViewer.Core`, `WowViewer.Core.IO`, `WowViewer.Core.Runtime`).
- [x] T010 [P] Create `tests/WowViewer.Core.Anim.Tests/WowViewer.Core.Anim.Tests.csproj` (xUnit, ProjectReference to `WowViewer.Core.Anim`, plus the three core projects).
- [x] T011 Edit `WowViewer.slnx`: add a `/src/core/Anim/` folder with `WowViewer.Core.Anim.csproj`, and a `/tests/anim/` folder with `WowViewer.Core.Anim.Tests.csproj`.
- [x] T012 [P] [US1] Implement `src/core/WowViewer.Core.Anim/PathNormalizer.cs` with two static methods: `NormalizeForOutput(string absolutePath)` (forward-slash, lowercase) and `AssertNoStalePath(string path)` (throws `InvalidOperationException` if `H:\CLIENTS` is detected, case-insensitive). No I/O.
- [x] T013 [US1] Implement `src/core/WowViewer.Core.Anim/M2AnimationPoseSource.cs`: a `sealed record` holding `M2ModelDocument Model`, `string SourcePath` (normalized), `string SourceFormat` ("classic"|"chunked"|"era1121"), `string ContentHash` (SHA-256 hex).
- [x] T014 [US1] Implement `src/core/WowViewer.Core.Anim/M2PoseSourceLoader.cs` with two static methods (revised per R-0.1: use `M2ModelReaderDispatcher.ReadDetailed` for era detection; map `M2Era1121EraTag` to source format).
- [x] T015 [P] [US1] Mirror for MDX: `MdxAnimationPoseSource.cs` and `MdxPoseSourceLoader.cs` (uses `MdxFile.Load`).
- [x] T016 [P] [US1] Write `tests/WowViewer.Core.Anim.Tests/PathNormalizerTests.cs`: covers forward-slash, lowercase, and `H:\CLIENTS` rejection (12 tests).
- [x] T017 [P] [US1] Write `tests/WowViewer.Core.Anim.Tests/M2PoseSourceLoaderTests.cs`: stale-path rejection, non-existent file, bad magic (5 tests).
- [x] T018 [P] [US1] Write `tests/WowViewer.Core.Anim.Tests/MdxPoseSourceLoaderTests.cs`: same shape as T017 but for MDX (4 tests).
- [x] T019 Run `dotnet test wow-viewer/tests/WowViewer.Core.Anim.Tests/ -c Debug`. **21/21 pass.** Pre-existing PM4 build error fixed (one-line `using` addition in `WowViewer.Core.PM4/Caching/pm4PerFileCacheService.cs:7`); pre-existing test failures in `WowViewer.Core.Tests` (14 fixture-related failures in `ChunkedFileReader.ReadTopLevelChunks`) are unrelated to this feature and are NOT fixed here.

**Checkpoint**: `WowViewer.Core.Anim` compiles, both loaders work, all 21 foundation tests pass. PM4 build unblocked. Pre-existing failures in unrelated tests surfaced to user.

---

## Phase 2: Sequence Alias Resolution + BoneTrackStream Extraction (Foundational)

**Purpose**: Build the data extractor that turns a loaded `M2AnimationPoseSource` into per-bone TRS keyframe streams. **Blocks US1, US2, US5.**

- [ ] T020 [P] [US1] Implement `src/core/WowViewer.Core.Anim/M2ResolvedSequence.cs`: a `sealed record` with `int RequestedSequenceIndex`, `int ResolvedSequenceIndex`, `IReadOnlyList<int> AliasChain`, `bool UsesExternalFile`, `string? CompanionPath`, `M2ExternalAnimationDocument? ExternalPayload`, and `string Source` ("inline"|"external"|"external-unresolved"|"alias").
- [ ] T021 [US1] Implement `src/core/WowViewer.Core.Anim/M2SequenceAliasResolver.cs`: `static IReadOnlyList<M2ResolvedSequence> ResolveAll(M2ModelDocument model, Func<string, byte[]?> externalAnimationLoader)`. For each sequence, calls `M2ExternalAnimationRuntime.TryChoose` (from `src/core/WowViewer.Core.Runtime/M2/M2ExternalAnimationRuntime.cs:7`), follows the alias chain, and for the terminal sequence loads the `.anim` via the injected `externalAnimationLoader` (which is `null` for inline). Records `external-unresolved` if the loader returns `null` and the sequence claims external. Detects alias loops and throws `InvalidDataException` (the caller catches and records as an error per NFR-003).
- [ ] T022 [P] [US1] Implement `src/core/WowViewer.Core.Anim/M2BoneTrackStream.cs`: a `sealed class` with `int BoneIndex`, `IReadOnlyList<M2BoneKeyframe> Keyframes` (each keyframe has `int TimeMs`, `Vector3 Translation`, `Quaternion Rotation`, `Vector3 Scaling`, `bool UsesGlobalSequence`). Constructor enforces `Keyframes` is non-null.
- [ ] T023 [P] [US1] Implement `src/core/WowViewer.Core.Anim/M2BoneTrackStreamExtractor.cs`: `static M2BoneTrackStream Extract(M2ModelDocument model, byte[] payload, int sequenceIndex, int boneIndex)`. Reads `model.Bones[boneIndex].TranslationTrack`, `.RotationTrack`, `.ScalingTrack` (the latter may be `null` for some bones). For each track, walks the array references in the payload directly (do not use `M2TrackSampler` — we want the *raw* keyframes, not interpolated samples). The extractor reads `M2TrackSequenceSlice` for the requested sequence index from the track's `TimestampArray` and `ValueArray` offsets; for global-sequence tracks, uses `model.GlobalLoops[track.GlobalSequenceIndex]` as the wrap period. Emits a single identity keyframe at `t=0` for tracks with no data.
- [ ] T024 [US1] Add `ExtractAll(M2ModelDocument, byte[], int sequenceIndex)` returning `IReadOnlyList<M2BoneTrackStream>` for all bones in bone-index order.
- [ ] T025 [P] [US1] Implement `src/core/WowViewer.Core.Anim/MdxBoneTrackStreamExtractor.cs`: same shape as the M2 extractor but reads `MdlBone.TranslationTrack`, `.RotationTrack`, `.ScalingTrack` (`MdlAnimTrack<T>` from `src/core/WowViewer.Core.IO/Mdx/MdxModels.cs:94`). Filters keys by `MdlSequence.Time.Start..End`. Linear interpolation only in v1.
- [ ] T026 [P] [US1] Write `tests/WowViewer.Core.Anim.Tests/M2SequenceAliasResolverTests.cs`: covers simple alias, deep alias chain, alias loop detection (expect `InvalidDataException`), external file loaded vs unresolved.
- [ ] T027 [P] [US1] Write `tests/WowViewer.Core.Anim.Tests/M2BoneTrackStreamExtractorTests.cs`: covers a single non-empty track, an empty track (identity fallback), a global-sequence track. Real-data: gate on staged client.
- [ ] T028 [P] [US1] Write `tests/WowViewer.Core.Anim.Tests/MdxBoneTrackStreamExtractorTests.cs`: covers MDX tracks filtered by sequence range.
- [ ] T029 `dotnet build` and `dotnet test` green.

**Checkpoint**: Alias chains resolve correctly, keyframe extraction produces sensible streams, all tests pass.

---

## Phase 3: BVH Writer + BVH Reader (US1, US3 dependency)

**Purpose**: Implement the in-memory BVH representation, the writer, and a parser for round-trip tests.

- [ ] T030 [P] [US1] Implement `src/core/WowViewer.Core.Anim/BvhDocument.cs`: `sealed class` with `string Name`, `BvhJoint Root`, `List<BvhMotion> Motions`. `BvhJoint` has `string Name`, `Vector3 Offset`, `List<BvhJoint> Children`, `List<BvhChannel> Channels` (enum: `Xposition`/`Yposition`/`Zposition`/`Zrotation`/`Xrotation`/`Yrotation`). `BvhMotion` has `string Name`, `int FrameCount`, `double FrameTimeMs`, `List<BvhFrame> Frames`. `BvhFrame` has `List<double> ChannelValues` in joint DFS order.
- [ ] T031 [P] [US1] Implement `src/core/WowViewer.Core.Anim/BvhDocumentBuilder.cs`: `static BvhDocument Build(M2AnimationPoseSource source, IReadOnlyList<M2ResolvedSequence> sequences, IReadOnlyList<IReadOnlyList<M2BoneTrackStream>> streamsBySequence)`. Maps the M2 bone hierarchy to a `BvhJoint` tree; bone 0 is root. Joint name = `model.Bones[i].Name` (fallback to `Bone_<i>`). Per joint, emits 3 position channels + 3 rotation channels in the order Z X Y (the BVH standard). Skips scale (per OQ-2). One `BvhMotion` per non-alias sequence.
- [ ] T032 [US1] Implement `src/core/WowViewer.Core.Anim/BvhDocumentWriter.cs`: `static void Write(BvhDocument doc, TextWriter writer)`. Emits Biovision BVH grammar: `HIERARCHY` → `ROOT <name>` with `OFFSET` and `CHANNELS <n>` → recursive `JOINT` or `End Site` → `MOTION` with `Frames:`, `Frame Time:` (converted from ms to seconds), then one line per frame of whitespace-separated channel values. Use `\n` line endings (LF only). Use `CultureInfo.InvariantCulture` for numbers.
- [ ] T033 [US1] Implement `src/core/WowViewer.Core.Anim/BvhDocumentReader.cs`: `static BvhDocument Read(TextReader reader)`. Parses `HIERARCHY` and `MOTION` blocks. Strict format: throws `InvalidDataException` on malformed input. Used only for round-trip tests (NFR-001).
- [ ] T034 [P] [US1] Write `tests/WowViewer.Core.Anim.Tests/BvhDocumentWriterTests.cs`: golden output for a tiny synthetic `BvhDocument` (3 joints, 2 frames). Byte-equal assertion against a checked-in string literal.
- [ ] T035 [P] [US1] Write `tests/WowViewer.Core.Anim.Tests/BvhRoundTripTests.cs`: build → write → read → assert equality of joint hierarchy, frame count, and channel values to within 1e-9.
- [ ] T036 [P] [US1] Write `tests/WowViewer.Core.Anim.Tests/BvhDocumentBuilderTests.cs`: build a small synthetic `M2ModelDocument` (use the in-memory builder from T017's test helper or a new helper), assert the BvhDocument has the expected joint tree and motion count.
- [ ] T037 `dotnet build` and `dotnet test` green.

**Checkpoint**: BVH round-trip is deterministic and bit-equal. BVH files will be valid BVH per the grammar.

---

## Phase 4: Pose Manifest + Skeleton Subcommand (US7, US1)

**Purpose**: Ship the per-model `manifest.json` schema and the first working CLI subcommand (`skeleton`).

- [ ] T038 [P] [US7] Implement `src/core/WowViewer.Core.Anim/PoseManifest.cs`: `sealed class` with all fields from FR-008 — `ModelPath`, `SourceFormat`, `ContentHash`, `BoneCount`, `Bones[]` (`Index`, `Parent`, `Name`, `Pivot [x,y,z]`, `FlagsRaw`), `Sequences[]` (`Index`, `Name`, `AnimationId`, `VariationIndex`, `Duration`, `IsAlias`, `ResolvedSequenceIndex`, `Source`, `FrameCount`). Mark `[JsonSerializable]` for source generation. Property naming: `JsonPropertyName` for snake_case keys.
- [ ] T039 [US7] Implement `src/core/WowViewer.Core.Anim/PoseManifestBuilder.cs`: `static PoseManifest Build(M2AnimationPoseSource source, IReadOnlyList<M2ResolvedSequence> sequences, IReadOnlyList<IReadOnlyList<M2BoneTrackStream>> streamsBySequence)`. Pulls bones from `model.Bones`, sequences from the resolved list, frame counts from the stream lists. For a failed sequence, sets `Source: "error"` and `errorMessage`.
- [ ] T040 [P] [US7] Write `specs/053-m2-animation-pose-farm/contracts/manifest.schema.json` (JSON Schema Draft 2020-12). Required fields only. Reference every field from FR-008.
- [ ] T041 [US7] Create `tools/animfarm/WowViewer.Tool.AnimFarm/WowViewer.Tool.AnimFarm.csproj` (net10.0, Exe, ProjectReferences to `WowViewer.Core.Anim`, `WowViewer.Core`, `WowViewer.Core.IO`, `WowViewer.Tools.Shared`).
- [ ] T042 [US7] Edit `WowViewer.slnx`: add a `/tools/animfarm/` folder with `WowViewer.Tool.AnimFarm.csproj`.
- [ ] T043 [P] [US7] Implement `tools/animfarm/WowViewer.Tool.AnimFarm/UsageText.cs` with `--help` text for the tool and each subcommand. Mirrors the style of `tools/inspect/WowViewer.Tool.Inspect/Program.cs:4220+`.
- [ ] T044 [P] [US7] Implement `tools/animfarm/WowViewer.Tool.AnimFarm/Program.cs`: parses the first arg as the subcommand (`dump`/`batch`/`skeleton`/`--help`/`-h`) and dispatches. Each dispatch is a stub for now (dump and batch exit 1 with "not yet implemented").
- [ ] T045 [US7] Implement `tools/animfarm/WowViewer.Tool.AnimFarm/SkeletonCommand.cs`: parses `--input` and `--output`, creates the output directory, calls `M2PoseSourceLoader.LoadFromFile`, calls `M2SequenceAliasResolver.ResolveAll` (with a no-op external loader for now), builds the manifest via `PoseManifestBuilder`, writes `<output>/manifest.json`, prints a one-line summary to stdout.
- [ ] T046 [P] [US7] Write `tests/WowViewer.Core.Anim.Tests/PoseManifestTests.cs`: golden JSON for a synthetic source. Confirms the serializer produces byte-equal output across two runs (determinism check).
- [ ] T047 [P] [US7] Write `tests/WowViewer.Core.Anim.Tests/SkeletonCommandTests.cs`: end-to-end test using a small synthetic M2 on disk, asserts `<output>/manifest.json` exists and has the expected structure.
- [ ] T048 `dotnet build` and `dotnet test` green. Real-data smoke: `WowViewer.Tool.AnimFarm skeleton --input <staged-3.3.5-orc.m2> --output <tmp>` produces a valid `manifest.json`.

**Checkpoint**: First end-to-end CLI works. `skeleton --help` exits 0. Real-data manifest.json produced for one staged M2.

---

## Phase 5: `dump` Subcommand + BVH Output (US1 — MVP)

**Purpose**: End-to-end P1: `dump` produces BVH files + manifest for a single model. This is the MVP.

- [ ] T049 [P] [US1] Implement `tools/animfarm/WowViewer.Tool.AnimFarm/AnimFarmPaths.cs`: helpers for `SafeOutputRoot(string outputArg)` (creates the dir, rejects if under `H:\CLIENTS`), `RelativeToOutput(string absolutePath, string outputRoot)` (forward-slash, lowercase).
- [ ] T050 [P] [US1] Implement `src/core/WowViewer.Core.Anim/M2ExternalAnimationFilesystemLoader.cs`: implements the `Func<string, byte[]?>` signature for `M2SequenceAliasResolver` — tries to read the `.anim` from the filesystem first, then the staged client root, then returns null. Catches all IO exceptions and returns null (the resolver will record `external-unresolved`).
- [ ] T051 [US1] Implement `tools/animfarm/WowViewer.Tool.AnimFarm/DumpCommand.cs`: parses `--input`, `--output`, `--with-bvh` (default true), `--with-pose-clip` (default true, no-op until P6), `--with-fbx` (default false, no-op until P8), `--include`/`--exclude` (no-op for single-file mode but accepted). Pipeline: load → resolve aliases (with filesystem loader) → extract bone streams per sequence → write `manifest.json` → for each non-alias sequence, write `<sequenceName>.bvh`. Per-sequence errors recorded in the manifest but do not abort the run.
- [ ] T052 [P] [US1] Wire `DumpCommand` into `Program.cs`'s `case "dump":` branch.
- [ ] T053 [US1] Add `src/core/WowViewer.Core.Anim/PoseClipJsonOptions.cs` (or extend an existing `JsonOptions` class) with the deterministic `JsonSerializerOptions`: `WriteIndented = true`, `DefaultIgnoreCondition = WhenWritingNull`, `PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower`, `Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping`, `Converters = { new JsonStringEnumConverter(JsonNamingPolicy.SnakeCaseLower) }`. Used by manifest, pose clip, and library index.
- [ ] T054 [P] [US1] Write `tests/WowViewer.Core.Anim.Tests/DumpCommandTests.cs`: build a small synthetic M2 on disk, run `dump`, assert BVH files exist, assert byte-identical output across two runs (NFR-002 determinism).
- [ ] T055 [P] [US1] Write `tests/WowViewer.Core.Anim.Tests/DeterminismTests.cs`: parametrized test that runs `dump` over the same synthetic input twice and asserts SHA-256 equality of the entire output directory.
- [ ] T056 Real-data: `WowViewer.Tool.AnimFarm dump --input <staged-3.3.5-humanoid>.m2 --output <tmp>`. Inspect one BVH: confirm `HIERARCHY` block, joint count matches bone count, `MOTION` block frame count matches the source sequence's keyframe count.
- [ ] T057 Write `specs/053-m2-animation-pose-farm/quickstart.md`: one command to run `dump`, a one-paragraph explanation, expected output tree.
- [ ] T058 [P] [US1] Update `specs/053-m2-animation-pose-farm/spec.md` SC-005, SC-006 with validation results (commands run, frame counts observed).
- [ ] T059 `dotnet build` and `dotnet test` green.

**Checkpoint**: US-1 complete. `dump` produces BVH + manifest for any M2. Two runs = byte-identical output.

---

## Phase 6: Pose Clip Sidecar with Mixamo Normalization (US5 — P1, ML deliverable)

**Purpose**: Emit `.poseclip.json` per sequence with Mixamo-normalized bones. This is the v1 ML deliverable that makes the tool a real "pose library".

- [ ] T060 [P] [US5] Implement `src/core/WowViewer.Core.Anim/MixamoSkeletonMap.cs`: `sealed class` with the 22-bone humanoid layout in declaration order: `["Hips", "Spine", "Spine1", "Spine2", "Neck", "Head", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"]`. Static `MixamoSlotNames` property returns this list. Hand-maintained `WoWBoneNameToMixamoSlot` dictionary covers the common WoW humanoid names: `["Spine", "Spine1", "Spine2", "Neck", "Head", "Left Arm", "Right Arm", "Left Forearm", "Right Forearm", "Left Hand", "Right Hand", "Left Shoulder", "Right Shoulder", "Left Leg", "Right Leg", "Left Knee" → LeftLeg, "Right Knee" → RightLeg, "Left Foot", "Right Foot", "Left Toe", "Right Toe", "Left Thigh" → LeftUpLeg, "Right Thigh" → RightUpLeg, "Pelvis" → Hips, "Hip" → Hips, "Root" → Hips]. Normalize input: lowercase, strip underscores/spaces, common-suffix strip ("01", "_L", etc.).
- [ ] T061 [P] [US5] Implement `src/core/WowViewer.Core.Anim/RigClassHeuristic.cs`: `enum RigClass { Humanoid, Quadruped, Creature, Inanimate }`. `static RigClass Classify(M2ModelDocument model)`. Logic: collect all bone names (lowercased, no spaces). If both `hips`-like and `leftupleg`-like and `rightupleg`-like exist → Humanoid. Else if any name matches `frontleg|hindleg|legfront|legrear|front_thigh|rear_thigh` → Quadruped. Else if `model.Bones.Count == 0` → Inanimate. Else → Creature.
- [ ] T062 [P] [US5] Implement `src/core/WowViewer.Core.Anim/PoseClipDocument.cs`: `sealed class` matching FR-019. Top-level: `SchemaVersion = 1`, `ModelPath`, `SequenceIndex`, `SequenceName`, `AnimationId`, `VariationIndex`, `DurationMs`, `FrameCount`, `SkeletonTarget = "mixamo"`, `BoneCount`, `SourceHash`, `Tags[]`, `Summary { RootMotionDelta [3], BoundsMin [3], BoundsMax [3] }`, `BoneOrder[]` (the 22 Mixamo slots + any `__raw_*` extras), `Keyframes[]` (each `{ TMs, Bones: float[] of length BoneCount*10 }`), `Extras` (dict: slot → `{ SrcBoneIndex, SrcBoneName, Pivot: float[3] }`).
- [ ] T063 [US5] Implement `src/core/WowViewer.Core.Anim/PoseClipBuilder.cs`: `static PoseClipDocument Build(M2AnimationPoseSource source, M2ResolvedSequence sequence, IReadOnlyList<M2BoneTrackStream> streams, MixamoSkeletonMap skeleton)`. Per keyframe: emit 22 Mixamo slots first (identity TRS for empty), then any `__raw_*` extras. Each bone contributes 10 floats in `[tx, ty, tz, qx, qy, qz, qw, sx, sy, sz]` order. `Extras` records the source bone for each slot. `Summary`: walk all keyframes, compute root motion (Hips translation delta from frame 0 to last), compute AABB (min/max XYZ across all bone world positions per frame — uses `M2BonePoseEvaluator` for world positions, or simplified local if evaluator is too heavy).
- [ ] T064 [P] [US5] Implement `src/core/WowViewer.Core.Anim/PoseTagger.cs`: `static IReadOnlyList<string> Derive(M2AnimationPoseSource source, M2ResolvedSequence sequence, RigClass rigClass)`. Calls `M2AnimationNameResolver.GetSequenceDisplayName(animationId, variationIndex)` to get the family name, derives a primary tag (e.g. "Attack1H" → "attack"), and a weapon tag when applicable (1h/2h/2hl/bow/unarmed/staff). Adds the rig class as a tag. Returns sorted, distinct tags.
- [ ] T065 [US5] Wire `PoseClipBuilder` into `DumpCommand.Run`: after writing the BVH (if `--with-bvh`), write `clip.<sequenceName>.poseclip.json`. Gate on `--with-pose-clip` (default true).
- [ ] T066 [P] [US5] Write `specs/053-m2-animation-pose-farm/contracts/poseclip.schema.json` (JSON Schema Draft 2020-12). Reference every field from FR-019.
- [ ] T067 [P] [US5] Write `tests/WowViewer.Core.Anim.Tests/MixamoSkeletonMapTests.cs`: covers common WoW humanoid names, misspellings, and unknowns.
- [ ] T068 [P] [US5] Write `tests/WowViewer.Core.Anim.Tests/RigClassHeuristicTests.cs`: one test per RigClass. Build a synthetic `M2ModelDocument` with the appropriate bone names.
- [ ] T069 [P] [US5] Write `tests/WowViewer.Core.Anim.Tests/PoseTaggerTests.cs`: covers each family (walk, run, attack1h, attack2h, spell, death, idle) and the rig class tag.
- [ ] T070 [P] [US5] Write `tests/WowViewer.Core.Anim.Tests/PoseClipBuilderTests.cs`: golden JSON for a tiny synthetic input (1 bone, 2 keyframes). Asserts byte-equal output.
- [ ] T071 Real-data: dump one humanoid M2 from the staged 3.3.5 client. Open `clip.*.poseclip.json` in Python, `json.load` it, convert `keyframes[0].bones` to `np.array` of shape `(22, 10)`. Assert bone 0 (Hips) is non-identity.
- [ ] T072 `dotnet build` and `dotnet test` green.

**Checkpoint**: US-5 complete. Pose clip is JSON-parseable in Python, has the expected `(N, 22, 10)` shape, contains real humanoid bone data for a real M2.

---

## Phase 7: `batch` Subcommand + Library Index + Errors Log (US2, US4, US6)

**Purpose**: Corpus-mining deliverable. Walk a staged client, write per-model subdirs, aggregate into `library.index.json`, log failures to `errors.jsonl`.

- [ ] T073 [P] [US2] Implement `tools/animfarm/WowViewer.Tool.AnimFarm/ErrorsJsonlWriter.cs`: `sealed class` wrapping a `StreamWriter` in append mode. `Write(AnimFarmError error)` serializes one `AnimFarmError` (record: `string ModelPath`, `string Error`, `string ErrorType`) as a single JSON line with no trailing comma. Auto-flushes.
- [ ] T074 [P] [US2] Implement `tools/animfarm/WowViewer.Tool.AnimFarm/BatchProgressReporter.cs`: `sealed class` with `Report(int current, int total, string modelPath, string status)`. Writes `[current/total] <modelPath> ... <status>\n` to stderr. No-op when `Quiet == true`.
- [ ] T075 [P] [US2] Implement `src/core/WowViewer.Core.Anim/BatchEnumerator.cs`: `static IReadOnlyList<string> Enumerate(string cacheDirectoryPath, string cacheKey, Regex? include, Regex? exclude)`. Calls `ArchiveListfileCache.TryRead` (`src/core/WowViewer.Core.IO/Files/ArchiveListfileCache.cs:35`). Throws a custom `BatchEnumerationException` with a clear "run `wowviewer-inspect archive build-listfile-cache` first" message when the cache is missing.
- [ ] T076 [P] [US2] Implement `src/core/WowViewer.Core.Anim/PoseLibraryIndexEntry.cs`: a `sealed class` with `ModelPath`, `SequenceIndex`, `SequenceName`, `AnimationId`, `VariationIndex`, `DurationMs`, `FrameCount`, `Tags[]`, `BvhPath` (nullable), `FbxPath` (nullable), `PoseClipPath` (nullable), `RootMotionDelta [3]`, `BoundsMin [3]`, `BoundsMax [3]`. `JsonPropertyName` for snake_case.
- [ ] T077 [P] [US2] Implement `src/core/WowViewer.Core.Anim/PoseLibraryIndex.cs`: `sealed class` with `string SchemaVersion = "1"`, `string GeneratedAtUtc` (formatted in `en-US`), `int TotalClips`, `IReadOnlyList<PoseLibraryIndexEntry> Clips`. Sorted by `(ModelPath, SequenceIndex)` before serialization.
- [ ] T078 [US2] Implement `src/core/WowViewer.Core.Anim/PoseLibraryIndexBuilder.cs`: a builder that accumulates entries as models are processed. `Add(modelPath, entry)` adds a single entry. `Build()` returns the sorted `PoseLibraryIndex`. Thread-affine (single-threaded in v1).
- [ ] T079 [P] [US2] Write `specs/053-m2-animation-pose-farm/contracts/library-index.schema.json` (JSON Schema Draft 2020-12).
- [ ] T080 [US2] Implement `tools/animfarm/WowViewer.Tool.AnimFarm/BatchCommand.cs`: parses `--client-root`, `--cache-key`, `--cache-dir` (default `<output>/listfile-cache/`), `--output`, `--include`, `--exclude`, `--with-bvh`, `--with-pose-clip`, `--with-fbx`, `--limit`, `--quiet`. Pipeline: enumerate → for each model, load via `M2PoseSourceLoader.LoadFromVirtualFile(client-root)` → run the same dump pipeline as `DumpCommand` (refactor T051's pipeline into a shared `ModelDumper` static class first) → collect entries into `PoseLibraryIndexBuilder` → on any failure, append to `errors.jsonl` and continue → at the end, write `library.index.json` to the batch root.
- [ ] T081 [P] [US2] Refactor T051's dump pipeline into a shared `src/core/WowViewer.Core.Anim/ModelDumper.cs` static class: `static ModelDumpResult Dump(M2AnimationPoseSource source, string outputDir, DumpOptions options)`. Both `DumpCommand` and `BatchCommand` call this.
- [ ] T082 [P] [US2] Wire `BatchCommand` into `Program.cs`'s `case "batch":` branch.
- [ ] T083 [P] [US2] Write `tests/WowViewer.Core.Anim.Tests/BatchEnumeratorTests.cs`: covers filter (M2/MDX), include regex, exclude regex, missing cache (asserts clear error).
- [ ] T084 [P] [US2] Write `tests/WowViewer.Core.Anim.Tests/PoseLibraryIndexBuilderTests.cs`: deterministic sort, byte-identical across two builds.
- [ ] T085 [P] [US2] Write `tests/WowViewer.Core.Anim.Tests/ErrorsJsonlWriterTests.cs`: writes two errors, parses back, asserts valid JSON lines.
- [ ] T086 Real-data: `WowViewer.Tool.AnimFarm batch --client-root <staged-3.3.5> --output <tmp> --limit 50`. Confirm 50 model subdirs written, `library.index.json` is valid JSON with `TotalClips` matching the per-model sequence count sum, `errors.jsonl` exists (some models will fail — expected).
- [ ] T087 [P] [US2] Real-data, real-region: `batch --client-root <staged-3.3.5> --output <tmp> --include "creature/orc/.*"`. Confirm it completes, `library.index.json` has rows, and the Python `tags contains ["walk"]` filter works. This satisfies SC-001.
- [ ] T088 `dotnet build` and `dotnet test` green. Update SC-001, SC-003 in spec.md with results.

**Checkpoint**: US-2, US-4, US-6 complete. `batch` over a real client region works end-to-end. Library index is byte-deterministic.

---

## Phase 8: FBX ASCII Writer (US3 — P2)

**Purpose**: Add FBX as a second output format. Non-blocking for v1, but small enough to bundle.

- [ ] T089 [P] [US3] Implement `src/core/WowViewer.Core.Anim/FbxAsciiDocument.cs`: in-memory representation of an FBX 7.4 ASCII file. Sections: `FBXHeaderExtension` (with `FBXHeaderVersion: 1003`, `FBXVersion: 7400`, `CreationTimeStamp`, `Creator: "WowViewer.Tool.AnimFarm 1.0"`), `Objects` (with `GlobalSettings`, `Model::RootNode` + one `Model` per joint, `Pose::BindPose` with `Model::P` nodes, `AnimationStack` per sequence, `AnimationCurveNode` per (joint, channel) per sequence, `AnimationCurve` per actual keyframe data), `Connections` (a list of `OO`/`OP` records linking everything by numeric UIDs).
- [ ] T090 [P] [US3] Implement `src/core/WowViewer.Core.Anim/FbxAsciiDocumentWriter.cs`: `static void Write(FbxAsciiDocument doc, TextWriter writer)`. Emits FBX 7.4 ASCII grammar (each section is a property list with nested nodes, lines like `NodeName: { Property: "value", Child: { ... } }`). Use `\n` line endings. Use `CultureInfo.InvariantCulture` for numbers.
- [ ] T091 [US3] Implement `src/core/WowViewer.Core.Anim/FbxAsciiDocumentBuilder.cs`: `static FbxAsciiDocument Build(M2AnimationPoseSource source, IReadOnlyList<M2ResolvedSequence> sequences, IReadOnlyList<IReadOnlyList<M2BoneTrackStream>> streamsBySequence)`. Creates one `Model` per joint (named after `BvhJoint.Name`), one `AnimationStack` per non-alias sequence, `AnimationCurveNode` per (joint, channel) per sequence, and `AnimationCurve` per actual keyframe data (KeyTime + KeyValueFloat arrays in FBX time units: `frame / frameRate * 46186158000` for 30fps baseline; v1 uses `tMs * 46186.158` since 1ms = 46186.158 FBX time units at the default 1/46186158000s base).
- [ ] T092 [US3] Wire `FbxAsciiDocumentBuilder` into `ModelDumper` (T081): when `options.WithFbx` is true, write `<sequenceName>.fbx` for each sequence. Wire `--with-fbx` into both `DumpCommand` and `BatchCommand`.
- [ ] T093 [P] [US3] Write `tests/WowViewer.Core.Anim.Tests/FbxAsciiDocumentWriterTests.cs`: golden output for a tiny synthetic document.
- [ ] T094 [P] [US3] Write `tests/WowViewer.Core.Anim.Tests/FbxAsciiDocumentBuilderTests.cs`: builds a small M2, asserts the FBX document has the expected number of Models, AnimationStacks, and AnimationCurves.
- [ ] T095 [P] [US3] Write `tests/WowViewer.Core.Anim.Tests/FbxAsciiReaderTests.cs`: hand-rolled simple FBX ASCII reader (parse the `Objects:` and `Connections:` sections, extract Model and AnimationCurve counts) — used to confirm the writer's output is parseable. No need to round-trip the curves themselves, just the structural counts.
- [ ] T096 Real-data: `WowViewer.Tool.AnimFarm dump --input <staged-humanoid>.m2 --output <tmp> --with-fbx`. Open the resulting `.fbx` in a text editor and confirm it has the expected sections. If Blender is installed, open it and confirm bone count and keyframe count.
- [ ] T097 `dotnet build` and `dotnet test` green. Update spec SC-006 with FBX results.

**Checkpoint**: US-3 complete. `dump --with-fbx` and `batch --with-fbx` work end-to-end. FBX files are syntactically valid FBX 7.4 ASCII.

---

## Phase 9: Architecture Doc + End-to-End Demo + Memory Bank + Final Validation

**Purpose**: Document the system, run a real end-to-end demo, update the memory bank, and validate every success criterion.

- [ ] T098 Write `docs/architecture/m2-anim-pose-farm-2026-06-09.md` with 7 sections per plan.md P9-1. Includes the end-to-end demo: file tree, first 30 lines of a BVH, a snippet of a pose clip, a row from the library index.
- [ ] T099 [P] Update `gillijimproject_refactor/memory-bank/activeContext.md` with a one-paragraph note about the anim farm v1 shipping. (Note: this is the only memory-bank file in this repo and is in the read-only reference codebase, but the AGENTS.md memory bank rule applies to it. Per the memory-bank rule, we update it as a *continuity note*, not as new code.)
- [ ] T100 [P] Update `gillijimproject_refactor/memory-bank/progress.md` with one dated entry: "2026-06-09: anim farm v1 ships — BVH + pose clip + library index; v2 rasterization pending via WoWViewer capture surface."
- [ ] T101 [P] Update `specs/053-m2-animation-pose-farm/spec.md` SC-001 through SC-006 with the actual validation results (commands run, file counts, byte-identical hashes observed).
- [ ] T102 Final test pass: `dotnet build wow-viewer/WowViewer.slnx -c Debug && dotnet test wow-viewer/tests/WowViewer.Core.Anim.Tests/ -c Debug`. All green.
- [ ] T103 [P] Final end-to-end run: `batch --client-root <staged-3.3.5> --output <tmp> --include "creature/orc/.*"`. Confirm file tree, library index, error log. Capture the exact commands and results in the architecture doc.
- [ ] T104 [P] Mark `[x]` for every task in this file once verified. Commit the changes.

**Checkpoint**: All SC-001 through SC-006 demonstrably met. Architecture doc exists. Memory bank updated. Build + test green.

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 0 (research)
  └─→ Phase 1 (library + loaders) ─────── BLOCKS all user stories
        └─→ Phase 2 (alias + extractors) ── BLOCKS US1, US2, US5
              └─→ Phase 3 (BVH) ──────────── BLOCKS US1, US3
                    ├─→ Phase 4 (manifest + skeleton) ── US7
                    │     └─→ Phase 5 (dump) ── US1 (MVP)
                    │           └─→ Phase 6 (pose clip) ── US5
                    │                 └─→ Phase 7 (batch + index) ── US2, US4, US6
                    │                       └─→ Phase 8 (FBX) ── US3
                    │                             └─→ Phase 9 (docs + validation)
```

### Within-Phase Dependencies

- **Phase 1**: T009/T010 (csproj) before T011 (slnx). T011 before T012-T015. T016-T018 (tests) can run in parallel with T014/T015 (production code). T019 last.
- **Phase 2**: T020 (record) before T021 (resolver) before T022-T025 (extractors). T026-T028 (tests) can run in parallel with T023-T025. T029 last.
- **Phase 3**: T030 (record) before T031 (builder) before T032 (writer) before T033 (reader). T034-T036 (tests) parallel to each other. T037 last.
- **Phase 4**: T038/T040 (manifest record + schema) before T039 (builder). T041 (csproj) before T042 (slnx). T043/T044 (program) before T045 (skeleton). T046/T047 (tests) parallel. T048 last.
- **Phase 5**: T049/T050 (helpers) before T051 (dump). T052 (wire) after T051. T053 (json options) parallel. T054/T055 (tests) parallel. T056-T058 (real-data + docs) after T051. T059 last.
- **Phase 6**: T060-T062 (data classes) before T063-T064 (builder + tagger). T065 (wire) after T063. T066 (schema) parallel. T067-T070 (tests) parallel. T071 (real-data) after T065. T072 last.
- **Phase 7**: T073-T075 (helpers) parallel. T076/T077 (records) parallel. T078 (builder) after T076. T079 (schema) parallel. T080 (batch) after T073-T078. T081 (refactor) before T080. T082 (wire) after T080. T083-T085 (tests) parallel. T086/T087 (real-data) after T080. T088 last.
- **Phase 8**: T089 (record) before T090 (writer) before T091 (builder). T092 (wire) after T091. T093-T095 (tests) parallel. T096 (real-data) after T092. T097 last.
- **Phase 9**: T098 (doc) first. T099-T101 (updates) parallel. T102 (test) before T103 (final run). T104 (mark) last.

### Parallel Opportunities (high-value)

- **Phase 1**: T009, T010, T012, T015 (csprojs + PathNormalizer + MdxPoseSourceLoader — different files)
- **Phase 2**: T020, T022, T025 (records + Mdx extractor — different files)
- **Phase 3**: T030, T031 (data + builder — different files); T034, T035, T036 (tests — different files)
- **Phase 4**: T038, T040, T043, T046, T047 (records, schema, usage, tests — different files)
- **Phase 5**: T049, T050, T053, T054, T055 (helpers + tests — different files)
- **Phase 6**: T060, T061, T062, T066, T067, T068, T069, T070 (data classes, schema, tests — different files)
- **Phase 7**: T073, T074, T075, T076, T077, T079, T083, T084, T085 (helpers, records, tests — different files)
- **Phase 8**: T089, T090, T093, T094, T095 (data, writer, tests — different files)
- **Phase 9**: T099, T100, T101, T103 (memory bank + spec + final run — different files)

---

## Implementation Strategy

### MVP First (P1 user stories only)

1. Complete Phase 0 (research) — required pre-work
2. Complete Phase 1 (library + loaders) — required foundation
3. Complete Phase 2 (alias + extractors) — required foundation
4. Complete Phase 3 (BVH) — required for US1
5. Complete Phase 4 (manifest + skeleton) — first end-to-end CLI
6. Complete Phase 5 (`dump`) — **US-1 MVP ships here**
7. **STOP and VALIDATE**: `dump` over a real humanoid M2 produces a BVH + manifest that round-trips.

### Incremental Delivery

After MVP:
- Add US-5 (pose clip) via Phase 6
- Add US-2/US-4/US-6 (batch + index) via Phase 7
- Add US-3 (FBX) via Phase 8
- Final docs and validation via Phase 9

### Parallel Strategy

For a single developer: execute serially per the dependency order above.

For a team of 2-3:
- Dev A: Phases 1-5 (library → dump → US-1)
- Dev B (after Phase 3 done): Phases 6 (pose clip)
- Dev C (after Phase 5 done): Phases 7-8 (batch, FBX)
- All: Phase 9 together

---

## Notes

- **One concern per task**: each task writes one or a small set of related files for one reason.
- **Each task is independently completable**: every task is small enough to land, build, and test in isolation.
- **Tests are interleaved with production code**: each phase has its own test tasks. Tests are written alongside (or just after) the production code they cover, not deferred to the end.
- **Real-data validation is explicit**: T017, T027, T056, T071, T086, T087, T096, T103 each name a staged-3.3.5 client path and a concrete expected output.
- **Determinism is enforced**: T055 specifically tests SHA-256 equality of the entire output directory across two runs. Other determinism checks are baked into the JSON serializer (T053) and the BVH writer (T032).
- **No `H:\CLIENTS`**: T012's `AssertNoStalePath` enforces this at the library level; the CLI surface uses `PathNormalizer` consistently.
- **No gillijimproject_refactor writes**: T099, T100 update memory-bank files but do not add new code there. This is the only kind of write allowed to that codebase per RULE 1, and it matches the AGENTS.md memory-bank rule.
- **104 tasks across 9 phases**, all bite-sized (no task is more than a single-file change in most cases; the few multi-file tasks are the slnx edits and the architecture doc, which are clearly scoped).
