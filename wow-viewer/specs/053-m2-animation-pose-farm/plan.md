# Implementation Plan: M2/MDX Animation Pose Farm

**Branch**: `053-m2-animation-pose-farm` | **Date**: 2026-06-09 | **Spec**: `specs/053-m2-animation-pose-farm/spec.md`

## Summary

Build a new library `WowViewer.Core.Anim` and a new CLI tool `WowViewer.Tool.AnimFarm` that farm bone animations from M2/MDX models in a staged WoW client and emit BVH motion files, a Mixamo-normalized pose clip sidecar (`.poseclip.json`), and a tagged `library.index.json`. The library glues together the existing `M2ModelReader`, `M2ExternalAnimationRuntime`, `M2TrackSampler`, `M2BonePoseEvaluator`, and `M2AnimationNameResolver` — no new format readers. The CLI is a thin wrapper that walks either a single model or the listfile cache and orchestrates the library. Per-frame image rasterization is explicitly v2 and will route through the existing `WoWViewer` capture surface, not a new renderer.

## Technical Context

**Language/Version**: C# / .NET 10 (matches the rest of `wow-viewer/`).

**Primary Dependencies** (existing, all already in repo):
- `WowViewer.Core` — `M2ModelDocument`, `M2SequenceDefinition`, `M2BoneDefinition`, `M2AnimationBlocks`, `M2ExternalAnimationDocument`, `M2AnimationNameResolver`, `M2ModelIdentity`, `M2GlobalLoops`.
- `WowViewer.Core.IO` — `M2ModelReader`, `M2AnimationReader`, `M2TrackSampler` (sampling), `M2Era1121ModelReader` (legacy era), `M2ChunkedModelReaderDispatcher` (chunked M2), `MdxFile.Load` (MDX), `ArchiveVirtualFileReader` + `ArchiveListfileCache` (batch enumeration).
- `WowViewer.Core.Runtime` — `M2ExternalAnimationRuntime` (alias + .anim resolution), `M2BonePoseEvaluator` (world transforms).

**New code only**:
- `WowViewer.Core.Anim` library (new project under `src/core/`)
- `WowViewer.Tool.AnimFarm` CLI (new project under `tools/animfarm/`)
- `WowViewer.Core.Anim.Tests` xUnit project (new project under `tests/`)

No new NuGet packages. No new format readers. No new renderers. The `WoWViewer` viewer app's capture surface is **not** consumed in v1 (it is the future v2 rasterization host).

**Storage**: Plain files on disk under a user-supplied `--output` directory. Per-model subdirectory contains BVH/FBX/`.poseclip.json`/`manifest.json`. Batch root contains `library.index.json` and `errors.jsonl`. **No database, no Zarr, no NPZ.** This is intentional — the pose clip JSON is itself ML-loadable via `json.load` + `np.array` (FR-020). The harvester's Zarr pipeline is for terrain tiles, not pose data.

**Testing**: xUnit (`WowViewer.Core.Anim.Tests`), mirroring the pattern in `WowViewer.Core.Tests` (e.g. `M2Era1121ModelReaderTests`, `M2FoundationTests`). Real-data fixtures from `gillijimproject_refactor/test_data/development/World/Maps/development` are off-limits (RULE 1) — we use the staged 3.3.5 client at `I:\parp\parp-tools\output\tmp\wowarchive-clients\` (RULE 9) for any end-to-end tests.

**Target Platform**: Windows .NET 10 (matches the rest of `wow-viewer/`). No cross-platform concerns beyond what the rest of the repo already handles.

**Project Type**: Library + CLI tool. Mirrors `WowViewer.Core` + `WowViewer.Tool.Harvest`.

**Performance Goals**: Not a v1 concern (spec assumption: a 3.3.5 client with ~30k models may take hours). Single-threaded; users shard across client regions.

**Constraints**:
- Determinism (NFR-002): fixed culture `en-US`, all collections sorted, byte-identical outputs across runs.
- Robustness (NFR-003): batch never throws on a single bad model — errors go to `errors.jsonl`, run continues.
- No `H:\CLIENTS` (RULE 9 / FR-014): all client paths go through the staged wowarchive-clients tree.

**Scale/Scope**: One library, one CLI, one test project. Roughly 1.5k–2.5k LOC of new C#.

## Constitution Check

| Principle | Status | Notes |
|---|---|---|
| I. Repo Independence | ✅ | `WowViewer.Core.Anim` lives under `wow-viewer/src/core/`; only references `WowViewer.Core`, `WowViewer.Core.IO`, `WowViewer.Core.Runtime`. No project references outside `wow-viewer/`. |
| II. Library-First | ✅ | All pose-extraction logic lives in `WowViewer.Core.Anim`. The CLI is a thin wrapper. |
| III. Real-Data Validation | ✅ | Every phase ends with a real-data validation step against the staged 3.3.5 client. |
| IV. Residual Model Chain | N/A | This feature is data, not ML. No model chain concerns. |
| V. Streaming-First Dataset Pipeline | ✅ by simplification | No streaming needed; per-file JSON/BVH is the v1 contract. Zarr/NPZ are out of scope. The pose clip is single-object `json.load`. |
| VI. No Game Client Path Assumptions | ✅ | FR-014 forbids `H:\CLIENTS`. All client access is via staged wowarchive-clients path. |

**Safety constraints**:
- Read-Only Reference Codebase (RULE 1): Plan does not touch `gillijimproject_refactor`.
- Format Reader/Writer Ownership (RULE 3): Plan does not add new M2/MDX/anim readers. All reads go through existing readers.
- Terrain Alpha Risk Area: N/A — no terrain code.
- `AlphaWdtWriter` Frozen (RULE 10): N/A.

**Development workflow**:
- One Phase at a Time (RULE 8): phases are bite-sized (3-7 steps each), each validated before the next.
- Spec Docs Are Source of Truth (RULE 11): `specs/053-m2-animation-pose-farm/spec.md` is the source of truth. Architecture doc `docs/architecture/m2-anim-pose-farm-2026-06-09.md` will be written in phase 1.
- Memory Bank Discipline (RULE 11): `activeContext.md` and `progress.md` get a one-paragraph update at the end of each phase.

## Project Structure

### Documentation (this feature)

```text
specs/053-m2-animation-pose-farm/
├── spec.md              # The feature specification
├── plan.md              # This file
├── research.md          # Phase 0 output (filled by Phase 0)
├── data-model.md        # Phase 1 output (filled by Phase 1) — schemas
├── quickstart.md        # Phase 1 output (filled by Phase 1) — one demo command
├── contracts/           # Phase 1 output (filled by Phase 1) — JSON Schemas
│   ├── manifest.schema.json
│   ├── poseclip.schema.json
│   └── library-index.schema.json
└── tasks.md             # Phase 2 output (`$speckit-tasks`)
```

### Source Code (repository root)

```text
wow-viewer/
├── src/
│   └── core/
│       └── WowViewer.Core.Anim/                    # NEW library
│           ├── WowViewer.Core.Anim.csproj
│           ├── M2AnimationPoseSource.cs            # loaded model + resolved sequence metadata
│           ├── M2PoseSourceLoader.cs               # bridges ArchiveVirtualFileReader → M2/MDX
│           ├── M2BoneTrackStream.cs                # per-bone TRS keyframe stream for one sequence
│           ├── M2BoneTrackStreamExtractor.cs       # walks M2 track defs via M2TrackSampler
│           ├── M2SequenceAliasResolver.cs          # alias chain → terminal sequence
│           ├── BvhDocument.cs                      # in-memory Biovision representation
│           ├── BvhDocumentWriter.cs                # BVH serializer
│           ├── BvhDocumentReader.cs                # BVH parser (for NFR-001 round-trip tests)
│           ├── FbxAsciiDocument.cs                 # minimal ASCII FBX 7.4 representation
│           ├── FbxAsciiDocumentWriter.cs           # FBX serializer
│           ├── MixamoSkeletonMap.cs                # 22-bone humanoid layout + WoW bone name → slot map
│           ├── PoseClipBuilder.cs                  # builds PoseClipDocument from a BoneTrackStream
│           ├── PoseClipDocument.cs                 # JSON-serializable pose clip
│           ├── PoseManifest.cs                     # per-model manifest
│           ├── PoseLibraryIndex.cs                 # batch top-level index
│           ├── PoseLibraryIndexBuilder.cs          # collects clips into the index
│           ├── PoseTagger.cs                       # derives tags from (model, animationId, bones)
│           ├── RigClassHeuristic.cs                # humanoid|quadruped|creature|inanimate
│           ├── PoseClipSchema.cs                   # JSON schema constants
│           └── PathNormalizer.cs                   # forward-slash, lowercase, no H:\CLIENTS
│
├── tools/
│   └── animfarm/
│       └── WowViewer.Tool.AnimFarm/                # NEW CLI
│           ├── WowViewer.Tool.AnimFarm.csproj
│           ├── Program.cs                          # entry: dispatch to dump|batch|skeleton
│           ├── AnimFarmPaths.cs                    # path normalization + safe output root
│           ├── DumpCommand.cs                      # `dump` subcommand
│           ├── BatchCommand.cs                     # `batch` subcommand
│           ├── SkeletonCommand.cs                  # `skeleton` subcommand
│           ├── BatchProgressReporter.cs            # logs to stderr; one line per model
│           ├── ErrorsJsonlWriter.cs                # append-only error sink
│           └── UsageText.cs                        # `--help` text
│
├── tests/
│   └── WowViewer.Core.Anim.Tests/                  # NEW xUnit project
│       ├── WowViewer.Core.Anim.Tests.csproj
│       ├── BvhDocumentWriterTests.cs
│       ├── BvhRoundTripTests.cs
│       ├── FbxAsciiDocumentWriterTests.cs
│       ├── MixamoSkeletonMapTests.cs
│       ├── PoseClipBuilderTests.cs
│       ├── PoseManifestTests.cs
│       ├── PoseLibraryIndexBuilderTests.cs
│       ├── PoseTaggerTests.cs
│       ├── RigClassHeuristicTests.cs
│       ├── M2SequenceAliasResolverTests.cs
│       ├── DeterminismTests.cs                     # re-running same inputs = byte-identical output
│       └── RealDataSmokeTests.cs                   # gated on staged 3.3.5 client presence
│
├── docs/
│   └── architecture/
│       └── m2-anim-pose-farm-2026-06-09.md         # NEW — end-to-end demo + design notes
│
├── WowViewer.slnx                                   # edit to add the new projects
└── (existing structure unchanged)
```

**Structure Decision**: New library + new CLI + new test project, all under existing top-level dirs. No new top-level dirs are introduced. `WowViewer.slnx` gets three new `<Project Path="..." />` entries in a new `/tools/animfarm/` folder node and a new `/tests/anim/` folder node.

## Implementation Phases

Phases are ordered by dependency. Each phase ends with a validation step and produces a self-contained, runnable artifact. The constitution's "one phase at a time" rule means each phase must validate before the next starts.

### Phase 0 — Research & Decisions (no code)

**Goal**: Lock down the technical choices that the rest of the plan depends on. The spec leaves a few as open questions or notes them as "default"; this phase commits them or surfaces blockers.

**Tasks** (small, ≤10):
1. **R-0.1** — Confirm the exact API contract for `M2ModelReader.Read(Stream, sourcePath)`: which file formats it auto-detects (classic M2 vs chunked M2 vs era-1121), whether it throws on bad headers, and whether it requires `.m2` extension or accepts `.mdx` too. (Initial read of `src/core/WowViewer.Core.IO/M2/M2ModelReader.cs` shows it accepts `.m2` and uses `MD20` signature detection internally — confirm chunked-M2 path is exercised.)
2. **R-0.2** — Confirm `MdxFile.Load(path)` returns a `MdxFile` with `Model.Sequences` (List<MdlSequence>) and `Model.Bones` (List<MdlBone>) populated for a typical `.mdx`. Verify the per-bone TRS tracks (`MdlBone.TranslationTrack` etc.) match the M2 track structure closely enough for a shared extractor.
3. **R-0.3** — Confirm `ArchiveListfileCache.TryRead(cacheDir, cacheKey)` returns the entries; check how `WowViewer.Tool.Inspect archive build-listfile-cache` populates the cache (what it writes as `cacheKey` and where the cache file lives by default).
4. **R-0.4** — Confirm `M2ExternalAnimationRuntime.Choose` returns a `CompanionPath` and the search path it uses. (From prior read: companion path is built via `M2ModelIdentity.BuildAnimationPath`.) The batch tool must construct a working `M2ExternalAnimationRuntimeState.Load(animDoc)` so the same search path the viewer uses is honored.
5. **R-0.5** — Decide on the JSON Schema strategy. Two options: (a) ship `contracts/*.schema.json` files that Python validators can use; (b) document the schema in `data-model.md` and rely on round-trip tests. **Default: (a) + (b)** — schemas are useful for downstream consumers and cheap to write.
6. **R-0.6** — Decide the BVH version and channel conventions. BVH 1.0 / Biovision Hierarchical is the safe pick. Confirm position-then-rotation order: per BVH spec, channels are declared in the order they appear in the joint's `CHANNELS` line, not position-then-rotation globally. The `BvhDocumentWriter` must emit Zrotation Xrotation Yrotation per joint in the order most BVH consumers expect (Blender, bvh-python) — which is Z X Y rotations after position, matching the spec's FR-007.
7. **R-0.7** — Decide the FBX 7.4 ASCII writer's scope. Minimum viable FBX has: `FBXHeaderExtension`, `Objects` section with `Model` (limb node) and `Pose` (bind pose) and `AnimationStackNode` + `AnimationCurveNode` + `AnimationCurve` for each animated channel, `Connections` linking them. The writer must produce a file that Blender's FBX importer can open — test with a known small file in `RealDataSmokeTests`.
8. **R-0.8** — Write `research.md` summarizing the above decisions.

**Validation**: `research.md` exists and the 7 items above each have a concrete answer. No code yet.

---

### Phase 1 — Library skeleton + PoseSourceLoader + path normalization

**Goal**: Stand up the new library project, wire it to existing projects, and prove we can open an M2 or MDX from either a filesystem path or a virtual archive path and get back a `M2AnimationPoseSource` (or `MdxAnimationPoseSource`). Establish the path normalization contract.

**Tasks**:
1. **P1-1** — Add `WowViewer.Core.Anim.csproj` (net10.0, references the three existing core projects). Add it to `WowViewer.slnx` under a new `/src/core/Anim/` folder node.
2. **P1-2** — Add `WowViewer.Core.Anim.Tests.csproj` (xUnit, references the new library). Add to `WowViewer.slnx` under a new `/tests/anim/` folder node.
3. **P1-3** — Implement `PathNormalizer` with two methods: `NormalizeForOutput(string absolutePath)` (returns forward-slash, lowercase, no `H:\CLIENTS`) and `AssertNoStalePath(string)` (throws if `H:\CLIENTS` is detected). Pure functions, no I/O.
4. **P1-4** — Implement `M2AnimationPoseSource` (record-style class holding `M2ModelDocument Model`, `string SourcePath`, `string SourceFormat` ("classic"|"chunked"|"era1121"), `string ContentHash`). Just the data class for now.
5. **P1-5** — Implement `M2PoseSourceLoader.LoadFromFile(string path)` and `LoadFromVirtualFile(string virtualPath, IEnumerable<string> archiveRoots)`. The former opens the file and delegates to `M2ModelReader.Read(stream, sourcePath)`. The latter uses `ArchiveVirtualFileReader.ReadVirtualFile` to fetch bytes, wraps in a `MemoryStream`, then calls `M2ModelReader.Read`. Both compute SHA-256 of the source bytes and return an `M2AnimationPoseSource`. Detect chunked M2 by signature sniff (`AFM2` magic).
6. **P1-6** — Mirror the above for MDX: `MdxAnimationPoseSource`, `MdxPoseSourceLoader.LoadFromFile` / `LoadFromVirtualFile` (use `MdxFile.Load`). Place in the same `WowViewer.Core.Anim` project for now (small enough; can split later if needed).
7. **P1-7** — Add `M2PoseSourceLoaderTests.cs` with at least: (a) load a known M2 from disk, assert `SourceFormat` is correct; (b) load from a fake virtual path that throws, assert the exception is wrapped with a clear message; (c) `PathNormalizerTests.cs` covering the `H:\CLIENTS` rejection.
8. **P1-8** — Build the solution: `dotnet build wow-viewer/WowViewer.slnx -c Debug`. Fix any compile errors. Run `dotnet test wow-viewer/tests/WowViewer.Core.Anim.Tests/ -c Debug` and confirm green.

**Validation**:
- `dotnet build wow-viewer/WowViewer.slnx -c Debug` exits 0.
- `dotnet test wow-viewer/tests/WowViewer.Core.Anim.Tests/ -c Debug` passes.
- `M2PoseSourceLoader.LoadFromFile` can open one M2 from the staged 3.3.5 client (one manual smoke call, no test infra required yet).

---

### Phase 2 — Sequence alias resolution + BoneTrackStream extraction

**Goal**: For a loaded `M2AnimationPoseSource`, resolve all sequence aliases to their terminal sequences and extract the native keyframe TRS for every bone. This is the data the BVH/FBX/poseclip writers will consume.

**Tasks**:
1. **P2-1** — Implement `M2SequenceAliasResolver.ResolveAll(M2ModelDocument model)` returning `IReadOnlyList<M2ResolvedSequence>` where each entry has `RequestedSequenceIndex`, `ResolvedSequenceIndex`, `AliasChain`, `UsesExternalFile`, `CompanionPath?`, `ExternalPayload?` (loaded `.anim` bytes if applicable, else null). Reuse `M2ExternalAnimationRuntime.Choose` + `Load` for the per-sequence resolution.
2. **P2-2** — Add `M2ExternalAnimationPayloadResolver` (or extend `M2SequenceAliasResolver`) that, given a resolved sequence and a list of search-path providers (filesystem root, MPQ roots), tries to load the companion `.anim` bytes. For v1 the search path is just the model directory + any `--archive-root` paths the user passed; later we can plumb MPQ catalog. The current `M2ExternalAnimationRuntime.Load` requires a pre-loaded `M2ExternalAnimationDocument`, so the resolver wraps that.
3. **P2-3** — Implement `M2BoneTrackStreamExtractor.Extract(M2ModelDocument model, byte[] payload, int resolvedSequenceIndex, int boneIndex)` returning `M2BoneTrackStream` for one bone in one sequence. The extractor walks the bone's `TranslationTrack`, `RotationTrack`, `ScalingTrack` via `M2TrackSampler.SampleVector3` / `SampleCompressedQuaternion` / `SampleVector3` at the **track's own keyframe timestamps**, not by sampling at the sequence duration. This is the "native keyframes only" requirement (FR-005).
   - For a `M2TrackDefinition<T>` with no data (empty arrays), the extractor emits a single identity keyframe at `t=0` (this matches what `M2BonePoseEvaluator` falls back to).
   - For a track that references a global sequence (`UsesGlobalSequence == true`), the timestamps are clipped/wrapped to `GlobalLoops[GlobalSequenceIndex]`. The extractor emits keyframes at their raw timestamps and tags the stream with `UsesGlobalSequence` so the writer can decide.
4. **P2-4** — Implement `M2BoneTrackStreamExtractor.ExtractAll` returning a list of streams for all bones in the model, in bone-index order, with a parallel `IReadOnlyList<int> BoneIndices`.
5. **P2-5** — MDX path: implement `MdxBoneTrackStreamExtractor.Extract(MdxFile mdx, int sequenceIndex, int boneIndex)` for one bone in one sequence. The MDX tracks use `MdlAnimTrack<T>` with `MdlTrackKey<T>` (frame + value + in/out tangents), so the extractor reads the per-sequence range (`MdlSequence.Time.Start..End`) and filters keys by frame. The tangent data is preserved as a sidecar but not used in v1 (linear interpolation only).
6. **P2-6** — Tests: `M2SequenceAliasResolverTests` (alias chain, alias loop, terminal sequence), `M2BoneTrackStreamExtractorTests` (one stream, global sequence, no-data fallback). For real-data tests, use the staged 3.3.5 client and skip if missing.
7. **P2-7** — Build + test.

**Validation**:
- All existing tests still pass.
- New tests pass.
- Manual smoke: extract bone streams from one humanoid M2 and confirm `frameCount` matches the source's per-bone keyframe count.

---

### Phase 3 — BVH writer + BVH reader (round-trip)

**Goal**: Serialize one model's worth of `M2BoneTrackStream` per sequence as a Biovision BVH file. Include a parser so we can round-trip.

**Tasks**:
1. **P3-1** — Implement `BvhDocument` (in-memory): `string Name`, `BvhJoint Root`, `List<BvhMotion> Motions` (one per sequence). `BvhJoint` has `string Name`, `double OffsetX/Y/Z`, `List<BvhJoint> Children`, `List<string> Channels` (e.g. `"Xposition"`, `"Zrotation"`, etc.). `BvhMotion` has `string Name`, `int FrameCount`, `double FrameTime`, `List<BvhFrame> Frames`, where each `BvhFrame` is `List<double> ChannelValues` in joint DFS order.
2. **P3-2** — Implement `BvhDocumentBuilder.Build(M2AnimationPoseSource source, IReadOnlyList<M2ResolvedSequence> sequences, IReadOnlyList<IReadOnlyList<M2BoneTrackStream>> streamsBySequence, BvhJointNaming naming)`. Maps the M2 bone hierarchy to `BvhJoint` tree (single root if model has no parent for bone 0; otherwise the first bone is the root). Joints are named either by `M2BoneDefinition.Name` (if non-empty) or by a generated `Bone_<index>`. The builder emits position-then-rotation channels in the order Zrotation Xrotation Yrotation (the standard "Z first" BVH convention, matching FR-007). Scale is omitted from BVH (per OQ-2).
3. **P3-3** — Implement `BvhDocumentWriter.Write(BvhDocument doc, TextWriter writer)`. Standard BVH syntax: `HIERARCHY` header, `ROOT <name>` with `OFFSET` and `CHANNELS`, `JOINT` / `End Site` recursively, `MOTION` section with `Frames: <count>`, `Frame Time: <seconds>`, then one line per frame of whitespace-separated channel values.
4. **P3-4** — Implement `BvhDocumentReader.Read(TextReader reader)` for round-trip tests. Strip trailing whitespace, parse the `HIERARCHY` block, parse the `MOTION` block. Reject anything that doesn't match the format.
5. **P3-5** — Tests: `BvhDocumentWriterTests` (golden output for a tiny synthetic BvhDocument), `BvhRoundTripTests` (write then read, assert equality of hierarchy + frame count + channel values to within 1e-6).
6. **P3-6** — Build + test.

**Validation**:
- All tests pass.
- Manual: write one BVH for one humanoid sequence, open in a text editor, confirm the structure is valid BVH (or pipe to `bvh-python` if available).

---

### Phase 4 — Pose manifest + skeleton subcommand

**Goal**: Emit the per-model `manifest.json` (FR-008) and prove it with a `skeleton` subcommand in the CLI. This is the smallest end-to-end CLI surface that doesn't require BVH yet.

**Tasks**:
1. **P4-1** — Implement `PoseManifest` (record-style JSON-serializable): `ModelPath`, `SourceFormat`, `ContentHash`, `BoneCount`, `Bones[]` (`Index`, `Parent`, `Name`, `Pivot [x,y,z]`, `FlagsRaw`), `Sequences[]` (`Index`, `Name`, `AnimationId`, `VariationIndex`, `Duration`, `IsAlias`, `ResolvedSequenceIndex`, `Source` ("inline"|"external"|"external-unresolved"), `FrameCount`).
2. **P4-2** — Implement `PoseManifestBuilder.Build(M2AnimationPoseSource source, IReadOnlyList<M2ResolvedSequence> sequences, IReadOnlyList<IReadOnlyList<M2BoneTrackStream>> streamsBySequence)`. Pulls bone list straight from `M2ModelDocument.Bones`, sequence list from the resolved sequences, frame counts from the streams.
3. **P4-3** — JSON serializer config: `WriteIndented = true`, `NumberHandling = WriteAsString` for floats in [NaN, Infinity] (or use `IgnoreCycles` + custom converter). Use `CultureInfo.InvariantCulture` for all formatting. Sort dictionary keys to ensure determinism (NFR-002).
4. **P4-4** — Write `contracts/manifest.schema.json` describing the manifest. Use JSON Schema Draft 2020-12. Keep it short — just the required fields and types.
5. **P4-5** — Add `WowViewer.Tool.AnimFarm.csproj` (net10.0, references `WowViewer.Core.Anim`, `WowViewer.Core`, `WowViewer.Core.IO`, `WowViewer.Tools.Shared`). Add to `WowViewer.slnx` under `/tools/animfarm/`.
6. **P4-6** — Implement `Program.cs` entrypoint that dispatches `dump` / `batch` / `skeleton`. Each is a stub for now (`dump` and `batch` print "not yet implemented" and exit 1).
7. **P4-7** — Implement `SkeletonCommand` end-to-end: parse `--input`, load via `M2PoseSourceLoader`, resolve aliases, build `PoseManifest`, write to `--output/manifest.json`. Print a one-line summary to stdout.
8. **P4-8** — Tests: `PoseManifestTests` (golden JSON for a synthetic source), `SkeletonCommandTests` (smoke test that the CLI invocation produces the expected file path and content).
9. **P4-9** — Build + test. Manual: `WowViewer.Tool.AnimFarm skeleton --input <staged-3.3.5-orc.m2> --output <tmp>` produces a valid `manifest.json`.

**Validation**:
- All tests pass.
- `WowViewer.Tool.AnimFarm skeleton --help` exits 0 and prints usage.
- Real-data smoke on one humanoid M2 from the staged 3.3.5 client.

---

### Phase 5 — `dump` subcommand + BVH output (US-1)

**Goal**: End-to-end P1: `WowViewer.Tool.AnimFarm dump --input x.m2 --output <dir>` writes one BVH per non-alias sequence plus a manifest. This is User Story 1.

**Tasks**:
1. **P5-1** — Implement `DumpCommand.Run(string[] args)` with `--input`, `--output`, `--with-bvh` (default true), `--with-pose-clip` (default true — for now, just a flag that does nothing; wiring happens in Phase 6), `--with-fbx` (default false), `--include`/`--exclude` regex (used only in `batch`, but accepted here for consistency).
2. **P5-2** — Wire the pipeline: load model via `M2PoseSourceLoader` → resolve aliases via `M2SequenceAliasResolver` (with `.anim` companion lookup) → extract bone streams per sequence → write `manifest.json` → for each non-alias sequence, write `<sequenceName>.bvh`.
3. **P5-3** — Error handling: wrap the per-sequence work in try/catch. A single failing sequence is recorded in the manifest's sequence list with `Source: "error"` and a `errorMessage` field, and the run continues. The overall `dump` command exits non-zero only if loading the model itself fails.
4. **P5-4** — Determinism: sort all collections (sequences by `Index`, bones by `Index`, channel values by track order). Use `en-US` for all number formatting. Use `\n` line endings on the BVH (Boris convention; LF-only is fine per spec, document in `quickstart.md`).
5. **P5-5** — Tests: `DumpCommandTests` with a small synthetic M2 (build a `M2ModelDocument` in-test, write to a temp file, run dump, assert the BVH files are byte-identical across two runs).
6. **P5-6** — Real-data: dump one humanoid M2 from the staged 3.3.5 client. Inspect one BVH in a text editor. Count BVH `MOTION` block frames and compare to the source M2's per-sequence keyframe count. They should match.
7. **P5-7** — Write the architecture doc `docs/architecture/m2-anim-pose-farm-2026-06-09.md` with this end-to-end demo and the BVH/manifest screenshots (text dumps). Update SC-006 in the spec with the result.
8. **P5-8** — Build + test.

**Validation**:
- All tests pass.
- `dump --input <staged-humanoid>.m2 --output <tmp>` produces at least one `.bvh` and one `manifest.json`.
- BVH file passes the `BvhDocumentReader` round-trip test.
- Two runs of `dump` over the same input produce byte-identical output.

---

### Phase 6 — Pose clip sidecar (US-5, P1)

**Goal**: Emit `.poseclip.json` per sequence with Mixamo-normalized bones. This is the v1 ML deliverable.

**Tasks**:
1. **P6-1** — Implement `MixamoSkeletonMap` with the 22-bone humanoid layout (Hips, Spine, Spine1, Spine2, Neck, Head, LeftShoulder, LeftArm, LeftForeArm, LeftHand, RightShoulder, RightArm, RightForeArm, RightHand, LeftUpLeg, LeftLeg, LeftFoot, LeftToeBase, RightUpLeg, RightLeg, RightFoot, RightToeBase). Provide `TryMapWoWBone(string wowBoneName, out string mixamoSlot)` using a name normalization lookup (lowercase, strip underscores, common aliases). The lookup table is hand-maintained and covers the common WoW humanoid bone names.
2. **P6-2** — Implement `RigClassHeuristic.Classify(M2ModelDocument model)` returning `RigClass.Humanoid | Quadruped | Creature | Inanimate`. Heuristic: Humanoid if bones with names matching Hips + LeftUpLeg + RightUpLeg (or L_Thigh/R_Thigh aliases) all exist; Quadruped if any bone suggests front/hind legs (FrontLeg/HindLeg/LegFront/LegRear); else Creature (≥1 bone) or Inanimate (0 bones). Tests cover each case.
3. **P6-3** — Implement `PoseClipDocument` (record-style JSON-serializable). Top-level fields per FR-019: `SchemaVersion: 1`, `ModelPath`, `SequenceIndex`, `SequenceName`, `AnimationId`, `VariationIndex`, `DurationMs`, `FrameCount`, `SkeletonTarget: "mixamo"`, `BoneCount`, `SourceHash`, `Tags[]`, `Summary { RootMotionDelta [x,y,z], BoundsMin [x,y,z], BoundsMax [x,y,z] }`. `BoneOrder[]` is the 22-bone Mixamo slot list. `Keyframes[]` has `{ tMs, bones: [boneCount * 10 floats in order tx,ty,tz,qx,qy,qz,qw,sx,sy,sz per bone] }`. `Extras` maps Mixamo slot → `{ srcBoneIndex, srcBoneName, pivot: [x,y,z] }`. For a creature with no humanoid bones, the standard slots are present but emit identity TRS at the root pose; actual bones are recorded in `extras` (using a special `"__raw_<i>"` slot name to avoid collisions).
4. **P6-4** — Implement `PoseClipBuilder.Build(M2AnimationPoseSource source, M2ResolvedSequence sequence, IReadOnlyList<M2BoneTrackStream> streams, MixamoSkeletonMap skeleton)`. Iterates the streams, for each (boneIndex, keyframe) emits the 10-float TRS. The `bones` array in each keyframe is laid out in `BoneOrder` order (Mixamo slots first, then raw extras in index order). Computes `summary` by walking all keyframes and accumulating root motion (XZ delta of bone 0's translation) and AABB (min/max XYZ across all bone positions per frame).
5. **P6-5** — Implement `PoseTagger.Derive(M2AnimationPoseSource source, M2ResolvedSequence sequence, RigClass rigClass)` returning a deterministic list of tags. Algorithm: take the resolved name from `M2AnimationNameResolver.GetSequenceDisplayName(animationId, variationIndex)`, split into tokens, lowercase, strip numbers, take the family word (e.g. "Attack1H" → "attack"). Add the weapon family (1h/2h/2hl/bow/unarmed) when the family is "Attack" or "Ready" or "Parry". Add the rig class as a tag. Sort the tag list lexicographically for determinism.
6. **P6-6** — Wire `PoseClipBuilder` into `DumpCommand` and `BatchCommand`. The pose clip file is written alongside each BVH (or FBX) as `clip.<sequenceName>.poseclip.json`.
7. **P6-7** — Write `contracts/poseclip.schema.json`.
8. **P6-8** — Tests: `MixamoSkeletonMapTests` (map a known set of WoW humanoid bone names to Mixamo slots), `RigClassHeuristicTests` (one test per RigClass), `PoseClipBuilderTests` (golden JSON for a small synthetic input), `PoseTaggerTests` (one per family).
9. **P6-9** — Real-data: dump one humanoid M2 from the staged 3.3.5 client. Open the resulting `clip.<name>.poseclip.json` in a Python REPL, parse with `json.load`, convert `keyframes[0].bones` to a `np.array` of shape `(22, 10)`, print min/max per channel. Spot-check that bone 0 (Hips) is non-identity.
10. **P6-10** — Build + test.

**Validation**:
- All tests pass.
- `dump --input <staged-humanoid>.m2` produces both `.bvh` and `clip.*.poseclip.json` files.
- The pose clip parses cleanly in Python and has the expected `(N, 22, 10)` shape after `np.array`.
- Two runs produce byte-identical pose clips.

---

### Phase 7 — `batch` subcommand + library index + errors.jsonl (US-2, US-4, US-6)

**Goal**: End-to-end batch run over a staged client root, writing per-model subdirs + a top-level `library.index.json` + `errors.jsonl`. This is the corpus-mining deliverable.

**Tasks**:
1. **P7-1** — Implement `BatchCommand.Run` with args: `--client-root <dir>`, `--cache-key <key>`, `--cache-dir <dir>` (defaults to a per-output `listfile-cache/`), `--output <dir>`, `--include <regex>`, `--exclude <regex>`, `--with-bvh`, `--with-pose-clip`, `--with-fbx`, `--limit <N>` (for testing).
2. **P7-2** — Implement `BatchEnumerator.Enumerate(string clientRoot, string cacheKey, string cacheDir)` that:
   - Calls `ArchiveListfileCache.TryRead(cacheDir, cacheKey)`. If null, abort with a clear error message ("run `wowviewer-inspect archive build-listfile-cache --archive-root <clientRoot> --cache-key <cacheKey> first`").
   - Filters `manifest.AllEntries` to entries ending in `.m2` or `.mdx` (case-insensitive).
   - Applies `--include`/`--exclude` regex filters.
   - Returns a sorted `IReadOnlyList<string>` of virtual paths.
3. **P7-3** — For each virtual path: open via `M2PoseSourceLoader.LoadFromVirtualFile` (using the client root as the archive root). On success, run the dump pipeline. On failure, append a JSON line to `errors.jsonl` (`{modelPath, error, errorType}`) and continue.
4. **P7-4** — Implement `PoseLibraryIndexBuilder` that collects, per model, per sequence: the BVH/FBX/pose clip paths (relative to batch root), the tags, the summary stats. After all models are processed, write `library.index.json` sorted by `(modelPath, sequenceIndex)`, formatted with `en-US` culture, byte-deterministic.
5. **P7-5** — `BatchProgressReporter`: writes one line per model to stderr (`[1234/30000] dumping creature/orc/orc.m2 ... ok 47 sequences`). Suppress under `--quiet`.
6. **P7-6** — Tests: `BatchEnumeratorTests` (filter, regex, missing cache), `PoseLibraryIndexBuilderTests` (deterministic sort, byte-identical across two builds), `ErrorsJsonlWriterTests` (one JSON object per line, valid JSON, doesn't crash on first failure).
7. **P7-7** — Real-data: against the staged 3.3.5 client, run `batch --client-root <staged> --output <tmp> --limit 50`. Confirm 50 model subdirs written, `library.index.json` exists and is valid JSON, `errors.jsonl` exists and is non-empty (some models will fail — that's expected and correct). Open the library index in Python and confirm one row per exported sequence.
8. **P7-8** — Real-data: drop `--limit`, run over a small region (e.g. `Creature/Orc` only, via `--include "creature/orc/.*"`), confirm it completes and produces a useful library. This is the SC-001 validation.
9. **P7-9** — Build + test.

**Validation**:
- All tests pass.
- `batch` over a small subset of the staged 3.3.5 client produces the expected file tree, valid `library.index.json`, and a non-empty `errors.jsonl`.
- `library.index.json` is byte-identical across two runs over the same input (NFR-002).
- A Python consumer can `json.load` the library index and run a `tags contains ["walk"]` filter.

---

### Phase 8 — FBX ASCII writer (US-3)

**Goal**: Add FBX as a second output format. This is P2 and not blocking v1's core, but it's small enough to bundle here.

**Tasks**:
1. **P8-1** — Implement `FbxAsciiDocument` (in-memory). Sections: `FBXHeaderExtension` (with `FBXHeaderVersion: 1003`, `FBXVersion: 7400`, `CreationTimeStamp`, `Creator`), `Objects` (with `GlobalSettings`, `Model::RootNode` + one `Model` per BVH joint, `Pose::BindPose` with `Model::P` nodes, `AnimationStack` per sequence, `AnimationCurveNode` per (joint, channel) per sequence, `AnimationCurve` per actual keyframe data), `Connections` linking everything.
2. **P8-2** — Implement `FbxAsciiDocumentWriter.Write(FbxAsciiDocument doc, TextWriter writer)`. Emits the FBX 7.4 ASCII grammar (each section is a property list with nested nodes). Keep the writer simple — no support for blend shapes, no skeleton root, no materials (those are v2+).
3. **P8-3** — Implement `FbxAsciiDocumentBuilder.Build(M2AnimationPoseSource source, IReadOnlyList<M2ResolvedSequence> sequences, IReadOnlyList<IReadOnlyList<M2BoneTrackStream>> streams)`. For each non-alias sequence, create one `AnimationStack` named after the sequence. For each bone and each animated channel (Tx, Ty, Tz, Rx, Ry, Rz, Sx, Sy, Sz), create an `AnimationCurveNode` and an `AnimationCurve`. The curve's `KeyTime` and `KeyValueFloat` arrays are derived from the bone's TRS stream.
4. **P8-4** — Tests: `FbxAsciiDocumentWriterTests` (golden output for a tiny synthetic input), `FbxAsciiDocumentBuilderTests` (one per rig class).
5. **P8-5** — Real-data: dump a humanoid as FBX, open in Blender (manual), confirm bone count and keyframe count match the source. (If Blender is not available in CI, just confirm the FBX file parses with a hand-rolled simple FBX reader test.)
6. **P8-6** — Wire `--with-fbx` and `--format fbx|both` into `dump` and `batch`. Document in `quickstart.md`.
7. **P8-7** — Build + test.

**Validation**:
- All tests pass.
- FBX output for one humanoid M2 opens in Blender with the expected bone count and keyframe count.
- `dump --with-fbx` and `batch --with-fbx` produce FBX files alongside the BVH/pose clip.

---

### Phase 9 — Architecture doc, end-to-end demo, memory bank update, final validation

**Goal**: Document the system, run a real end-to-end demo, update the memory bank, and validate every success criterion.

**Tasks**:
1. **P9-1** — Write `docs/architecture/m2-anim-pose-farm-2026-06-09.md`:
   - Section 1: Summary and motivation.
   - Section 2: Architecture (library + CLI, dependency diagram).
   - Section 3: Per-feature user stories mapped to the phases that implemented them.
   - Section 4: Output schemas (with reference to `contracts/*.schema.json`).
   - Section 5: End-to-end demo: pick one humanoid M2 from the staged 3.3.5 client, run `dump`, show the resulting file tree, paste the first 30 lines of the BVH, paste a snippet of the pose clip JSON, paste a row from `library.index.json`.
   - Section 6: v2 boundary — what the existing `WoWViewer` capture surface will be used for, with explicit naming of `ViewerApp_CaptureAutomation`, `ScreenshotRenderer`, `AssetExporter`. Update OQ-4 to "default (a) — new subcommand on `WowViewer.Tool.AnimFarm`".
   - Section 7: Known limitations, deferred items, future work.
2. **P9-2** — Update `gillijimproject_refactor/memory-bank/activeContext.md` (one short paragraph: "v1 of the anim farm shipped; poses are dumped in BVH + JSON, with a per-model manifest and a batch library index; v2 rasterization pending via WoWViewer capture surface").
3. **P9-3** — Update `gillijimproject_refactor/memory-bank/progress.md` (one dated entry).
4. **P9-4** — Update `wow-viewer/specs/053-m2-animation-pose-farm/spec.md` SC-001 through SC-006 with the actual validation results (commands run, file counts, byte-identical hashes).
5. **P9-5** — Final test pass: `dotnet build wow-viewer/WowViewer.slnx -c Debug && dotnet test wow-viewer/tests/WowViewer.Core.Anim.Tests/ -c Debug`. All green.
6. **P9-6** — Final end-to-end run: `batch --client-root <staged-3.3.5> --output <tmp> --include "creature/orc/.*"`. Confirm the file tree, library index, and error log match expectations.
7. **P9-7** — Update the spec's "Known bug" note in the repo's memory bank to mention that the anim farm's library index is now available as a stable reference for any future dataset work.

**Validation**:
- All SC-001 through SC-006 are demonstrably met.
- Architecture doc exists and is referenced from the spec.
- Memory bank files are updated and compressed (RULE 11).
- `dotnet build` and `dotnet test` exit 0.

---

## Complexity Tracking

No constitution violations. All principles pass. The plan deliberately:

- Skips Zarr/NPZ (Constitution V is met by *not* using a streaming pipeline — the v1 contract is plain files, which is appropriate for the use case).
- Skips new renderers (Constitution II is met by reusing the existing `WoWViewer` capture surface for any future v2 rasterization, and explicitly saying so in the spec).
- Skips cross-3rd-party FBX writer dependency (hand-rolled ASCII FBX is small and self-contained).

If the user later wants a streaming Zarr-backed variant (for a different use case), that will be a separate spec; this plan stays within the simple-files contract that matches BVH + JSON consumers.

## Phase Summary

| Phase | Focus | LOC est. | Validates |
|---|---|---|---|
| 0 | Research decisions | 0 (just `research.md`) | No code |
| 1 | Library skeleton + PoseSourceLoader | ~400 | Library compiles, loads one M2 from disk |
| 2 | Alias resolution + bone stream extraction | ~500 | One sequence's worth of streams extracted |
| 3 | BVH writer + reader | ~600 | BVH round-trip test green |
| 4 | Manifest + `skeleton` subcommand | ~400 | First end-to-end CLI surface |
| 5 | `dump` subcommand + BVH | ~300 | US-1 done; SC-002, SC-005, SC-006 met |
| 6 | Pose clip sidecar | ~700 | US-5 done; pose clip parses in Python |
| 7 | `batch` + library index + errors.jsonl | ~500 | US-2, US-4, US-6 done; SC-001, SC-003 met |
| 8 | FBX ASCII writer | ~600 | US-3 done |
| 9 | Docs, demo, memory bank, final validation | 0 (docs) | All SC met |

**Total**: ~9 phases, ~4000 LOC of new C# across 1 library + 1 CLI + 1 test project, plus 3 JSON schemas and 1 architecture doc. Each phase is independently validatable and bite-sized per the constitution's RULE 8 / RULE 11. No phase exceeds 10 steps.
