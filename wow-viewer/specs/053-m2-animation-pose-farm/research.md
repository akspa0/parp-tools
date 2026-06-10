# Research: M2/MDX Animation Pose Farm

**Date**: 2026-06-09
**Spec**: `specs/053-m2-animation-pose-farm/spec.md`
**Plan**: `specs/053-m2-animation-pose-farm/plan.md`

This file documents the technical decisions and API contracts discovered during Phase 0 research, before any production code lands.

---

## R-0.1 — M2 reader API contract

**Source**: `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/M2ModelReaderDispatcher.cs:10`

**Discovery**: There IS a real `M2ModelReaderDispatcher` (the spec's plan correctly named it; the plan's T014 in `plan.md` mistakenly said "M2ModelReader.Read" — the correct entry point is `M2ModelReaderDispatcher.Read`, which dispatches to the right era reader). This supersedes any direct `M2ModelReader.Read` calls in the plan.

**API contract**:
- `static M2ModelDocument Read(string path)` — opens the file and dispatches.
- `static M2ModelDocument Read(Stream stream, string sourcePath, Func<string, byte[]?>? companionReader = null)` — accepts a seekable stream. The optional `companionReader` is used to resolve cross-file references (e.g. skin profiles) but **NOT** `.anim` files. Anim files are resolved separately via `M2ExternalAnimationRuntime`.
- `static M2DispatchResult ReadDetailed(...)` — same but returns `(M2ModelDocument, M2Era1121EraTag)`. **We want this** so we can record the source era in the manifest.

**Auto-detection logic** (line 54–82): reads the first 4 bytes as a magic. Three supported eras:
- `MDLX` (0x4D444C58) → chunked (post-11.x) → `M2ChunkedModelReader`
- `MD20` (0x4D443230) v100 or v101 → era 11.2.1 → `M2Era1121ModelReader`
- `MD20` v108+ → modern (3.x WotLK+) → `M2ModelReader`
- Anything else → `InvalidDataException` (bad magic) or `NotSupportedException` (TBC 2.x, tracked in spec 049).

**Error type**: `InvalidDataException` for bad magic; `NotSupportedException` for unsupported era. Our `M2PoseSourceLoader` should catch both and re-throw with a clear message.

**Implication for the plan**: T014 should use `M2ModelReaderDispatcher.ReadDetailed` and store the era in the `SourceFormat` field. The plan's claim of "AFM2 magic for chunked" is wrong — it's `MDLX`, not `AFM2` (which is a separate container for anim files, see R-0.4).

**Implication for the spec**: The spec's FR-002 says "via the existing `M2ModelReaderDispatcher`" — confirmed. The plan's P1-5 must be revised to use the dispatcher.

---

## R-0.2 — MDX reader API contract

**Source**: `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxFile.cs:122` and `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxHeaders.cs:10`

**API contract**:
- `static MdxFile Load(string path)` — opens the file.
- `static MdxFile Load(Stream stream)` — accepts a stream.
- `static MdxFile Load(BinaryReader br)` — lowest-level form.

**Magic** (`MdxHeaders.cs:10`): `0x584C444D` (which is "MDLX" in ASCII little-endian). Interesting — this is the same magic the chunked M2 reader uses (`MdxMagic.Mdlx`). The MDX format and the chunked M2 format share a top-level magic but diverge in body structure (MDX uses the WoW 0.5.3 / 1.x era chunk layout; chunked M2 uses the post-11.x chunk layout). The dispatcher's `M2Era1121EraTag.Mdlx` handles chunked M2; `MdxFile.Load` handles classic MDX. The magic is identical because they descend from the same lineage.

**Populated fields** (after a successful `Load`):
- `mdx.Version` (uint)
- `mdx.Model` (`MdlModel`, from MODL chunk)
- `mdx.Sequences` (`List<MdlSequence>`, from SEQS chunk)
- `mdx.GlobalSequences` (List<CiRange>, from GLBS chunk)
- `mdx.Materials`, `mdx.Textures`, `mdx.Geosets`, `mdx.GeosetAnimations`, `mdx.ParticleEmitters2`, `mdx.RibbonEmitters`
- `mdx.PivotPoints` (List<C3Vector>, from PIVT chunk)
- `mdx.Bones` (`List<MdlBone>`, from BONE + HELP chunks, with deferred pivot assignment at line 260)

**Per-bone structure** (`wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxModels.cs:122`):
- `MdlBone.Name` (string)
- `MdlBone.ObjectId` (int — pivot point index)
- `MdlBone.ParentId` (int — parent bone index, -1 if root)
- `MdlBone.Pivot` (C3Vector, set after PIVT chunk is read)
- `MdlBone.TranslationTrack`, `.RotationTrack`, `.ScalingTrack` (`MdlAnimTrack<T>?`, may be null)

**Per-track structure** (`wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxModels.cs:94`):
- `MdlAnimTrack<T>.InterpolationType` (MdlTrackType.Linear | Hermite | Bezier)
- `MdlAnimTrack<T>.GlobalSeqId` (int, -1 if not global)
- `MdlAnimTrack<T>.Keys` (List<MdlTrackKey<T>>)

**Per-key structure** (`wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxModels.cs:113`):
- `MdlTrackKey<T>.Frame` (int — frame index, NOT milliseconds)
- `MdlTrackKey<T>.Value` (T)
- `MdlTrackKey<T>.InTan`, `MdlTrackKey<T>.OutTan` (T — Bezier/Hermite tangents, ignored in v1)

**Frame time conversion**: MDX keyframes are in frames, not ms. The standard conversion is `ms = frame * (1000 / framerate)` where framerate is typically 30 fps for MDX (some files may differ; v1 assumes 30 fps and documents this in `quickstart.md`).

**Error type**: `InvalidDataException` for bad magic (`MdxFile.cs:149`) or chunk overruns. The catch block at line 247 wraps all other exceptions with a chunk trail for debugging.

**Implication for the plan**: T015 (`MdxPoseSourceLoader`) and T025 (`MdxBoneTrackStreamExtractor`) can be implemented as planned. Frame→ms conversion is added to the extractor.

---

## R-0.3 — Listfile cache API contract

**Source**: `wow-viewer/src/core/WowViewer.Core.IO/Files/ArchiveListfileCache.cs:31` and `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs:1303, 2978`

**API contract**:
- `static ArchiveListfileCacheManifest? TryRead(string cacheDirectoryPath, string cacheKey)` returns the manifest or null. Returns null if the file doesn't exist OR if `FormatVersion != 1`.
- `static string Write(string cacheDirectoryPath, string cacheKey, ...)` writes a new manifest.

**Cache file format** (line 92): `<cacheDir>/<sanitized-cache-key>.json` where invalid filename chars in the key are replaced with `_`.

**Cache directory default** (`Program.cs:2984`): `<repo-root>/output/cache/archive-listfiles/`. The tool walks up from `AppContext.BaseDirectory` to find `WowViewer.slnx` (max 8 levels) and uses that as the repo root.

**Cache key format** (`Program.cs:1317-1322`): a free-form user-supplied string. Convention is the client build identifier (e.g. `3.3.5a`, `1.12.1`, `wotlk`). The key is required; there's no auto-derived default.

**Manifest structure** (`ArchiveListfileCache.cs:5`):
```csharp
public sealed record ArchiveListfileCacheManifest(
    int FormatVersion,         // must equal 1
    string CacheKey,          // echo of the key
    string[] ArchiveRoots,    // the archive roots it was built from
    DateTimeOffset GeneratedAtUtc,
    string[] TrustedInternalEntries,   // from MPQ internal listfiles
    string[] SupplementalEntries);     // from user-supplied listfile.txt
{
    public IReadOnlyList<string> AllEntries { get; } = BuildAllEntries(...);
}
```

`AllEntries` is the deduped, sorted, case-insensitive union of internal + supplemental. Entries use backslash separators (Windows-style, normalized in `NormalizeEntries` line 100).

**Build command** (`Program.cs:1303`): `WowViewer.Tool.Inspect archive build-listfile-cache --archive-root <dir> --cache-key <key> [--listfile <txt>] [--cache-dir <dir>]`. We must instruct users to run this before `batch` in `quickstart.md`.

**Implication for the plan**: T075 (`BatchEnumerator`) reads the manifest, filters to `.m2`/`.mdx` (case-insensitive), and applies regex filters. Cache key/dir default to the inspect tool's defaults so users get a smooth experience. The `quickstart.md` (T057) must document the build-listfile-cache prerequisite.

---

## R-0.4 — External animation resolution

**Source**: `wow-viewer/src/core/WowViewer.Core.Runtime/M2/M2ExternalAnimationRuntime.cs:1-111` and `wow-viewer/src/core/WowViewer.Core/M2/M2ModelIdentity.cs:37`

**API contract** (line 27–78):
- `static M2ExternalAnimationRuntimeState Choose(M2ModelDocument model, int sequenceIndex)` — resolves the alias chain, returns the terminal sequence's state. Throws `ArgumentOutOfRangeException` for bad index, `InvalidDataException` for alias loops.
- `static M2ExternalAnimationRuntimeState Load(M2ExternalAnimationRuntimeState state, M2ExternalAnimationDocument animation)` — binds a pre-loaded `.anim` document. Throws if the animation's `SourcePath` doesn't match `state.CompanionPath` (case-insensitive, normalized via `M2ModelIdentity.PathsEqual`).

**Companion path construction** (`M2ModelIdentity.cs:37`):
```csharp
string extension = Path.GetExtension(CanonicalModelPath);  // ".m2"
string basePath = CanonicalModelPath[..^extension.Length];
return $"{basePath}{animationId:D4}-{variationIndex:D2}.anim";
// e.g. "Creature/Orc/OrcMale0000-00.anim" for animation 0, variation 0
```

**External vs inline** (`M2SequenceDefinition.cs:95`): `UsesExternalAnimationFile => (Flags & 0x130) == 0`. Sequences with this flag are backed by `.anim` files; sequences with `0x20` (`StoredInline`) embed their data in the M2.

**Search path**: the `M2ExternalAnimationRuntime.Choose` returns `CompanionPath` as a virtual path. The caller is responsible for *loading* the bytes. Our `M2ExternalAnimationFilesystemLoader` (T050) will:
1. Try `File.ReadAllBytes(companionPath)` directly.
2. Try reading from each `--archive-root` via `ArchiveVirtualFileReader.ReadVirtualFile`.
3. Return `null` if all paths fail (the resolver records `external-unresolved`).

**Implication for the plan**: T021 (`M2SequenceAliasResolver`) uses `M2ExternalAnimationRuntime.Choose` for chain resolution. T050 implements the loader. The flow is: `Choose` → get `CompanionPath` → loader returns bytes → wrap in `M2ExternalAnimationDocument` → call `Load` to bind. The bound state tells the caller whether loading succeeded.

---

## R-0.5 — JSON Schema strategy

**Decision**: ship `contracts/*.schema.json` (JSON Schema Draft 2020-12) + cross-reference from `data-model.md` and from the spec.

**Rationale**: Python validators (`jsonschema` package) and TypeScript consumers can use the schemas directly. Round-trip tests in C# are complementary but don't help external consumers. Cost of writing a schema is low (one file per artifact) and the benefit is high (downstream tooling).

**Schemas to ship**:
- `contracts/manifest.schema.json` (per-model manifest, FR-008)
- `contracts/poseclip.schema.json` (per-sequence pose clip, FR-019)
- `contracts/library-index.schema.json` (batch top-level index, FR-021)

**Implication for the plan**: T040, T066, T079 each write one schema file. The schemas are referenced from `data-model.md` (TBD in P1) and from the architecture doc (P9).

---

## R-0.6 — BVH format decision

**Decision**: Biovision Hierarchical 1.0 (the most common BVH dialect used by Blender, bvh-python, Cascadeur, and most mocap tools).

**Channel conventions**:
- 3 position channels per joint, in order **Xposition Yposition Zposition** (FR-007).
- 3 rotation channels per joint, in order **Zrotation Xrotation Yrotation** (this is the Blender/bvh-python convention; rotations are *intrinsic* Z-X-Y Euler).
- Scale is omitted (per OQ-2).
- Channel declaration order on the `CHANNELS` line is "position first, then rotation" matching the order values appear in the frame data.

**Frame time**: BVH's `Frame Time:` is in seconds, not ms. We convert `timeMs / 1000.0`. Use `en-US` (`.` decimal) — this is BVH convention; some tools choke on locale-specific separators.

**Line endings**: `\n` (LF only). BVH consumers accept either LF or CRLF; LF is the safer choice and matches the rest of the C# code's text-file convention.

**Coordinate system**: BVH uses right-handed Y-up. WoW M2 uses left-handed Y-up. We do not transform coordinates in v1 — consumers that need to convert can do it downstream. We document this in `quickstart.md`.

**Implication for the plan**: T031 (`BvhDocumentBuilder`) emits channels in the declared order. T032 (`BvhDocumentWriter`) formats the file. T033 (`BvhDocumentReader`) is the parser for round-trip tests only; it does not need to be a full BVH parser — just enough to confirm write→read equality.

---

## R-0.7 — FBX 7.4 ASCII writer scope

**Decision**: minimum-viable FBX 7.4 ASCII writer that produces a file Blender's FBX importer can open.

**Sections to emit**:
1. **`FBXHeaderExtension`**: `FBXHeaderVersion: 1003`, `FBXVersion: 7400`, `CreationTimeStamp`, `Creator: "WowViewer.Tool.AnimFarm 1.0"`.
2. **`Objects`**:
   - `GlobalSettings`
   - One `Model` node per joint (named after the joint, parented via `Connections`)
   - `Model::RootNode` (skeleton root, parent of all joint models)
   - `Pose::BindPose` containing `PoseNode` per joint with `NodeAttribute::LclTranslation/Rotation/Scaling` at the bind pose
   - One `AnimationStack` per non-alias sequence
   - `AnimationCurveNode` per (joint, channel) per sequence — channels are `T` (translation), `R` (rotation), `S` (scaling)
   - `AnimationCurve` per (joint, channel) per sequence — contains the actual `KeyTime` + `KeyValueFloat` arrays
3. **`Connections`**: a list of `OO` records linking the above nodes by numeric UID.

**What's NOT in v1**:
- No materials
- No textures
- No blend shapes
- No skeleton root metadata beyond the bind pose
- No lighting
- No cameras (those are tracked separately and out of scope per the spec's "Explicitly Out of Scope" section)

**Time units**: FBX uses KTime, which is a 64-bit integer where 1 unit = 1/46186158000 of a second. The default 30fps baseline means `frame * 46186158000 / 30 = frame * 1539538600` is one KTime value. For ms→KTime: `tMs * 46186.158` (rounded to integer).

**Implication for the plan**: T089–T091 are feasible. The writer is ~600 LOC and the parser for round-trip is ~200 LOC (just count nodes and check sections). A real-data test against a known M2 is required.

---

## R-0.8 — research.md delivered

This file. T008 ✅.

---

## Spec deltas discovered during research

1. **`M2ModelReaderDispatcher` exists** (R-0.1). The plan's T014 should use it, not `M2ModelReader.Read` directly. No spec change — the spec already names the dispatcher.
2. **Chunked M2 magic is `MDLX`, not `AFM2`** (R-0.1). `AFM2` is the anim container magic only. No spec change — `SourceFormat` is just a label.
3. **MDX frames are frame indices, not ms** (R-0.2). The `MdxBoneTrackStreamExtractor` must convert `frame * (1000 / fps)` assuming 30 fps for v1. The pose clip's `tMs` is therefore a derived value. Document in `quickstart.md`.
4. **External anim loading is a caller responsibility** (R-0.4). `M2ExternalAnimationRuntime.Load` requires a pre-loaded document; we wrap that in a filesystem-aware loader. No spec change.

These are minor clarifications, not spec bugs. The plan's T014 is the only one with a concrete code-path change; the rest of the plan is sound.

---

## Open architectural questions (deferred to implementation)

These came up during research but are not blocking. Track them as follow-up tasks if they become problems during implementation.

- **A. Coordinate system conversion**: BVH is Y-up, M2 is also Y-up but left-handed. Should we swap Z on output? Default: **no** (keep raw values, document in `quickstart.md`).
- **B. FPS assumption for MDX**: 30 fps is the standard but not guaranteed. We could parse the MODL chunk for an FPS hint. Default: **30 fps**, document the assumption.
- **C. Global sequence wrap behavior**: M2 global sequences loop; BVH doesn't have a "loop" concept. We emit the raw timestamps as-is; consumers handle looping. Default: **no wrap in v1**.
- **D. Hidden/invisible bones**: M2 has flags that hide bones at certain times. The pose clip's "identity at root" treatment for unmatched slots might want to respect visibility. Default: **ignore visibility in v1**; bone is always at its TRS.
