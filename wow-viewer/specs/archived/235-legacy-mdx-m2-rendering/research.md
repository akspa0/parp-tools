# Phase 0 Research: Legacy MDX/M2 (1.0.0–3.0.1) Reconciliation

**Status**: Partially answered 2026-09-10. The architecture questions (FR-014 and the routing gap)
are **answered with code evidence**. The real-data survey items are **not started** — see "Blocked"
at the end.

## Answered — with file/line evidence

### 1. FR-014: `FormatProfileRegistry` vs `M2ModelReaderDispatcher` — both exist, different jobs

**`FormatProfileRegistry.ResolveMdxProfile` is dead code.** Caller search across `src/` returns
zero callers outside its own definition. The MDX-side per-build profile table (`MdxProfile_070_3694`,
`MdxProfile_090x_Unknown`, …) is never consulted by anything. **This is why "MDX is fine"** — real
MDX dispatch happens on magic bytes in `M2ModelReaderDispatcher.DetectEra` (`MDLX` →
`M2ChunkedModelReader`), which never asks the registry. The `major >= 1` fallback bug in that method
is real but inert.

**`FormatProfileRegistry.ResolveModelProfile` (M2) is live and load-bearing.** Callers in
`AssetProbe.cs`, `ViewerApp.cs`, `ModelRenderer.cs`, `WorldAssetManager.cs`, `WmoRenderer.cs`,
`WarcraftNetM2Adapter.cs`. Its `major == 1` → `null` return **disables two real routes for the whole
Vanilla retail era**:

- `WarcraftNetM2Adapter.SupportsEmbeddedNativeRoute` → `IsEmbeddedProfileRoute` only accepts
  `M2Profile20xUnknown` / `M2Profile30xUnknown` / `M2Profile3018303` (major 2, major 3.0, and the
  3.0.1 pre-release). A `major == 1` build resolves to `null` and is refused before it starts
  (`WorldAssetManager.cs:1340`).
- The adapter-embedded-profile fallback right after it compares against `M2Profile3018303.ProfileId`
  (`WorldAssetManager.cs:1375`); `null?.ProfileId` fails that too.

**Order of authority**: `M2ModelReaderDispatcher.DetectEra` (byte-level, build-agnostic) decides era
first; `FormatProfileRegistry` is consulted only for what `DetectEra` does not claim. They are not
duplicates — but the registry's per-literal-build-string table is the fragmentation worth deleting,
and most of it is provably unused.

### 2. The world-placement loader never reaches the 1.0.0 reader

`WorldAssetManager`'s "no external `.skin` found" chain (`~1338–1470`) tries, in order:
`SupportsEmbeddedNativeRoute` (excludes major 1), the `M2Profile3018303` check (excludes major 1),
then `DetectEra` — but its `if` matches only `Md20_1X_V100` / `Md20_1X_V101` (the 1.12.1-shaped
layout). **There is no branch for `Md20_1X_V100_Era100`.** A genuine 1.0.0 classic-layout file
matches nothing, falls through to a blind `ConvertM2ToMdx` attempt, and fails.

Consequence: **fixing `M2Era100ModelReader`'s bone parser (Spec 154's D1/D2) would not by itself make
world objects render** — that reader is never invoked from this path. This was not documented by
Spec 104 or 154; both looked at readers, not at this caller.

### 3. "No bounding boxes either" — there is no bounding-box fallback

Every failure path in that chain ends at `return null` (`WorldAssetManager.cs:1470`, and the outer
catch at `:1483`). Nothing draws a placeholder. The symptom is exact: total load failure means the
object is invisible, not boxed. FR-005's bounding-box fallback does not exist and must be built, not
merely reconnected.

### 4. The `0x102`–`0x107` wall is deliberate, and its citation is stale

`M2ModelReaderDispatcher.DetectEra` throws `NotSupportedException("MD20 v0x{version:X} is the 2.x
TBC era, which is not yet supported. Tracked under spec 049.")` for everything between `0x101` and
`0x108`. This confirms Spec 154's measured `0x100`–`0x107` broken range as **hardcoded logic**, not
just an observation. **"Spec 049" is wrong** — `specs/archived/049-viewer-ui-consolidation` is an
unrelated UI spec. Fix the citation when the wall comes down.

### 5. Warcraft.NET already does this generically — and is already wrapped

`WarcraftNetM2Adapter` wraps Warcraft.NET's `MD21` / `Model` / `Skin` types directly. Warcraft.NET's
`MD21.LoadBinaryData` reads the **entire header in one fixed sequential layout** with **zero
per-version branching** — one conditional, keyed on a *flag bit* (`UseTextureCombinerCombos`), not a
version number. Its own annotation is
`[AutoDocChunk(0, VersionBeforeLegion, VersionAfterWoD)]` — i.e. this single reader is declared valid
across the whole era in scope here.

**It captures `ViewCount` and never walks the embedded skin/view table.** That is the one real gap,
and it is exactly what Spec 104 diagnosed independently ("reads `viewCount` … but passes
`embeddedSkinProfileOffset: 0`"). Filling that one gap in the generic path is the small fix; the
per-era reader classes are the large detour.

### 6. Finding 6 RESOLVED (2026-09-11): Real-file inspection proves unified layout across 0x100–0x107

Real client archives from `H:\CLIENTS` were inspected directly across five representative client generations:
- 1.0.0.3980 retail (`World\ArtTest\Boxtest\xyz.m2`) — version 0x100 (256)
- 2.0.0.5610 pre-release (`CHARACTER\BloodElf\Male\BloodElfMale.m2`) — version 0x100 (256)
- 2.4.3.8606 retail (`CHARACTER\BloodElf\Male\BloodElfMale.m2`) — version 0x107 (263)
- 3.0.1.8303 pre-release (`CHARACTER\BloodElf\Male\BloodElfMale.m2`) — version 0x107 (263)
- 3.3.0.10958 retail (`CHARACTER\BloodElf\Male\BloodElfMale.m2`) — version 0x108 (264)

**Findings (Binary Offset Evidence):**
Across the **entire** 1.0.0 through 3.0.1 range (`0x100` through `0x107`, versions 256 through 263), the header layout is **identical**:
- `0x14`: Global loops (count, offset)
- `0x1C`: Sequences (count, offset)
- `0x24`: Sequence/Animation lookup (count, offset)
- `0x2C`: Playable animation lookup / secondary animation lookup (count: 201 in 1.0.0, 203 in TBC/Wrath)
- `0x34`: **Bones** (count, offset)
- `0x3C`: **KeyBoneLookup** (count, offset)
- `0x44`: **Vertices** (count, offset)
- `0x4C`: **Views / Divisions** (count, offset) — contains **embedded** skin profiles (`M2Division`), NOT external `.skin` files!

At **3.3.0 (`0x108`, version 264)**:
The secondary lookup at `0x2C` was removed, shifting subsequent fields back by 8 bytes:
- `0x2C`: Bones
- `0x34`: KeyBoneLookup
- `0x3C`: Vertices
- `0x44`: ViewCount (referencing external `.skin` files)

**Why all current readers failed on legacy files:**
- Warcraft.NET and `M2ModelReader` assume the `0x108` layout where `0x2C` is Bones. On legacy models, they read `0x2C` (count 201–203) as bone count, multiply by 88-byte stride, and throw an out-of-range exception beyond EOF.
- `M2Era100Constants.cs` correctly recognized `Bones` at `0x34`, `Vertices` at `0x44`, and `Divisions` at `0x4C`, but its bone parser was incomplete / zeroed out.
- `M2ModelReaderDispatcher.DetectEra` threw `NotSupportedException` on `0x102`–`0x107` citing "spec 049".
- `WorldAssetManager` had no branch for `Md20_1X_V100_Era100`, routing to `ConvertM2ToMdx` and failing.
- `WorldAssetManager` lacked a bounding-box fallback, returning `null` on failure and making models invisible.
- `M2SkinProfileRuntime` attempted to fetch external `.skin` files, which do not exist for version <= 263.

## Operator direction (2026-09-10, verbatim intent)

- "We should not need per-build specific changes… only the damn version number changes." Reference
  Noggit/noggit-red and the wowdev.wiki MDX/M2 pages rather than deriving per-build behaviour.
- "MDX is fine, M2 is not fully there." — confirmed by finding 1 above.
- "1.0.0+ uses .MDX as the extension, but M2 as the format, with the MD20 chunk." Verified the
  routing already honours this: `ViewerApp.LoadModelFromBytesWithContainerProbe` and
  `WorldAssetManager`'s path resolution both dispatch on magic bytes; extension appears only in
  candidate-path generation and in a diagnostic mismatch log. **Not a bug source.**
- "0.12.0, 1.0.0 use the same .mdx file format but M2 format files." Recorded; not yet verified
  against 0.12.0 specifically.
- "We literally wrote an adapter around Warcraft.NET's M2 code… just refer to the wowdev wiki's M2
  page and Warcraft.NET's M2 implementation." — this is the direction the plan should take.

## Unblocked (2026-09-11)

The invocation issue was resolved: the model in 1.0.0.3980 is named `World\ArtTest\Boxtest\xyz.m2`, not `xyz.mdx`.
Real client inspections succeeded, establishing the single header difference between legacy (<= 263) and modern (>= 264).

## The Reframe for plan.md

One version-tolerant legacy reader supporting versions 256–263 (`0x100`–`0x107`):
1. Reads `Bones` at `0x34`, `KeyBoneLookup` at `0x3C`, `Vertices` at `0x44`, and `Views` at `0x4C`.
2. Reads the embedded skin/division records from the view table (no external `.skin` files).
3. Deletes the `0x102`–`0x107` refusal wall in `M2ModelReaderDispatcher`.
4. Connects legacy embedded models to `WorldAssetManager`.
5. Adds a bounding-box fallback so failed loads draw a bounding box rather than remaining invisible.

