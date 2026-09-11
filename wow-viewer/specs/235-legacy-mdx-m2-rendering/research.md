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

### 6. The Era100 offsets are unverified, and their tests cannot verify them

`M2Era100ModelReaderTests.cs` states in its own header comment: *"built on synthetic fixtures, so
they run without a staged client."* The fixtures are constructed to the assumed offsets and then
asserted back — self-consistent and incapable of failing on a wrong offset. This is the same shape as
the BSDIFF and MCNR-axis defects already recorded in this project's history.

Against Warcraft.NET's generic field order, Spec 104's Ghidra-derived 1.0.0 offsets are **+8 bytes
shifted, consistently** (bones `0x34` vs `0x2C`, vertices `0x44` vs `0x3C`, divisions `0x4C` vs
`ViewCount` `0x44`). Either 1.0.0 genuinely carries 8 extra header bytes, or the trace mis-attributed
a field. **Unresolved, and it needs one real file to settle — not more byte arithmetic.**

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

## Not started

Items 2, 3, 4, 6, 7, 8 of plan.md Phase 0 — build enumeration, per-build declared versions, the three
`3.0.1` builds surveyed separately, `0x102`–`0x106` coverage, Warcraft.NET's real supported range,
a concrete fuckported asset, and whether the torch/light gap is parse- or wiring-side.

## Blocked — and how to unblock it

**Real-file verification did not happen.** `m2 inspect --archive-root <client> --virtual-path <path>`
returned `FileNotFoundException` for every path tried against
`H:\CLIENTS\Vanilla\1.x\1.X_Retail_Windows_enUS_1.0.0.3980\World of Warcraft`, including
`World\ArtTest\BoxTest\XYZ.mdx` — the operator-named test object that exists in every build, and which
is present in the repo's own listfile (`libs/wowdev/wow-listfile/listfile.txt:808150`). That client's
`Data/` holds the older content-segmented archives (`base.MPQ`, `model.MPQ`, `dbc.MPQ`, `terrain.MPQ`,
`wmo.MPQ`, …), not numbered patch archives.

**This is an invocation error, not a missing capability.** The operator's correction stands: this repo
already has tooling that inspects everything. **Next session: find the documented/working invocation
before improvising** — check `tools/inspect`'s own usage output, the quickstarts in specs that ran real
client reads (104, 154, 205's `inspect adt liquid-formats --client <dir>`, `inspect dbc dump --client
<dir>`), and note that several commands take `--client`/`--game-path`, not `--archive-root`. Do not
guess flags or asset paths again.

Also: `wowdev.wiki` returns HTTP 403 to WebFetch (both `/M2` and `?action=raw`). Use a browser or a
local copy.

## The reframe this points to (for plan.md, not yet applied)

One generic version-tolerant reader — Warcraft.NET's, already wrapped — plus:

1. the embedded skin/view walk it is missing (versions ≤ 263, no external `.skin`),
2. deletion of the `0x102`–`0x107` refusal wall,
3. an `Md20_1X_V100_Era100` branch (or its removal in favour of the generic path) in
   `WorldAssetManager`,
4. a real bounding-box fallback so a failed load is visible rather than silent,

with per-era bespoke readers kept **only** where a real file proves the generic path cannot handle it.
That is a materially smaller and better-evidenced plan than the six-phase per-era structure currently
in `plan.md`, which was written before findings 1–6 existed.
