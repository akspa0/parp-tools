# Active Context — wow-viewer

Last updated: 2026-07-15 (Spec 108 RGB-only WDL prior built; user-run compact-corpus training next)

## Spec 108 — RGB-only WDL prior (2026-07-15, implemented/tested; no training launched)

- Added a small independent RGB→WDL model with the exact paired target: outer
  `height_257[::16,::16]` (17×17) plus inner `height_257[8::16,8::16]` (16×16), never a 33×33
  substitute. It serializes the input/target contract and writes row-addressed generated-prior
  archives.
- `infer_spec103_v7.py --generated-wdl-priors` now consumes that generated outer lattice for V8
  ch6 and derives its height hints from it; missing/misaligned rows fail closed rather than fall
  back to ground truth. This only closes the WDL handoff: V8's remaining auxiliary-channel
  predictors remain later independent residual slices.
- Train directly from the real V18 paired store **through the existing compact representative-pattern
  curation manifest**; the trainer reads only selected rows and honors manifest partitions. Do not
  train on the full V18 corpus. CPU contracts: 4/4 passed. User owns the CUDA train/infer run;
  `specs/108-image-wdl-prior/quickstart.md` has the commands.
- **Required first quality proof:** `evaluate_spec103_wdl_prior.py` runs inference from RGB alone,
  then opens the real tile's height only to report outer/inner lattice world-unit error and save both
  lattices. Its exported `input_minimap.png` can immediately prove standalone PNG→lattice inference.
- The initial tiny800 run plateaued at normalized validation L1 `0.037808`; trainer now has default
  `--patience 10`, writes `best_epoch`/stale count, and stops rather than wasting later epochs.
- Correction after a black first evaluation tile: tiny800 includes dark water/placeholders and object
  pages. The trainer now excludes pathological rows, RGB mean below 25, and any object coverage by
  default. On real 3.3.5 tiny800 this leaves 62 train / 10 held-out clean ChamberOfAspectsBlack rows;
  row 906 (`ChamberOfAspectsBlack_29_27`) is the first bright, valid real-tile evaluation target.
- **Corrected corpus owner:** the old 10-tile/30-row synthetic store was only a smoke fixture. The
  default generator now authors 320 varied fields (10 families × 2 amplitudes × 16 independently
  seeded parameterizations); three authored lighting times yield 960 RGB→WDL rows. Train on this
  controlled corpus first, hold out its full crater family (96 rows), then measure the explicit
  synthetic-to-real gap. Do not replace it with the tiny real-data difficulty sample.

## Spec 107 — Lighting quick controls + confident hover (2026-07-15, implemented/build green)

- **Fog defect fixed:** LIT/DBC fog decoding was already correct; `GetSceneFarPlane` silently forced a
  6000-unit minimum, defeating short valid FogEnd values and exposing distant terrain through fog.
  It now uses `FogEnd + 1024` padding with only a projection-valid 1-unit minimum.
- **Quick controls:** Tools > Quick now exposes time, FogStart/FogEnd, LIT fog toggle, active range,
  computed clip distance, and a one-action link to Utilities > Lighting (the full LIT evidence owner).
- **Hover contract:** an exact-path card now requires a nearest ray hit. Brush/overlap candidates
  remain usable for click inspection but no longer claim a precise tooltip. Isolated Debug build to
  `output/tmp/spec107-build` passed with 0 errors; normal output was locked by the user's running viewer.

## Spec 106 — Native day/night lighting fidelity (2026-07-15, planned; no implementation/capture run)

- **Exact 0.5.3.3368 live/PDB proof:** `DayNightUpdateLighting` (`0x006bd6c0`) calls separate
  color and direction paths. `SetDirection` (`0x006bca40`) computes the native downward ray as
  `(sin(phi)cos(theta), sin(phi)sin(theta), cos(phi))`; all theta samples are 225 degrees and phi
  varies 110–127 degrees. Thus source azimuth is fixed 45 degrees but elevation varies about
  20–37 degrees. The old `cos(pi/4)` finding is **unit/dynamic shadow projection only**, never the
  world sun. Recorded in `docs/architecture/wow-1.0.0-world-lighting-shadow-model-2026-07-15.md` §2.1.
- **Current gap:** LIT/DBC colors are already exact-build data, but `LitTerrainDayNightProfile` and
  interactive `TerrainLighting` still inherit/compute authored direction. The one remaining empirical
  proof is a *single fixed native-vector → viewer-terrain coordinate/sign transform*, validated at one
  lock and two held-out times; it is not an azimuth search. USER owns the native/capture comparison.
- **Plan:** `specs/106-native-daynight-lighting/` contains spec, research, data model, contract,
  quickstart, and phased library-first plan. It links (not replaces) Spec 032 renderer parity and
  Spec 103 T040 synthetic provenance. Do not claim client-exact capture until a calibrated transform
  plus coherent LIT-or-DBC source is recorded; MCSH attenuation and sky-band altitude stay separate.

## Spec 105 — SCOPE EXPANDED to combined 1.0.0 M2 render (2026-07-15, spec+plan, NOT implemented)

- **USER showed a live screenshot: CentaurKhan.m2 (v256) renders GRAY, FLAT, STATIC** (frame
  2081/3333 advancing). USER: "still flat shaded and only a single material and not animated."
  Chose **"one combined 1.0.0 M2 render spec"** → spec 105 expanded from animation-only to **THREE
  rendering pillars**. Folder still named `105-format-version-profiles` (cosmetic; scope is now
  "1.0.0 M2 renders correctly"). Title updated.
- **THE THREE PILLARS + verified causes (sequenced texture→lighting→animation):**
  - **P-Texture (gray/single-material)** — the reader parses filenames fine
    (`M2Era100ModelReader.cs:302`) AND `M2Renderer` has a full load+replaceable path (`:1023`), so the
    gray is a **resolution failure** (batch→combo→texture), NOT a parse/support gap. Exact cause
    (combo mis-index vs BLP-not-found vs replaceable-with-no-runtime-skin) **NOT yet pinned** — Phase T
    opens with a **diagnosis-only pass** (FR-T1). Prime suspect: a creature texture whose combo
    resolves NEGATIVE (replaceable) when it should be a hardcoded BLP. Fix in `BuildEra100Material` /
    `WowViewerM2RuntimeBridge.cs:220`, era-100-scoped.
  - **P-Lighting ("flat shaded")** — Lambert ALREADY RUNS (`M2Renderer.cs:854`, gated on
    `Material.IsUnshaded` @`:697`); screenshot shows a visible N·L gradient. The "flat" look is
    **mostly the missing texture**. This pillar is CONFIRM-then-correct: audit material flag `0x01`
    (UNLIT) → `IsUnshaded` mapping; may reduce to "already correct, close it."
  - **P-Animation (frozen)** — CONFIRMED: **both** readers drop bones (`M2Era100ModelReader.cs:118`
    `bones: null`; `M2Era1121ModelReader.cs:95` `bones: []`). Rests on the version-profile foundation.
- **CORRECTION recorded (I overclaimed earlier):** "3333 = misread duration" is NOT why the pose is
  frozen. The freeze is `bones: null/[]`, full stop. The R1 sequence-offset misread is real but only
  corrupts the DISPLAYED timeline number (`M2RuntimeAnimator.cs:31` builds the timeline from
  `sequence.Duration`, which era-100 reads from field @0x04 = actually `start`).
- **Execution order (plan):** Phase 0 baseline (now ALSO captures 3.x/4.x TEXTURE/material state, since
  Phase T/L touch the shared `M2Renderer`) → **Phase T texture** → **Phase L lighting** → Phase 1–4
  (architecture + animation, unchanged). Headline success **SC-000**: CentaurKhan textured + lit +
  animated, user-confirmed.

## Spec 105 — foundation: M2 version profiles (2026-07-15, SPEC + PLAN done, NOT implemented)

- **`specs/105-format-version-profiles/`**: spec, checklist, **plan, research, data-model,
  contracts/m2-era-profile.md, quickstart**. `tasks.md` NOT yet generated (next: `speckit-tasks`).
  Written on `v0.5.1` — no feature branch, artifacts manual (`.specify` scripts are PowerShell-only
  and resolve the git root; known quirk).
- **🔴 LIVE BUG FOUND DURING PLANNING (research R1) — the M2Sequence record is MISREAD.**
  Ghidra `FUN_0070f960`: 1.0.0 `M2Sequence` (stride 0x44) = **{id u16@0, variationIndex u16@2,
  START u32@4, END u32@8, movespeed f32@0xC, flags u32@0x10 (bit0 clear = looping)}**.
  `M2Era100ModelReader.ReadSequences` reads **Wrath offsets** (`duration@4, moveSpeed@8, flags@0xC,
  frequency@0x10`) — **every field from +0x04 is shifted 4 bytes**. The stride was fixed per era; the
  **field offsets inside were copied from Wrath and never revisited** (the constant's own comment
  "0x44 — same as 1.12.1 V100" records that the size was checked and the interior wasn't).
  **This explains "frame 2030/3333, static": `Duration` is actually reading `start`** — a
  global-timeline offset displayed as a duration. It was never a duration.
  **Corroborated by wowdev.wiki**: `M2Sequence` documents `startTimestamp`/`endTimestamp` for **≤BC**,
  merged into a single `duration` from **Wrath**. Ghidra and the public docs agree.
  **⚠️ `M2Era1121ModelReader` HAS THE SAME SHAPE** (stride 0x6C, but `moveSpeed@0x08` = Wrath
  offsets) → **1.12.1 may be animating wrongly TODAY and nobody has reported it.** Marked
  PROVISIONAL, deliberately **NOT fixed by analogy** (1.0.0 vs 1.12.1 already proved they diverge
  despite sharing version 0x100 — analogy is the exact reasoning that caused this bug). Needs a
  1.12.1 client in Ghidra to settle. 3.x `M2ModelReader` (stride 0x40) is **correct** (true Wrath).
- **KEY DESIGN DECISION (research R2) — normalize sequences to `{Start, Duration}`:** Wrath sets
  `Start = 0`, `Duration = duration@4`; 1.0.0 sets `Start = start@4`, `Duration = end@8 - start@4`.
  Sampler becomes ONE rule for all eras: `sampleTime = Start + (elapsed mod Duration)` — which
  **reduces to today's exact `elapsed % duration` when Start = 0**. So the **3.x/4.x no-regression
  guarantee is STRUCTURAL, not empirical**. This is the spine of the whole plan.
- **Phasing (each ends with a gate, not code):** P0 pin the 3.x/4.x baseline **BEFORE touching any
  shared type** (ordering is load-bearing — a later baseline measures the new code against itself);
  P1 profile contract in core + delete inert `M2Profile` (net ≈ **−50 lines**, SC-004 shrink bar
  verified feasible in R6); P2 era-aware track addressing + time base + the R1 offset fix; P3
  populate era-100 bones (blocked on P2 — FR-007); P4 deterministic era resolution, kill the
  trial-parse.
- **Constitution Check: PASS**, no violations, Complexity Tracking empty. Core→viewer tension for
  the build hint resolved by **dependency inversion** (core exposes an optional build hint; the
  viewer supplies its existing `_dbcBuild`) — R4.
- **OPEN, needs USER:** (1) FR-017 reading — deleting the inert `M2Profile` leaves a dangling
  validation call in `WarcraftNetM2Adapter`; remove it (cannot change behaviour — it validates
  against records that can only ever pass) or keep a local copy? (2) Is 1.12.1 animating correctly
  today? (see the era-1121 warning above). (3) The R4 header-size discriminator is **reasoned, not
  measured** — must be confirmed against real 1.0.0 + 1.12.1 models before it's relied on.
- **TWO PREMISE CORRECTIONS — both verified in code, both invalidate prior framing:**
  1. **Our M2 code is NOT Warcraft.NET wrappers.** `WowViewer.Core{,.IO,.Runtime}` (~5,000 lines of
     M2 readers) have **zero** Warcraft.NET dependency — it is not in those .csproj at all. The
     native path (`WowViewerM2RuntimeBridge` → `M2StaticRenderModel` → `M2Renderer`) is 100% ours,
     including the 3.x/4.x path that works. Warcraft.NET's M2 = Legion+ chunked MD21; its ONLY
     consumer is `WarcraftNetM2Adapter.cs` (viewer, 3,070 lines), the legacy M2→`MdxFile` fallback
     for `MdxRenderer` (it feeds raw MD20 into `new MD21(...)`, which works only because MD21's
     payload IS an MD20 blob). **There is nothing to build Warcraft.NET adapters for.** USER's
     original ask was based on the inverted premise; corrected and accepted.
  2. **`FormatProfileRegistry` (`src/viewer/WoWViewer/Terrain/`, 717 lines) is a PRIOR ATTEMPT at
     the exact "per-version profile" architecture** — found before writing the spec, so it is the
     spec's anchor rather than being reinvented. It is scheme (A): keyed on **build string**
     ("3.3.5.12340"), viewer-side, **validation-only**, returns `null` for 1.x. Scheme (B) is
     `M2ModelReaderDispatcher`: keyed on the **file's own version field**, core-side, actually
     parses. They never talk (core can't reference the viewer — Constitution I). **That is the
     "stepping on toes."** Violates Constitution II ("one canonical owner per format surface").
     The **M2 half of the registry is INERT**: all 5 `M2Profile` records carry identical strides
     (0x70/0x2C/0xD4/0x7C) and `SkinLikeA`/`EffectLikeB` naming admits guessed semantics. The
     Adt/Wmo/Mdx halves DO carry real varying values (MclqLayerStride 0x2D4 vs 0x324) — not
     uniformly worthless.
- **USER SCOPE DECISIONS (this session):** (a) full profile architecture, not a narrow track fix;
  (b) **M2 surface ONLY** — delete the inert M2Profile records, leave Adt/Wmo/Mdx on the registry
  untouched (they sit in the Constitution's Terrain Alpha Risk Area), design so they *can* migrate
  later under their own spec; (c) `WarcraftNetM2Adapter` + MdxRenderer fallback **untouched**, tech
  debt only.
- **Ghidra `FUN_0070f6d0` (track sampler) RE-VERIFIED against WoW.exe 1.0.0.3980.** Last session
  recorded it as **prose only — no evidence file**, unlike the other ~40 traces. Re-decompiled: the
  prose is **CONFIRMED accurate on every point**, incl. the total-count clamp. Raw evidence NOW
  saved: `specs/104-legacy-m2-rendering/evidence/1.0.0-ghidra/m2_track_sampler.c`.
  **NEW:** `out[0]` is an **IN/OUT cached-key seed** driving a 3-way search (fwd scan if
  `time-ts[cache]` ∈ [0,500); bsearch if far; bwd scan if ∈ [-500,0)). It is **purely a perf cache —
  all branches converge on the same key**, so a **stateless bracketing search is equivalent**: do
  NOT port the cache, do NOT thread per-track mutable state. This makes the port simpler than the
  prior session assumed.
- **`FUN_0070f960` (the caller) TRACED — prior "largest open risk" RETIRED + a NEW gap found.**
  2,082 lines, 58 sampler call sites. Independently re-confirms bone stride **0x6C**, `parentBone`
  **@0x08**, rotation track **@0x28**, scale **@0x44** from a second site.
  - **animIndex IS the sequence index.** Per-bone state (model+0x80, stride **0x118**) holds the
    anim id at **+0xa4**; the same id indexes the M2Sequence array (`id * 0x44`). So
    `interpRanges[i]` describes `sequences[i]` — **our existing `sequenceIndex` is the right value**.
    No alias translation at this layer. Risk retired; no bone-pose guesswork needed.
  - **NEW GAP (FR-005a) — 1.0.0 sequence time is a GLOBAL TIMELINE, not sequence-local.**
    **M2Sequence (0x44) = {id u16@0, variationIndex u16@2, START u32@4, END u32@8, movespeed f32@0xC,
    flags u32@0x10 (bit0 = non-looping)}**. Client: `time = start + (elapsed % (end - start))`, handed
    to the sampler as an **absolute** time bracketed against a flat key array shared by ALL
    animations. **That is WHY interpRanges exists.** Wrath+ replaced start/end with a single
    `duration` and gave each sequence its own key array → interpRanges became unnecessary.
    **Our `M2SequenceDefinition` carries `duration` ONLY (verified) — it cannot express 1.0.0
    start/end**, and `M2TrackSampler.ResolveSampleTime` does `timeMs % duration` = the Wrath rule.
    **Implementing flat+ranges addressing ALONE still yields wrong keys.** Era-dependent *time base*
    is a first-class concern alongside era-dependent *track addressing*.
  - **Deliberately NOT modelled (noted, not flattened):** animation state is **per-bone** (each bone
    its own (time, animIndex); **-1 = inherit parent's**), and a **second** (time, animIndex) pair at
    state[0x31]/[0x32] drives **cross-animation blending** via a second sampler call. Our runtime is
    one-sequence-per-model. Out of scope for a first pass; neither blocks the P1 slice.
- **`H:\CLIENTS` IS NO LONGER FORBIDDEN — Constitution amended to v1.1.0 (2026-07-15, USER-approved).**
  USER clarification: the old ban existed because that folder once held **broken clients of unknown
  origin they didn't trust**. They have since cleaned it out. It is now a **curated temporary SSD
  staging area** they copy individual builds into, from **WoWArchive (~150 GB, cold HDD storage)** —
  the authoritative corpus. Both are legitimate sources. Principle VI now reads "never *hardcode* a
  client root" (config, not prohibition); `output/tmp/wowarchive-clients/` stays the default for
  automated validation. Static RE evidence from a staged binary is explicitly permitted.
  **Stop flagging the 1.0.0 Ghidra program's `H:/CLIENTS/...` origin as a violation — it is not one.**
  The "forbidden legacy client root" language in the Boundaries section below is **superseded**.
  **⚠️ CODE STILL ENFORCES THE OLD BAN — tracked follow-up, NOT yet done:**
  `src/core/WowViewer.Core.Anim/PathNormalizer.cs` (`StaleClientsRoot = @"H:\CLIENTS"`) **THROWS
  `InvalidOperationException`** on any path containing it, and `PathNormalizerTests` pins that
  behaviour. So the **spec 053 pose-farm library will refuse a legitimate staging path today.**
  Left alone on purpose this session (it's a code change to another spec's library, outside the
  scope that raised it). Remove or retarget it before pointing any Core.Anim work at `H:\CLIENTS`.

Last updated: 2026-07-15 (1.0.0 M2 geometry FIXED + Ghidra header remap; sidebar tabs unblocked)

## 1.0.0 M2 + viewer fixes (2026-07-15, session)

- **1.0.0 M2 geometry now RENDERS** (USER-confirmed on a real client: DoomGuard, TrollMale).
  Root cause was `M2Era100Constants` **guessing** the section layout ("Per-field layout not fully
  recovered from Ghidra; using standard M2SkinSection fields"). Real `M2SkinSection` is ten uint16
  + C3Vector: **indexStart @0x08 uint16, indexCount @0x0A uint16** (reader had uint32 @0x08/0x0C,
  folding indexCount into indexStart's high bits → every section failed the bridge bounds check →
  "contained no drawable embedded sections"). **`Level` carries the high bits** of vertexStart/
  indexStart. Commit `7d7caa50`. New `M2Era100ModelReaderTests` (T003/T004, never existed before —
  the 9 "passing" M2Era1121 tests never touched the era-100 path); 3/4 fail on the old layout.
- **RECOVERED 1.0.0 M2 header map (Ghidra, WoW.exe 1.0.0.3980)** — commit `6b22bd80`. This beta has
  **one MORE array across 0x74-0xAC than documented v256**, so everything after texture_transforms
  is shifted one slot vs wowdev. From relocator strides in `FUN_0071e190`:
  `0x74` M2TextureTransform[] stride **0x54** (=3 old M2Tracks 0x1C, `FUN_0071fc40`) ·
  `0x7C` replaceableTexLookup int16 · **`0x84` M2Material[] stride 4** {u16 flags,u16 blendMode}
  (`FUN_0071fe10`) · `0x8C` boneCombos · **`0x94` textureCombos** · `0x9C` textureCoordCombos ·
  `0xA4` textureWeightCombos · `0xAC` textureTransformCombos (all int16, `FUN_0071f2c0` stride 2).
  Confirmed independently by runtime pointer reads in `FUN_0071a540`: model+0x98/+0xA0/+0xB0
  (M2Array = {count,offset}, so ptr at +N ⇒ count at N-4). We had read textureLookup from **0x7C**
  (the replaceable lookup) — that was the texturing bug.
- **STILL BROKEN (next session, in priority order):**
  1. **textureCombos is SIGNED int16; negative = REPLACEABLE texture.** `FUN_0071a540`:
     `if ((short)v < 0) v = *(ushort*)(~v * 0x20 + 0xc + *(int*)(runtime+0x9c));` — resolve via a
     runtime table (0x20-B records, field @+0xC). Our `TextureLookup` is `IReadOnlyList<ushort>`, so
     negatives become ~65535 → bounds check fails → null texture → **"everything is a single
     color"** (USER report on TrollMale — characters are almost all replaceable skins).
  2. **Blend modes never applied**: `BuildEra100Material` (`WowViewerM2RuntimeBridge`) hardcodes
     `renderFlags: 0` + `Opaque` + `isHeuristic:true`. The 0x84 material array is now located but
     NOT read. `FUN_0071a910`: flags bit **0x01=UNLIT**, **0x02=UNFOGGED**; blendMode @short 2
     (3/4 = additive). This is why everything is **shiny** (nothing is ever unlit). Runtime already
     has `renderFlags`/`rawBlendMode`/`M2GeometryRenderFlag` plumbing — wiring only, no new arch.
  3. **Animations: ROOT CAUSE FOUND + BONE LAYOUT FULLY RECOVERED (implementation is the only
     thing left).** `M2Era100ModelReader` builds its document with **`bones: null`** (line ~118) —
     1.0.0 models reach the renderer with **zero bone tracks**, so sequences advance the frame
     counter against nothing (USER: frame 2030/3333, static). Not a stride bug; the bones are
     simply never read.
     **M2CompBone = 0x6C, proven from relocator `FUN_0071f440`** (stride 0x6c; sub-array offsets
     decode exactly, and 3 tracks + pivot sum to 0x6C):
     `0x00` keyBoneId i32 · `0x04` flags u32 · `0x08` parentBone i16 · `0x0A` submeshId u16 ·
     **`0x0C` translation M2Track** · **`0x28` rotation M2Track** (values via `FUN_00720d30` =
     quaternion) · **`0x44` scale M2Track** · **`0x60` pivot C3Vector(12)**.
     **OLD M2Track = 0x1C** (confirmed): `+0x00` interpType u16 · `+0x02` globalSeq u16 ·
     **`+0x04` interpRanges M2Array over 8-byte M2Range{u32 start,u32 end}** · `+0x0C` timestamps
     M2Array · `+0x14` values M2Array.
     **SAMPLING ALGORITHM — FULLY RECOVERED from `FUN_0070f6d0`** (the track sampler; called 58×
     from `CM2Model::Update` `FUN_0070f960`). Signature
     `(model, uint time, int animIndex, track*, uint out[3])` → `out = {key0, key1, lerpT}`:
     ```
     if (track.interpRanges.count == 0) { start = 0; end = track.timestamps.count - 1; }
     else { start = interpRanges[animIndex].start;      // offset + animIndex*8
            end   = interpRanges[animIndex].end; }      // offset + animIndex*8 + 4
     if (end <= start) return {start, start, 0.0};                    // single key, no interp
     if (track.globalSequence != 0xFFFF)                              // u16 @ track+0x02
         time = *(u32*)(model[0x5c] + globalSequence*4);              // global timers override time
     k0 = search timestamps[start..end] bracketing time;              // cached-key fwd / bsearch / bwd
     k1 = k0 + 1;
     if (track.timestamps.count <= k1) return {k0, k0, 0.0};          // NB: clamps on TOTAL count
     return {k0, k1, (float)(time - ts[k0]) / (float)(ts[k1] - ts[k0])};
     ```
     Then `value = lerp(values[key0], values[key1], lerpT)` (slerp for the quaternion track).
     **KEY SEMANTIC:** `interpRanges` is indexed by **animation index**, stride 8 =
     `{u32 firstKey, u32 lastKey}` — **inclusive** bounds into the FLAT timestamps/values arrays.
     Empty interpRanges ⇒ use the whole array. Wrath+ instead nests `M2Array<M2Array<T>>` per
     sequence and drops interpRanges (track 0x1C→0x14).
     **REMAINING WORK:** our `M2TrackDefinition<T>` carries only `TimestampArray`/`ValueArray`
     (`M2TrackArrayReference`) shaped for the nested Wrath form. Add an interp-ranges reference +
     a flat-range sampling path, then populate `bones` in the era-100 reader with the 0x6C layout
     above. **Do NOT populate bones without the flat+ranges sampler** — wrong poses are worse than
     none, and guessed layouts caused every bug fixed this session.
- **Sidebar "missing panels" root cause (FIXED, `7d7caa50`)** — nothing was ever deleted.
  `_activeBottomTabIndex` was shared across TWO nesting levels: Tools > Terrain and
  Tools > Utilities re-read the parent's index, so "Utilities" (index 4) rendered
  `UtilitiesBottomTab 4` = **Taxi** and "Terrain" (3) rendered **Weak Signal**. Capture Automation,
  Asset Catalog, Runtime Stats, Minimap, Log, Perf, Render Quality, and the **MCNK** tab were all
  unreachable. Tell: `GetTerrainBottomTabLabels()` existed but was **never called** — the
  second-level tab strip was never drawn. Each nested level now owns its index + draws its strip
  (`DrawNestedSubTabStrip`).
- **Lighting readout added** (`b601b3dc`): Tools > Utilities > **Lighting** — LIT load state,
  version, source path, evaluated sample (dominant light, ToD, direct/ambient/fog/sky swatches, fog
  range), effective terrain lighting + external light-dir override. All state already existed on
  `WorldScene` (`LitStatus` even defaulted to "LIT not loaded.") and was simply never surfaced.
- **CALIBRATION (USER, unverified in code):** **time of day ≈ 0.64 matches the real minimaps**, NOT
  our `TerrainLighting.GameTime` default of **0.35** ("morning"). Directly relevant to T040 /
  synthetic-minimap truth — the capture default is probably wrong.
- **WMO-only maps don't load their WMO (fix WRITTEN, NOT BUILT/PROVEN — viewer was running).**
  `StandardTerrainAdapter:86` had `IsWmoBased = _existingTiles.Count == 0` — **ignores the MPHD
  flag**, unlike its sibling `AlphaTerrainAdapter:89` (`_wdt.IsWmoBased`, flag-based) and unlike the
  client (`FUN_006976f0`: `if MPHD&1 → MWMO+MODF`). Line 97 already consults the flag to *parse*
  placements, but `WorldScene:7166` gates *loading* on `IsWmoBased`, so a WMO-only WDT that flags
  any MAIN tile parses its placements, logs them, then never loads them. Changed to
  `(_mphdFlags & 0x1) != 0 || _existingTiles.Count == 0`. **RISK to check:** if hybrid maps (MPHD&1
  *with* real terrain tiles) exist, this now sends them down the WMO-only branch and skips terrain
  streaming — the comment at `StandardTerrainAdapter:100` claims hybrids exist. If so, fix the
  *load gate* instead (load global WMO whenever `ModfPlacements.Count > 0`, independent of
  `IsWmoBased`). Needs the USER's failing map name + log to confirm which branch actually fails.
- **Pre-existing, NOT ours:** 11 Core test failures (`WorldFramePassCoordinatorTests`,
  `LkToAlphaRoundTripTests`, `RawArraySerializerTests`, `AdtV23SummaryReaderTests`,
  `EnrichmentStreamFormatTests`, `ModelFootprintReaderTests`, `V18StorePlacementsReaderTests`,
  `AdtMcrfRealDataTests`) — verified identical against the parent commit by stashing.
- **USER direction (queued, not started):** retire staged clients in `output/tmp/wowarchive-clients`
  → point at a configurable **base clients folder** (their 10.5 TB WoWArchive: every build 0.5.3–
  3.3.5, all locales, PTRs), shared with the harvester scripts; **DBC schema fallback = use the last
  version that fits** when no exact DBD match (e.g. builds up to ~1.5 use the 1.0.1 schema). Also
  wanted: uniqueID-lineage visualization across builds; strip dead v7-dataset/training File-menu
  cruft into a real panel; **Ghidra: document 1.0.0 WDT flags** (undocumented for 20+ years).
- **Wireframe for MDX/M2/WMO reportedly broken** (USER). NOT the tab bug — the "M2/WMO WF" checkbox
  is in `DrawBottomBar()` (always reachable) and `RenderVisibleObjectWireframeOverlay` is intact.
  Last touched by `5266b25f`. Needs runtime diagnosis.

Last updated: 2026-07-15 (Spec 103 prefab curation and renderer-faithful lighting prepared)

## Spec 104 — 1.0.0 M2 route (2026-07-15)

- Correct contract: 1.0.0 assets are `MD20` / `0x100` **M2**, not MDX. `.mdx`/`.mdl` are
  compatibility aliases, never a substitute route for a 1.x M2.
- Spec 104 has been re-scoped to the first gated slice and now has `tasks.md`. The era-100 reader
  must fail loudly rather than fall through to a different M2 layout; standalone UI copy no longer
  tells users to browse MDX/MDL instead.
- Code proof: `dotnet build wow-viewer/WowViewer.slnx -c Debug` (0 errors) and focused
  `M2Era1121ModelReaderTests` (9/9) pass. Still unproven: user-run visible mesh/material render from
  a named staged 1.0.0 client, then a WotLK+ no-regression check. Do not claim rendering signoff yet.
- Correction from live viewer evidence: the first route still adapted the M2 into `MdxRenderer` and
  produced visibly wrong geometry. Era-100 now builds a native `M2StaticRenderModel` and calls
  `LoadM2RuntimeModel` with no `MdxFile`; the next viewer check must show `Renderer: M2Renderer`.

## For tomorrow (pick up here)

1. **Spec 103 corpus gate:** implementation and focused verification are complete, but the next
   training corpus is not approved yet. The USER runs quickstart §3d against the existing bounded
   Azeroth/Kalimdor map canvases, reviews `curation_summary.json` plus both evidence ledgers, then
   uses §5 to package that new prefab-reduced manifest. Do not reuse the historical 2,253-tile
   bundle. Quickstart §6 is the separate user-run LIT/DBC capture-calibration lane.
2. **WoWViewer CI**: built and locally validated (see entry below) — `.github/workflows/
   wowviewer-build.yml` exists at the true repo root (`i:\parp\parp-tools\.github\`, NOT under
   `wow-viewer/` — this repo is `akspa0/parp-tools`, not a submodule, confirmed via `git
   rev-parse --show-toplevel` and `.gitmodules`). Push it and watch the first real run; nothing
   else to build. To cut a release: push a `v*` tag, or run the workflow manually with
   `publish_release: true`.
3. **WoWViewer real cross-platform viewer** (separate, larger, NOT started): swap `BlpFile.
   GetBitmap()` → `GetImage()` at ~19 call sites (listed in progress.md) so the existing
   `WoWViewer.CrossPlatform.csproj` becomes an actually-functional Linux/macOS viewer, not just
   a compiling one. Sizeable, touches core rendering files — scope with the user before starting.

## WoWViewer v0.5.0 release push (2026-07-14/15, USER-driven)

- **Multiplatform:** GetBitmap()/GDI+ → ImageSharp GetImage() migration across all renderers/
  exporters/tools (delegated to agent, ~15 sites; the CA1416 runtime blocker for Linux/macOS).
  Workflow now publishes win-x64 + linux-x64 + osx-arm64 + osx-x64 self-contained builds on
  `v*` tag / manual dispatch; `create-release` job attaches all four zips + release notes.
- **Changelog:** `wow-viewer/docs/releases/v0.5.0.md` (new) — headline is the MdxViewer
  decoupling; README CI section updated. Old release-notes convention was per-version md
  bundled into the zip (v0.4.7 precedent) — kept.
- **Viewer fixes tonight:** WMOv14 group-name mismatch (MOGI positional overwrite — fixed);
  status-bar facing 180° off (displayed = yaw+180; true North = yaw 0 per USER report — fixed,
  E/W chirality should be eyeballed in-app once); AreaTable status-bar lookup never ran for
  `_vlmTerrainManager` sessions (condition required `_terrainManager` — fixed). Removed
  Model>LOD stub tab + WMO backface-culling toggle (USER: both useless).
- **M2 readers for 0.11–2.4.3 render nothing but bounding boxes** — top known gap. **Spec 104
  written (specify + plan done, 2026-07-14):** `specs/104-legacy-m2-rendering/` (spec, plan,
  research, data-model, quickstart, contracts/m2-format-profile.md, checklist). Root cause:
  `M2ModelReader.cs` hardcodes `embeddedSkinProfileCount/Offset = 0`; for M2 format version ≤ 263
  (client 0.11–2.4.3) the skin profiles (geometry/submeshes/material bindings) are EMBEDDED in the
  .m2 (`nViews`/`ofsViews`), not external `.skin` files. Phased by format-version boundary:
  P1 = 2.4.3+1.12.1 (documented, validate vs wowdev.wiki + reference impl, no debugger),
  P2 = 2.0.0α/2.1/2.2/2.3, P3 = 1.0.0/0.12/0.11 (x64dbg dynamic tracing). x64dbg MCP configured +
  responding (`.mcp.json` → `C:\x64dbg`, `list_sessions` OK); **Ghidra NOT installed** (static RE
  is a separate setup step). Next speckit step: `speckit-tasks`. Not yet implemented — spec/plan only.
  NOTE: `.specify/*` scripts resolve the git root (parp-tools) not `wow-viewer/`, so spec artifacts
  are written to `wow-viewer/specs/` manually.
- **Status-bar facing fully fixed (2026-07-14):** was N/S-swapped (`yaw+180`), then E/W-swapped
  after the first fix; final form `(360 - yaw) % 360` (N/S are negation fixed points). USER to
  confirm all four cardinals in-app.

## Current target — Spec 103: revive the v7 terrain regressor on clean signals

- **New planned curation gate (2026-07-15):** The next real-data V8 corpus must be reduced by
  provenance-backed pattern/context coverage, not raw tile count. Spec 103 Phase 3B consumes (does
  not duplicate) Spec 076's full-map fractal/paste library; it must ledger every available alpha
  layer per map/tile/chunk/cell with region/family identity, terrain/MCLY, and object/liquid context,
  then retain deterministic representatives with duplicate lineage and family-safe splits. This is
  training-time curation only: it does not add alpha/object/mesh inputs to image-only deployment.
  **Correction:** ADT tiles are storage pages, not curation units. The ledger must begin per
  map-wide canvas/layer and preserve multi-scale fractal/cellular neighbour composition plus MCLY
  tileset anomalies that repeat with placements; tiny local brush strokes are explicitly worthless
  as final curation units.
  **Vocabulary correction:** canonical family = terrain-art **prefab**; its placements may be
  translated/mirrored/rotated/retextured. User reports initial 0.5.3–3.3.5 analysis found ~140
  prefab families. Retained MCLY texture is a tileset variant/provenance signal, not automatic
  family separation; split grouping must be at canonical prefab level.
  **Purpose:** reverse-engineer an explainable, editable historical art pipeline from image/game-data
  breadcrumbs. Preserve recovered evidence separately from model proposals and operator hand edits;
  the viewer/export path is the human-authoritative finishing step, never an automated claim of
  historical truth.

- **Governing law (image-only):** the only deployment input is one image tile. Every other signal is generated from it; no model reads a ground-truth signal at inference; downstream trains on generated (not ground-truth) upstream; a target the image cannot support is invalid. Validation is **label-free** (self-consistency), never label-comparison. See `specs/103-image-only-reconstruction/spec.md`.
- **Implemented (agent side, 2026-07-13):** v7 contract pinned in `specs/103-.../research-v7-contract.md`;
  lane ported to `data-harvester/src/harvester/spec103/` (`v7_model.py` — only deviation: `output_size`
  parameterized, 256 default; `v7_losses.py` verbatim; `v7_inputs.py` 13-ch assembler). 7/7 CPU sanity
  tests green (`tests/spec103/`). All Phase 2–4 scripts written: `spec103_make_synthetic_adts.py`,
  `spec103_build_synthetic_store.py`, `train_spec103_v7.py`, `infer_spec103_v7.py`,
  `spec103_build_real_store.py`, `spec103_export_mesh.py`, `validate_spec103_labelfree.py`.
  Commands: `specs/103-.../quickstart.md`. **Blocked on USER runs**: capture, training, T011 caveat catalog, T018 shadow capture.
- **Pinned 13-ch truth (plan's old aux guess was wrong):** 0-2 minimap, 3-5 normals (both recovery-attenuated
  ×0.85/×0.70 then ImageNet-normalized), 6 WDL prior (outer 17×17 only, align_corners=True, **0.5 fill when
  missing — dropout reuses this**), 7-8 tile height min/max hint planes (`--height-hints gt|wdl|none`),
  9 liquid mask, 10 liquid height, 11 object mask, 12 brush (zeros). Loss reads 9/11/12 — order is load-bearing.
  **The model architecture is unchanged (13 channels).**
- **WDL prior = verified transform:** `outer = height257[::16,::16]`, `inner = height257[8::16,8::16]`.
  Derived at batch time from `height_257` — no reharvest, nothing stored. **Never** `wdl_height_33`.
- **Procedural-synthetic PoC DROPPED as a gate (USER decision 2026-07-14):** flat/ramp/ridge/crater
  patterns don't replicate real terrain, and the WDL prior trivially solves them (v8 run: init
  l1_g ≈ 0.0006 — nothing to learn on the global channel). The intended "synthetic" lane was always
  **signals synthesized FROM real terrain** (deterministic shadow/hillshade renders of real height —
  T018's reinterpretation), never invented terrain. The 10-tile procedural store survives only as a
  pipeline smoke test. **Soundness test = the real-data v8 run** (quickstart §3; everything ready:
  V18 store + curation manifest, 2253 kept, Azeroth holdout 332/1921).
- **Synthetic chain (kept for smoke tests; all existing C# used as-is):** `map generate-blank` (Inspect tool) → known-height .npy →
  `terrain-patch-adt` (Converter) → `Capture render` (perspective-camera caveat recorded) or
  `--synthesize-minimaps` hillshade fallback. Synthetic tiles are placed non-adjacent so the patcher's seam
  stitching never mutates a known pattern.
- **Curation is mandatory (FR-013 / Principle #5), clean-by-default:** object tiles are impossible height
  targets (terrain under an object is occluded in the minimap), so they are DROPPED, not learned — the user
  was right and I initially defaulted keep-all in violation of the spec; fixed. `spec103_curate_dataset.py`
  buckets every tile and drops object_contaminated / blank_minimap / height_normal_mismatch, writes an
  auditable `curation_manifest.parquet` (+ map/height-regime buckets) the trainer consumes via
  `--curation-manifest`. **Default `--max-object-coverage 0.0`** (drop ANY object; was 0.02).
  Recorded V18 proof: 5134 total, 2650 object-free before other gates, 2253 kept after blank/signal
  filtering. `1.0` is v7-faithful keep-all ablation only. Trainer reports `val_no_prior` every epoch
  (prior-dropout robustness).
- **Banding investigation (2026-07-14):** verified live against V18 zarr — height_257/normal_xyz/
  liquid_height/object_precise_mask are all float32; only minimap_rgb is uint8 (correctly, the
  deployment image). No precise data is routed through 8-bit image encoding. Real causes found:
  (1) `output_head_mode` was never exposed to the trainer — every run silently hard-clamped a
  tanh-scaled residual every step (tanh saturation → residual clusters near ±scale = plausible
  v7 banding/terracing source); now `--output-head-mode {legacy_clamped, linear_unclamped_train}`,
  recorded in checkpoints, auto-resolved by inference. (2) v8's PixelShuffle upsampling lacked
  ICNR init (Aitken 2017) — a checkerboard-artifact class v7 never had (bilinear+conv instead);
  fixed + regression-tested. Left as-is (shared v7/v8, not a bug): the 17×17 WDL prior is only
  C0-continuous when bilinear-upsampled to 256×256 — visible ~16px facets the ±0.20 residual
  can't fully correct; watch for it in `val_previews/`. 15/15 tests green.
  Full writeup: `specs/103-image-only-reconstruction/research-v8-optimization.md` §6.
- **Local GPU training is OFF (2026-07-14):** USER's GPU overheated mid-run; **no more local
  training runs** — the path forward is RunPod deployment (see [[project_v24_runpod_migration]]
  for prior RunPod lessons: US datacenters only, runpodctl.exe location, verify before killing).
- **RunPod deployment built (2026-07-14, T022):** `scripts/package_spec103_runpod.py` +
  `runpod/spec103/{install_deps,verify_bundle,smoke,train}.sh`. Bundle subsets BOTH fields
  (only the 6 arrays `train_spec103_v7.py` reads, not the V18 store's other 18) AND rows
  (curation-kept only) — measured **3.2 GB -> 127 MB** (2253/5134 tiles), verified end-to-end
  through the real `V7TileDataset`. No HF downloads (v8/v7 train from scratch). Added `--limit`
  to the trainer for the smoke stage; `train.sh` always passes `--resume` (spot-preemption
  safe). Command: quickstart.md §5.
- **v8 is the PRIMARY architecture (USER decision 2026-07-13; implemented + tested):**
  `V8LeanUNet` (`src/harvester/spec103/v8_model.py`, ConvNeXt-V2 blocks, pixel-shuffle decoder,
  global-context mixer) — measured **6.2M params / 16.4 GFLOPs @256** vs v7's 117.06M / 119.9
  (73% of v7's params sat at 8×8–16×16). Identical 13-ch/trestle/bounds contract → loss, trainer,
  inference, previews, harness all unchanged. Trainer default `--arch v8` (`--arch v7` = 117M
  ablation only, NOT a gate); checkpoints record arch, inference auto-resolves. 13/13 CPU tests.
  Driver: v7's ~26 h time-to-signal was unacceptable; v8 targets minutes on synthetic. Survey +
  rationale: `specs/103-image-only-reconstruction/research-v8-optimization.md` (T021).
  Excluded: DA-family (blacklist), diffusion predictors, 100M+ depth foundations.
- **The USER runs all training/capture/heavy jobs.** The agent prepares scripts + commands only (AGENTS RULE 0).

## WoWViewer CI + cross-platform build (2026-07-14, new lane)

- **GitHub Actions added:** `.github/workflows/wowviewer-build.yml` (repo root — the actual
  GitHub repo is `akspa0/parp-tools`; `wow-viewer/` is a plain subdirectory, not a submodule).
  Three jobs: (1) build+test on `windows-latest` via `WowViewer.slnx` (the real, functional
  viewer — always runs); (2) compile-only check of `WoWViewer.CrossPlatform.csproj` +
  4 confirmed-portable tool projects on `ubuntu-latest` (`continue-on-error: true` — advisory,
  keeps the port from bit-rotting without gating on non-functional-yet code); (3) publish a
  self-contained win-x64 build + GitHub Release, gated on a `v*` tag push or manual
  `workflow_dispatch` with `publish_release: true` (never auto-triggered — matches
  [[feedback_no_auto_deploy]]). All three validated **locally** before commit: full solution
  build (0 errors), CrossPlatform target build (0 errors, 435 warnings — mostly the predicted
  CA1416 GDI+ hits), portable tools build clean.
- **Audit finding (Explore agent, full results in progress.md): the cross-platform port is
  further along than expected but NOT functional yet.** `WoWViewer.CrossPlatform.csproj`
  (plain `net10.0`, no WinForms) already existed, compiles cross-platform-clean at the TFM/
  dependency-graph level, and the three WinForms file-dialog calls in `ViewerApp.cs` were
  already correctly `#if WINDOWS`-guarded (SDK auto-defines `WINDOWS` only for `-windows`
  TFMs) with graceful `return null` fallback. **The real blocker: `BlpFile.GetBitmap()`**
  (`SereniaBLPLib`, System.Drawing/GDI+) is called at ~19 actual rendering/export sites
  (M2Renderer, WmoRenderer, TerrainRenderer, MinimapRenderer, GlbExporter, MapGlbExporter,
  LoadingScreen, AssetProbe, Core.Renderer's TextureCache, MDX-L_Tool's TextureService, plus
  harvest/converter/mask-validate tool code) and **throws `PlatformNotSupportedException` at
  runtime off-Windows since .NET 7** — compiles fine, crashes on first texture load. The fix
  (`BlpFile.GetImage()`, ImageSharp-based, already exists and is already used correctly in
  `BlpRgbReader.cs`/`AlphaBlpCompatibilityService.cs`) is scoped but NOT done — a real,
  sizeable follow-up task (~19 call sites in core rendering code), not started without user
  sign-off given the blast radius.
- **Fixed in this pass (small, unambiguous, verified compiling):** two hardcoded-backslash
  filesystem-path bugs that would break on Linux — `tools/harvest/.../Program.cs:398` and
  `tools/converter/.../LkToAlphaCommand.cs:1885`, both now `Path.Combine`. (Distinct from MPQ
  virtual-path strings elsewhere, which correctly and intentionally use `\` as the game-data
  convention — those were not touched.)
- **`WowViewer.Tool.ValidationCapture` is deliberately, permanently Windows-only by design**
  (throws `PlatformNotSupportedException` itself for its GPU hidden-window capture path) — not
  a portability bug, never expected to run on Linux.
- **Confirmed portable today** (tool-project level): `inspect` (`map generate-blank`),
  `wdl-read`, `enrich`, and `converter`'s `terrain-patch-adt` subcommand specifically (its
  other, minimap-related subcommands still hit `GetBitmap()`). `capture`, `harvest`, and
  `mask-validate` still have the GetBitmap runtime landmine.
- **`WowViewer.CrossPlatform.csproj`, `WmoMinimap`, `V22Enrich`, and `App.Defunct` are not in
  `WowViewer.slnx`** — deliberately left out of the solution file (CI builds them by direct
  csproj path instead) to avoid changing the user's local `dotnet build WowViewer.slnx` behavior
  without being asked.

## Spec 104 / 1.0.0 M2 — Ghidra static trace DONE (2026-07-15)

- **Ghidra MCP now live**: `H:\ghidra_11.3.2_PUBLIC` + GhidraMCP plugin (HTTP API
  `127.0.0.1:8080`) + `bridge_mcp_ghidra.py` wired into `.mcp.json` as the `ghidra`
  server (`uv run --script`). x64dbg MCP still broken (anaconda env gone — separate fix).
- **Viewer gap (user clarification 2026-07-15)**: the wow-viewer M2 reader already
  handles **0.11/0.12** (pre-`0x100`) fine; **1.x+ does not render correctly**. The
  format expanded incrementally 1.0 (`0x100`) → 3.0.1. The 1.0.0 game client's parser
  (`FUN_0071e190`) hard-requires `MD20`+version `0x100` (rejects others as `Corrupt
  model data`) — this confirms the 1.x on-disk format is `0x100` with the recovered
  layout; it is NOT the viewer's bug. Extension gate is not a factor (`.mdx`/`.mdl`→`.m2`).
- **User design direction**: the M2 reader should **accept any version** and dispatch to
  a **per-version codepath** (one per layout-change step 1.0→3.0.1), not hard-reject.
  This Ghidra trace fully specifies the **`0x100` (1.0.0/1.x) codepath**.
- **1.0.0 = version 0x100** (same version field as 1.12.1) — CORRECTS Spec 104
  `research.md` which had grouped 1.0.0 with 0.11/0.12 as "pre-256". Only 0.11/0.12
  are pre-256.
- **Fully embedded format**: no external `.skin` or `.anim` on 1.0.0. Skin profiles =
  `data->divisions` (0x4C, M2Division 0x2c) with vertexLookup/indices/sections(0x20)/
  batches(0x18). Complete header field map + all block sizes recovered.
- **Shaders**: `.bls` + CGx (CGxVertexShader/CGxTexFlags) + GL_NV_register_combiners —
  NOT the later `Combiners_*`/`Diffuse_*` named-effect system. M2 options minimal
  (`M2UseShaders`, `M2UseThreads`).
- **Deliverables**: `specs/104-legacy-m2-rendering/research-1.0.0-ghidra-trace.md`
  (full), `contracts/m2-format-profile.md` (1.0.0 entry populated), raw decompilations
  in `specs/104-legacy-m2-rendering/evidence/1.0.0-ghidra/` + `output/ghidra_1.0.0/`.
  Ready for a fresh chat to implement the 1.0.0 (`0x100`) reader branch.
- **Still open**: pre-`0x100` (0.11/0.12) layout (needs 0.12 client in Ghidra); 1.0.0 vs
  1.12.1 header diff; render validation against a staged 1.0.0 client.

## WoW 1.0.0 renderer features — Ghidra trace DONE (2026-07-15)

- **Doc**: `docs/architecture/wow-1.0.0-renderer-features-ghidra-trace-2026-07-15.md`.
  Goal: eventually reproduce all 1.0.0 client rendering in the wow-viewer renderer.
- **Liquids**: 1.0.0 has **12 liquid types** (`LIQUID_COUNT=0xC`, `LIQUID_NONE`), not
  just magma/water. `liquidTexBaseName[type]` table @ `0x00834d4c` → river_lake_a,
  river_fast_a, ocean_h, slime, lava, splash, water. **30 animated frames/type**
  (`FUN_00686d40`). **MCLQ** chunk (pre-3.0, NOT MH2O), MCNK sub-chunk @ +0x60
  (`FUN_006b4920`). `.bls` shaders: ocean0_s, MapObjExtWater0. Ripple: `Water0Ripple`/
  `WaterRadWave`. Cvars: waterParticulates/Ripples/Specular/Waves/MaxLOD/LOD, SetWaterDetail.
- **Particles**: `M2Particle=0x1f8` @ M2 0x13C; emitters `CParticleEmitter2` +
  Plane/Sphere/Spline subclasses, `ParticleSystemManager`, child emitters, `particleDensity`,
  footprint particles. `GetEmitter`=`FUN_0070ef60`.
- **Ribbons**: `M2Ribbon=0xdc` @ 0x134; `CRibbonEmitter`/`RibbonManager`/`CRibbonMat`.
- **Attachments (armor/equipment)**: `M2Attachment=0x30` @ 0x104 = {boneIndex(ushort)@0x04,
  pos(3f)@0x08}; worldXform = modelWorld * boneMatrix[bone] * offset (`FUN_0070e500`).
  API: HasAttachment/GetAttachment{Pivot,Position,WorldTransform}. Mount/Character/Pet
  attachments, `GetInventorySlotInfo`.
- **Helmet/hair geosets**: `HelmetGeosetVisData.dbc`, `CharHairGeosets.dbc` (`FUN_0057ef40`).
- **Skybox**: sky = M2/MDX models `Environments\Stars\{stars.mdl,DeathClouds.mdx,
  StratholmeSkybox.mdx}` + `LightDataSky`/`DNOverrideSky`; cvars SkyShow/SunGlare/CloudLOD/
  CloudDensity/CloudLayers. Sky init `FUN_006ce6c0`.
- **Camera/POV**: `CGCamera`/`CSimpleCamera` (smoothed yaw/pitch/zoom orbit) + `M2ModelCamera`
  (0x7c @ M2 0x124, model-authored cameras for preview/portrait). No true first-person mode
  on 1.0.0. Camera per race/sex via DBC.
- **Cross-cutting**: all render via CGx (`.bls` + register combiners), no `Combiners_*`/
  `Diffuse_*` named-effect system. Evidence: `specs/104-legacy-m2-rendering/evidence/1.0.0-ghidra/`
  (33 .c files). Open: per-field layouts of M2Particle/Ribbon/Camera records + liquid type
  index-map (read 12 ptrs @ 0x00834d4c).

## WoW 1.0.0 deep-dive (M2 layouts + WMO + dev tools) — DONE (2026-07-15)

- **Doc**: `docs/architecture/wow-1.0.0-deep-dive-ghidra-trace-2026-07-15.md`. Build is
  **beta 3** (`BETA_BUILD` string present). Sets up the whole 1.x object era.
- **M2 per-field layouts** recovered for every block (Bone 0x6c, Vertex 0x30, Division 0x2c,
  Sequence 0x44, Texture 0x10, Color 0x38, TexWeight/Transform 0x1c, Attachment 0x30,
  Event 0x2c, **Light 0xd4** [>3.3.5's 0x9c, ~6 tracks], **Camera 0x7c** [src/tgt/near/far/fov
  tracks], **Ribbon 0xdc**, **Particle 0x1f8** [2 string bufs + ~16 tracks]). Relocator
  legend documented. Enough to parse all of 1.0.0 M2; only per-field *semantic* naming remains.
- **WMO/WDT**: WDT = MVER→MPHD(0x20)→MAIN(0x8000)→[if MPHD&1: MWMO+MODF] (`FUN_006976f0`).
  WMO group **version 0x11** (`FUN_006c5380`), MOGP + sub-chunks MOPY/MOVT/MOLR/MOBA/MOCV/MLIQ,
  0x18-B batches, **max 12 portals/group**, `missingwmo.wmo` fallback, `WMOAreaTable.dbc`,
  doodad anim. Dev-humor assert `lameAssLink_IsLinked`.
- **Dev/dead code (live, referenced)**: `BETA_BUILD`, **Godmode** cheat, **developer console**
  (`ConsoleExec`/`SetConsoleKey`), **profiler** (`ProfileInternal`), debug toggles
  (`debugTargetInfo`/`TogglePortals`/`GetDebugStats`), `FIXME: Not yet implemented` leftover,
  intro movie. Console is the easy dynamic-validation entry point.
- **Evidence**: 36 .c decompilations in `specs/104-legacy-m2-rendering/evidence/1.0.0-ghidra/`.
- **Open**: M2 track semantic naming; M2Vertex field split; WMO *root* .wmo reader
  (MOHD/MOGN/MOGI/MOTX/MOMT/MOPV/MOPT/MOPR/MODS/MODD); liquid type index-map; full console
  command table.

## WoW 1.0.0 WMO rendering pipeline — Ghidra trace DONE (2026-07-15)

- **Doc**: `docs/architecture/wow-1.0.0-wmo-rendering-ghidra-trace-2026-07-15.md`.
  Goal: upgrade wow-viewer from brute-force renderer to proper 1.0-era world renderer.
- **WMO shaders (6 .bls)**: MapObjSpecular, MapObjTransSpecular, MapObjTransDiffuse,
  MapObjOverbright, MapObjMetal, MapObjExtWater0 — all loaded from `FUN_006abab0`.
  Shader backends: GL_NV_register_combiners, GL_NV_texture_shader, GL_ATI_fragment_shader,
  GL_ARB_fragment_program, D3D ps_1_1–ps_2_0.
- **Batch system**: `intBatchCount` (opaque) + `transBatchCount` (transparent) per group.
  VBOs: `group->vertexVB` / `group->indexVB` (GxBufSize). Liquid verts: `group->liquidVerts`.
  Render list: `renderList.IsLinked(batch)`. Max: `Gx_MaxBatchCount`.
- **Lighting**: 3 layers — MOCV (pre-baked vertex colors), MOLR→CMapLight (dynamic),
  CMapCacheLight (cache). `mapObjLightLOD` (0-2), `mapObjOverbright`. Light linking:
  `mapObjDefGroup->lightLinkList`. Dir light: PLightDirIntens/Color/Pos. Ambient format string.
- **Fog**: per-group (`SMOGroup::NUM_FOGS`), `FogQ`/`LightDataFog`. OpenGL glFogfv/f/i.
  ARB options: exp2/exp/linear. Console: SetFogNear/Far/Color/ClearFog.
- **Portals**: max 12/group (`portal->count <= 12`). `USPortalExt` struct. Debug: TogglePortals,
  Portal display/vis.
- **BSP**: MOBN (nodes) + MOBR (refs) + MORB (render batches). `AaBsp.cpp`. Node cache:
  `bspcache`/`BSP node caching`. Debug: BSP render enabled/disabled.
- **Liquids (WMO)**: MLIQ chunk (WMO-internal), MCLQ (terrain). 12 types (`LIQUID_COUNT=0xC`).
  `liquidTexBaseName[type]` → ocean/lava/slime. `CChunkLiquid`. Ripple: `Water0Ripple`/
  `WaterRadWave`. `MapObjExtWater0.bls` for WMO near water.
- **Doodads**: `CMapDoodadDef` (M2 in WMO). Detail doodad system (CDetailDoodadData/Geom/Inst).
  Linked via `mapObjDefGroup->doodadDefLinkList`. Toggles: showSimpleDoodads/showDetailDoodads.
- **Scene**: `WorldScene.cpp` (40+ functions in 0x0067cxxx-0x00682xxx). Query flags:
  WQF_doodadMask/gameObjMask/terrain/liquid. Vis lists: visMapObjDefGroupList/visDoodadList.
  Frustum: `mapObjDefGroup->frustumList`. Fadeout: `CWModelFadeout`.
- **CGx abstraction**: CGxDeviceOpenGl/D3d, CGxPixelShader (nvrc/arbfp1/ps_1_1/ps_2_0),
  CGxVertexShader, CGxStateBom (state batching), CGxVboBroker, CGxTex/CGxTexCache/CGxTexFlags.
  Vertex formats: CGxVertexPC (Pos+Color), CGxVertexPT0T1 (Pos+Tex0+Tex1).
- **WMO chunks confirmed**: MOGP, MOPY, MOVI, MOVT, MONR, MOBA, MORB, MOBR, MOBN, MOLR, MOCV, MLIQ.
- **Function map**: 16+ WMO render funcs (0x006b9xxx-0x006bcxxx), 6 shader load (FUN_006abab0),
  40+ WorldScene funcs, 23+ MapChunk funcs, 15+ MapObj funcs, 5 MapObjDef funcs.
- **Open**: decompile WMO render funcs (decompile endpoint was down); MOPY flag bits;
  MOBA/MORB batch struct; MOLR light format; MLIQ liquid format; portal vis algorithm;
  BSP traversal; WorldScene render order; CGxStateBom; CWModelFadeout algorithm.

- **DECOMPILED (8 functions, targeted code path tracing)**: Via GhidraMCP v5.14.2 on
  port 8089 (Ghidra 12.1.2). Evidence:
  `specs/104-legacy-m2-rendering/evidence/1.0.0-ghidra/wmo_render_pipeline.c` +
  `wmo_transparent_batch_renderer.c`.
  - `FUN_006abab0` = WMO shader loader (calls FUN_0058ee90 ×6)
  - `FUN_0058ee90` = shader load (virtual call on CGx device vtable[0xb4])
  - `FUN_006ba9d0` = WMO group VBO setup (vertex/index buffers, mode 3/4)
  - `FUN_006babc0` = **opaque batch renderer** (BSP-ordered batches, materials, state)
  - `FUN_006baea0` = batch draw setup (primType, baseVertex, startIndex, count, primCount)
  - `FUN_006ba940` = batch visibility check (6 shorts = bounding box → frustum cull)
  - `FUN_0067e340` = frustum cull wrapper → FUN_006827e0
  - `FUN_006baf70` = **transparent batch renderer** (mode 4, vertex colors, fog, specular)
- **RECOVERED STRUCT LAYOUTS**:
  - **Batch struct = 0x18 (24) bytes**: bbox minXYZ@0x00-0x0a (6 int16), baseVertex@0x0c,
    startIndex@0x10, count@0x12, primCount@0x14, flags@0x16 (bit0=strip, upper nibble=state),
    materialIndex@0x17
  - **Material struct = 0x40 (64) bytes**: flags@0x00 (bit3=render state, bit4=use material
    color, bit5=use vertex colors/MOCV), passCount@0x04 (0=single, 1=specular, 2=extended
    specular), color@0x14
  - **CMapObjGroup fields**: vertexVB@0x04, indexVB@0x08, transBatchCount@0x3c,
    intBatchCount@0x3e, BSPBatchCount@0x40, batchArray@0xd8, vertexBufSize@0xe8,
    indexBufSize@0xec, totalBatchCount@0x138, materialArray@0x1d8
  - **Batch array ordering**: [transparent (0x3c)] [interior (0x3e)] [other]
  - **Context object fields**: vertexColor RGB@0x110-0x112, fogColor RGB@0x118-0x11a,
    ambientColor RGB@0x11c-0x11e, materialArray@0x1d8
  - **Global state**: DAT_00a8732c (ambient@0xb0-0xb8, fog@0xbc-0xc4),
    DAT_00aadce4=specular enabled, DAT_00aadec1=specular supported
  - **Constants**: 0.003921569=1/255 (byte→float), 0x41600000=8.0f (specular intensity)
- **DECOMPILED (targeted code path tracing)**: 5 functions decompiled via
  `/decompile_function?address=0x...` endpoint. Evidence:
  `specs/104-legacy-m2-rendering/evidence/1.0.0-ghidra/wmo_render_pipeline.c`.
  - `FUN_006abab0` = WMO shader loader (calls FUN_0058ee90 ×6 for 6 .bls shaders)
  - `FUN_0058ee90` = shader load (virtual call on CGx device vtable[0xb4])
  - `FUN_006ba9d0` = WMO group VBO setup (creates vertex/index buffers, validates sizes)
  - `FUN_006babc0` = **WMO batch renderer** (iterates BSP-ordered batches, looks up
    materials, sets render state, draws — THE main render function)
  - `FUN_006baea0` = batch draw setup (fills draw params: primType, baseVertex, startIndex,
    count, primCount)
- **RECOVERED STRUCT LAYOUTS**:
  - **Batch struct = 0x18 (24) bytes**: baseVertex@0x0c, startIndex@0x10, count@0x12,
    primCount@0x14, flags@0x16 (bit0=strip, upper nibble=render state), materialIndex@0x17
  - **Material struct = 0x40 (64) bytes**: flags@0x00 (bit4=use material color),
    passCount@0x04 (0=single, 1=two-pass specular), color@0x14
  - **CMapObjGroup fields**: vertexVB@0x04, indexVB@0x08, transBatchCount@0x3c,
    intBatchCount@0x3e, batchCount@0x40, batchArray@0xd8, vertexBufSize@0xe8,
    indexBufSize@0xec, materialArray@0x1d8
  - **Draw params**: primType@0x00 (3=triangles, 4=strip), baseVertex@0x04,
    startIndex@0x08, count@0x0a, primCount@0x0c
  - **Globals**: DAT_00a93790=use strips, DAT_00aadec1=specular enabled,
    DAT_00a1ce58=CGx device ptr
- **Additional findings (deep sweep)**: MapRender.cpp (main map render entry, separate from
  WorldScene/MapObjRender). CSimpleRender engine framework + RENDERCALLBACKNODE. Culling:
  DistCull (distance, 1.0-max), SmallCull (size, 0.001-2.0), showCull debug. Triangle strips
  toggle (requires restart). Vertex opt regions (optRegion->vertexStart/Count). Max verts
  0x40000 (262K). Max 2 textures/batch (M2). Sort entries for transparent ordering. Texture
  cache (TextureCache.cpp, CTextureHash, TEXTURECACHEROW). Render state stack (CGxPushedRenderState,
  Gx_MaxRsStackDepth). Perf counters (GxPerfCounters_Last). WMO root chunks (MOHD/MOGN/MOGI/etc.)
  NOT in assertion strings — root reader uses generic chunk reader without per-chunk validation.
- **BLOCKED**: decompile + xrefs endpoints down (plugin context lost, API overloaded from
  string sweeps). Remaining items need decompilation: MOPY flag bits, MOBA/MORB batch struct,
  MOLR/MOCV formats, MLIQ liquid format, portal vis algorithm, BSP traversal, WorldScene
  render order, CGxStateBom, CWModelFadeout algorithm. Next session: restart GhidraMCP plugin,
  use decompile SPARINGLY (one function at a time, not batch sweeps).

- **UNBLOCKED + 5 items RESOLVED (2026-07-15 follow-up)**: GhidraMCP decompile back up
  (WoW.exe 1.0.0.3980, base 0x400000). Doc updated (§20) + evidence
  `evidence/1.0.0-ghidra/wmo_scene_portal_bsp.c`. ~20 funcs decompiled one-at-a-time.
  - **MOGP parsers**: `FUN_006c55a0` mandatory (MOPY@0xc0 size>>1, MOVI@0xc4, MOVT@0xcc size/0xc,
    MONR@0xd0, MOTV@0xd4 size>>3, MOBA@0xd8 size/0x18); `FUN_006c5810` optional gated by SMOGroup
    flag group[0x10]: 0x200 MOLR, 0x800 MODR, 0x1 MOBN+MOBR(BSP), 0x4 MOCV, 0x1000 MLIQ, 0x20000 MORI+MORB.
  - **MOPY** = 2 B/tri `{flags:u8, materialId:u8}`, cnt size/2. Runtime: renderer ignores materialId
    (MOBA carries matIndex@0x17; 0xFF=collision-only never batched); flags = per-face COLLISION FILTER
    MASK cached `flags&0x7f` in BSP cache (`FUN_00696bf0`), tested `(flags&queryMask)==0→skip` in
    `FUN_006a2c60`(box)/`FUN_006a2840`(line). Bit names cross-ref'd to documented SMOPoly set.
  - **MLIQ** (flag 0x1000): 30-B header {xVerts,yVerts,xTiles,yTiles u32×4, baseX/Y/Z f32×3, matId u16}
    →group[0xf4..0x110]; verts@group[0x114]=xVerts·yVerts×8B; tileFlags@group[0x118]=xTiles·yTiles×1B.
  - **BSP** (AaBsp.cpp): 16-B node {flags u16(0x4=leaf,low2=axis), neg i16@2, pos i16@4, nFaces u16@6,
    faceStart u32@8, planeDist f32@0xc}; MOBR=u16 refs. Ray `FUN_006965f0`, AABB `FUN_00696820`, leaf
    `FUN_00696560`; 8-way cache `FUN_00696ab0`/`FUN_00696bf0`. WMO RENDER batch select is FRUSTUM-based
    (`FUN_006babc0` culls each MOBA bbox); BSP is the COLLISION tree.
  - **WorldScene render order** = `FUN_0067c460` (CWorldScene::Render, from map-top `FUN_006742e0`):
    begin → `FUN_0067d4f0` cam-in-WMO test → outside=`FUN_0067e3c0`/inside=`FUN_00681690`+drain CExtView
    (max16) → opaque → fog select → transparent/effects → portal debug. Frustum STACK 32×0xfc @DAT_00a7a758,
    push `FUN_0067d760`/pop `FUN_0067e390`/build `FUN_0067dd30`.
  - **Portal vis** = SCREEN-RECT culling. Root arrays MOPV@scene[0x134], MOPT@[0x138] (20-B: startVtx,count,
    C4Plane), MOPR@[0x13c] (8-B: portalIdx,groupIdx,side,filler). `FUN_006ba230` projects portal→SPortalExt
    (0x1c B: flags,minX/minY/maxX/maxY,stamp); recursion `FUN_006b9d30` (back-face + rect-intersect + push
    sub-frustum + recurse cap DAT_00ab5d5c; exterior flag 0x8 → CExtView). Seeds `FUN_006b9600`(in)/
    `FUN_006b9900`(out); visible = frame stamp DAT_00aade18.
  - **Still open**: MOBA per-batch light/color; MOLR per-light record; MOCV consume; CGxStateBom;
    CWModelFadeout; MCLQ terrain-liquid grid.

## World render systems (M2 draw / terrain surface / water / blend) (2026-07-15)

- **`docs/architecture/wow-1.0.0-world-render-systems-2026-07-15.md`** + evidence
  `world_render_systems.c` — the geometry/material/blend pipeline (complements the lighting doc).
- **M2/doodad render**: tick `FUN_006bf060` → build list `FUN_00716c40` → dispatch `FUN_0071a150`
  (entry types 0 opaque/1 transparent-sorted/2 multi/3 particle/4 ribbon/5 proj-shadow), geo draw
  `FUN_0071b550`. **renderFlag={u8 flags,u16 blendMode}**: flags 0x01=UNLIT, 0x02=UNFOGGED (0x04
  two-sided…). **blendMode 0-6**: opaque/alphakey/alpha/add/add/mod/mod2x (std GL map in doc §2.3).
  **Max 2 tex/batch** + animated UV matrix (`FUN_0071a540`); color+alpha from anim tracks. Opaque
  pass then back-to-front transparent pass.
- **Terrain**: VBO **24 B = pos+normal** (`FUN_006c0db0`) → **hardware FFP-lit** (RESOLVES lighting
  doc's FFP-vs-bake open item; normals uploaded, GPU N·L). 4 MCLY layers × MCAL alpha, LOD strips.
- **Water** (MapWater.cpp): 12 types, **flipbook-animated `XTextures\<type>\*.%d.blp` ~30 frames**,
  8×8 tile grid/chunk with depth→shore fade, ocean0_s.bls/MapObjExtWater0.bls, drawn transparent.
- **State**: FFP via batched CGxStateBom stack; `.bls` = texture combine only; viewer just needs the
  state SET per batch. Build order (doc §6): lighting P0/P1 → M2 blend/2-tex/sort → animated water →
  particles/ribbons.

## World renderer reality (lighting/shadow/fog) + M2 camera tracks (2026-07-15)

- **Why**: user wants the viewer's renderer grounded in how 1.0.0 actually renders (lighting,
  shadows, fog) + wants **M2/MDX camera tracks** playable like taxi routes. Two new guides:
- **`docs/architecture/wow-1.0.0-world-lighting-shadow-model-2026-07-15.md`** — 1.0.0 is a
  **fixed-function** renderer: **1 directional (sun) + ambient**, Lambert `N·L` (SetLight/PLight
  cvars, glLight*/glFog*/glColorMaterial imports); `.bls`+combiners = texture combine only.
  Day/night (`DayNight.cpp` LightData/Fog/Sky) drives sun/ambient/fog/sky by time. Terrain
  (`FUN_006b4920`): MCVT/MCNR/MCLY/MCAL/**MCSH**(1-bit baked shadow)/MCLQ — **no MCCV**, so lit
  dynamically from normals × MCSH. Shadows: terrain baked MCSH + unit **ShadowBlob.blp**/projected
  (fixed ~45° sun). Fog FFP, interior/exterior selected in `FUN_0067c460`. **P0-P3 gap analysis**
  in doc §8 (P0 = normal-lit terrain + day/night sun; P1 = MCSH shadows + fog + WMO interior light).
- **`docs/architecture/wow-1.0.0-m2-camera-tracks-2026-07-15.md`** + evidence `m2_camera.c` —
  **M2Camera 0x7c** @ header 0x124, cameraLookup int16[] @0x12C. Layout: type@0, fov@4 (diagonal rad),
  far@8, near@0xc, positions(Vec3 spline)@0x10, positionBase@0x2c, targetPosition(Vec3)@0x38,
  targetBase@0x54, roll(float)@0x60. **GOTCHA: 1.0.0 uses OLD M2Track 0x1c** (has interpRanges@+0x04;
  Wrath+ drops it → 0x14). Runtime instance 0x84 @ model+0x398 (+0x80=src ptr); accessors
  `FUN_0070edc0/ee30/eeb0`. Eval = sample eye+target (SEPARATE splines)+roll+static fov → lookAt+persp;
  reuse taxi/anim sampler. Only new parsing = 0x7c record + old-0x1c track. Consumed by portraits/
  model-view widgets/cinematic system (drives CGCamera). Checklist in doc §7. FOV aspect = calibrate.

## Dropped / paused

- **V24 / Spec 094 is NOT functional — dropped.** Do not revive it.
- **Spec 102 M0 object-mask lane is paused/superseded** by Spec 103. Preserved: simple M0 trainer
  (`train_spec102_m0_simple.py`) + inference; strict fragment-trace target + 42/42-green tests remain inactive.

## Boundaries

- New work in `wow-viewer/`; `gillijimproject_refactor` is read-only reference (port from, never edit).
- **Client data (UPDATED 2026-07-15, Constitution v1.1.0):** `output/tmp/wowarchive-clients/` remains
  the **default for automated validation/harvesting**. `H:\CLIENTS` is **no longer prohibited** — it
  is a curated temporary SSD staging area fed from **WoWArchive (~150 GB, cold HDD)**, the
  authoritative corpus. The old "forbidden legacy client root" ban is **retired** (it targeted
  broken clients of unknown origin that the USER has since removed). The surviving rule: **never
  hardcode a client root** — it is configuration. Static RE from a staged binary is permitted; cite
  the build.
- Spec 080 owns the UI lane.

## Spec 103 — map-canvas prefab curation + synthetic lighting (2026-07-15)

- **Current target:** shrink the V8 corpus by unique terrain-art prefab/context coverage before the
  next training run, while making synthetic RGB reproduce the recoverable 1.x terrain-lighting
  contract. The proof owner is Spec 103; alpha/paste discovery remains owned by Spec 076.
- **Curation implemented:** `spec103_curate_prefabs.py` consumes complete-map Spec 076 canvases and
  regions, emits typed pattern/tile ledgers plus a reduced manifest, treats ADTs as storage pages,
  retains atomic/composite/blocky/non-brush states, derives multiscale/cellular placement features,
  records MCLY/tileset anomalies and object/liquid context, selects deterministic representatives,
  and prevents canonical prefab families from crossing partitions.
- **Lighting implemented:** shared strict LIT reader/profile export; all headers before all groups;
  BGRX timed colors; unique global/default clear selection; local LIT placement remains disabled.
  Sky geometry is Z-up. Terrain now carries MCNR and optional full MCSH through both render paths,
  uses one-sided Lambert, and modulates directional light only. Capture is one-tile top-down
  orthographic with hash-bound v2 sidecars.
- **DBC truth:** exact-build DBCD + bundled WoWDBDefs resolves `Light`, `LightParams`,
  `LightIntBand`, `LightFloatBand`, and `LightSkybox`, retaining DBC/DBD hashes and record IDs.
  `GameCoords` is fixed-scale X/Z/Y; the active `LightService` now converts `/36` correctly and
  uses this chain before any later flattened compatibility path.
- **Rights boundary:** licensed/generated authored variants can be `clean_synthetic`; any captured
  client LIT/DBC color is `private_byod`. No renderer or model output is presented as a legal status.
- **Verification:** 37/37 Spec 103 Python tests and 35/35 focused Core lighting/runtime tests pass;
  Capture, active WoWViewer, and Inspect projects build with zero errors. Existing Snappier advisory
  warnings remain outside this slice.
- **Unproven/user-run:** T030 bounded real-corpus curation report; T040 staged LIT/DBC image
  comparison and direction/MCSH/sky calibration; all capture, harvest, GPU, and training runs.
