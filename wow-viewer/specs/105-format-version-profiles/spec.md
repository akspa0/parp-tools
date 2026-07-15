# Feature Specification: 1.0.0 M2 Renders Correctly (on a Unified Version-Profile Foundation)

**Feature Branch**: `105-format-version-profiles`

**Created**: 2026-07-15 · **Scope expanded**: 2026-07-15 (see "Scope Expansion" below)

**Status**: Draft

**Input**: User description: "read memory bank files, we are trying to fix up the m2 rendering work for 1.0.0+ client data, we need you to use ghidra's mcp server to resolve the issue we have with interop ranges that 1.0.0's m2 format has and 3.x+ does not. We ultimately have to build adapters for the warcraft.net m2 code that we use for 3.x. I think that's what we've been doing, per-version of the format, but it's not clear and we should probably use speckit to plan for a better way to handle the versions of each format, so they don't all step on each other's toes. I'm not sure how much of our m2 code is just wrappers for warcraft.net or is actual code that is our own. the existing 3.x m2 code works well, even works right with 4.0.0 data too."

**Follow-up (screenshot evidence, 2026-07-15)**: "M2's are still flat shaded and only a single material and not animated." A 1.0.0 CentaurKhan renders as a uniform gray, unlit-looking, static blob with the frame counter advancing (2081/3333) against a frozen pose.

## Scope Expansion (2026-07-15)

The original spec addressed **animation only** — the version-profile architecture proven on the 1.0.0
interpolation-range gap. A live-viewer screenshot showed that fixing animation alone leaves a 1.0.0
model **gray and flat**: three distinct rendering defects, not one. The user chose to make this **one
combined spec** covering all three. The feature is now "make a 1.0.0 M2 render correctly," resting on
the version-profile foundation the animation pillar needs.

**The folder is still named `105-format-version-profiles`** — the version-profile architecture remains
the structural spine, but the deliverable is now a correctly rendered 1.0.0 model. The name is a
cosmetic legacy, not a scoping signal.

### The three pillars and their verified causes

| Pillar | Symptom | Root cause | Confidence |
|---|---|---|---|
| **P-Texture** | uniform gray, "single material" | batch→textureCombo→texture **resolution** yields no bindable path. Reader parses filenames fine (`M2Era100ModelReader.cs:302`); renderer has a full load+replaceable path (`M2Renderer.cs:1023`). So it is neither a parse nor a render-support gap. | narrowed to the resolution stage; exact cause (combo mis-index vs BLP-not-found vs replaceable-with-no-skin) needs a per-model trace |
| **P-Lighting** | "flat shaded" | Lambert **is** applied (`M2Renderer.cs:854`, gated on `Material.IsUnshaded` at `:697`) and the screenshot shows a visible N·L gradient. The flat *appearance* is dominated by the missing texture. A separate defect (material flag `0x01`=UNLIT mis-mapped to `IsUnshaded`) is possible but unconfirmed. | lighting runs; "flat" is mostly a texture artifact — confirm after P-Texture lands |
| **P-Animation** | frozen pose, frame counter advances | bones dropped by **both** readers: `M2Era100ModelReader.cs:118` (`bones: null`) and `M2Era1121ModelReader.cs:95` (`bones: []`). Frame advances against zero tracks. | **confirmed** |

**Correction recorded in the same session**: an earlier claim that the misread sequence duration
("3333") is what freezes the pose was wrong. The freeze is `bones: null/[]`, full stop. The duration
misread (research R1) is real but only corrupts the *displayed timeline number*, not the pose.

## Context: Two Findings That Reframe the Request

Both were verified in code during the 2026-07-15 session. They are recorded here because they invalidate parts of the original framing.

**Finding 1 — the Warcraft.NET premise is inverted.** The M2 readers in `WowViewer.Core`, `WowViewer.Core.IO`, and `WowViewer.Core.Runtime` (~5,000 lines) are entirely our own code. Warcraft.NET is not referenced in those project files at all. The native render path (`WowViewerM2RuntimeBridge` → `M2StaticRenderModel` → `M2Renderer`) is 100% ours, including the 3.x/4.x path the user reports "works well." Warcraft.NET's M2 support targets Legion+ chunked `MD21`; its only consumer is `WarcraftNetM2Adapter.cs` (viewer, 3,070 lines), a legacy M2→`MdxFile` fallback for the old `MdxRenderer`. **There is nothing to build Warcraft.NET adapters for.** That work is not in this spec.

**Finding 2 — the "stepping on toes" is two competing versioning schemes, and one is already a profile registry.** `FormatProfileRegistry` is a 717-line prior attempt at the architecture this spec is about. It coexists with, and never speaks to, the dispatcher that actually parses:

| | `FormatProfileRegistry` | `M2ModelReaderDispatcher` |
|---|---|---|
| Location | `src/viewer/WoWViewer/Terrain/` | `src/core/WowViewer.Core.IO/M2Chunked/` |
| Keyed on | build version string (`"3.3.5.12340"`) | the file's own version field (`0x100`, `0x108`) |
| Drives | validation only | actual parsing |
| Covers 1.x? | no — `ResolveModelProfile` returns `null` | yes |

Core cannot see the registry (`WowViewer.Core.IO` does not reference the viewer, and per Constitution I it must not). This violates Constitution II: *"One canonical owner per format surface."*

The M2 half of the registry is additionally **inert**: all five `M2Profile` instances carry identical strides (`SkinLikeAStride=0x70`, `SkinLikeBStride=0x2C`, `EffectLikeAStride=0xD4`, `EffectLikeBStride=0x7C`), and the `LikeA`/`LikeB` naming records that the semantics were guessed. It is ceremony that varies nothing. The `AdtProfile`/`WmoProfile`/`MdxProfile` halves of the same file *do* carry real, varying, load-bearing values (e.g. `MclqLayerStride` `0x2D4` vs `0x324`) — the registry is not uniformly worthless, and this distinction governs the scope question below.

**The success bar follows from this: reconcile the two schemes. A third scheme is a failure, not a delivery.**

## User Scenarios & Testing *(mandatory)*

The three rendering pillars (P1) are what the user sees; the two architecture stories (P2/P3) are the
foundation the animation pillar rests on. Sequenced **texture → lighting → animation**: the most
visible defect first, and a textured model is the prerequisite for judging whether lighting and
animation are actually right.

### User Story 1 - A 1.0.0 model shows its real textures (Priority: P1)

A 1.0.0 creature or character (e.g. CentaurKhan) loaded from a staged 1.0.0 client renders with its
actual skin texture applied, instead of a uniform gray surface.

**Why this priority**: It is the most visually dominant defect — the model reads as a featureless gray
blob — and until it is fixed, neither lighting nor animation can be judged (a gray model hides both).
It is sequenced first for that reason.

**Independent Test**: Load CentaurKhan in the viewer; confirm the body shows texture detail, not flat
gray. Delivers a recognizably textured model even if no other story ships.

**Acceptance Scenarios**:

1. **Given** a 1.0.0 model whose batch references a **hardcoded** (Type-0) texture, **When** it renders, **Then** that texture's BLP is resolved from the staged client and bound, and the surface shows texture detail.
2. **Given** a 1.0.0 model whose texture combo entry is **negative** (a replaceable creature/character skin), **When** no runtime skin is supplied, **Then** the batch renders with an explicit resolved-or-placeholder texture and the failure is logged — it does not silently collapse every batch onto one gray fallback.
3. **Given** a batch that resolves to more than one texture stage, **When** it renders, **Then** each stage binds independently — the model is not reduced to a single material.
4. **Given** a 3.x or 4.x model, **When** it renders, **Then** its texture binding is unchanged from the current baseline.

---

### User Story 2 - A 1.0.0 model animates (Priority: P1)

A 1.0.0-era model (e.g. TrollMale, DoomGuard, CentaurKhan) plays its animation sequences with visible
skeletal motion, instead of standing frozen while the frame counter advances (observed: 2081/3333,
static).

This is the architecture proving slice: 1.0.0 is the one era whose animation format **cannot be
expressed at all** by the current shared contract, so it is the honest test of whether the profile
system does real work.

**Why this priority**: It is the case that forces the version architecture (P2/P3) to be correct rather
than decorative, and it is a headline user complaint. Sequenced after texture because a moving gray
blob is weaker evidence than a textured static one.

**Independent Test**: Load a named 1.0.0 model, play a sequence, observe bone motion.

**Acceptance Scenarios**:

1. **Given** a 1.0.0 M2 whose tracks carry a non-empty interpolation-range array, **When** a sequence is sampled at a time inside that animation's range, **Then** the sampled value derives only from keys within that animation's inclusive range — never from a neighbouring animation's keys.
2. **Given** a 1.0.0 M2 track whose interpolation-range array is empty, **When** the track is sampled, **Then** the whole flat key array is treated as the range.
3. **Given** a 1.0.0 M2 track bound to a global sequence, **When** it is sampled, **Then** the global timer supplies the time and the caller's animation time is ignored.
4. **Given** a 1.0.0 model, **When** it is loaded, **Then** it reaches the renderer carrying bone tracks rather than none, and its sequence durations are plausible animation lengths (not global-timeline offsets).
5. **Given** a 3.x or 4.x M2, **When** it is sampled through the same shared sampler, **Then** its sampled values are unchanged from the current baseline.

---

### User Story 3 - A 1.0.0 model is correctly lit (Priority: P2)

A textured 1.0.0 model shows correct directional shading — neither uniformly fullbright nor uniformly
dark — matching how the 1.0.0 client lights M2s.

**Why this priority**: Lambert already runs (a gradient is visible in the current build), so this is
largely a *confirmation-and-correction* pillar, not a build-from-scratch one. It is P2 because the
visible "flat" appearance is mostly the missing texture (P1), and this pillar can only be judged once
P1 lands.

**Independent Test**: With P1 done, compare a lit 1.0.0 model against the client's shading; confirm no
batch is wrongly forced fullbright.

**Acceptance Scenarios**:

1. **Given** a 1.0.0 material whose flags do **not** set the UNLIT bit (`0x01`), **When** it renders, **Then** it is directionally lit (N·L), not fullbright.
2. **Given** a 1.0.0 material whose flags **do** set UNLIT, **When** it renders, **Then** it is fullbright — matching the client.
3. **Given** the material-flag→`IsUnshaded` mapping, **When** audited against the Ghidra evidence, **Then** it reflects flag bit `0x01`, not a guess or a hardcoded constant.

---

### User Story 4 - One canonical owner for format version decisions (Priority: P2)

A developer adding support for a new client build declares that build's layout facts in exactly one place, and both parsing and validation consume that declaration. Nobody has to know that two registries exist, or which one is authoritative.

**Why this priority**: This is the user's actual architectural ask, and it is the Constitution II violation. It is P2 rather than P1 because it has no user-visible output on its own — it is proven by the animation pillar (US2) riding on it.

**Independent Test**: Grep for the number of distinct places a version→layout decision is made. Before: two, unreconciled. After: one. Verified by the inert M2 half of the viewer registry being deleted, not wrapped.

**Acceptance Scenarios**:

1. **Given** the codebase after this change, **When** a version→layout decision is located, **Then** exactly one component owns it, and it lives in the core libraries where the readers can reach it.
2. **Given** the inert `M2Profile` records, **When** this story completes, **Then** they are removed rather than migrated, because they encode no varying information.
3. **Given** a layout constant in the profile system, **When** it is read, **Then** it either cites its evidence (Ghidra address, runtime trace, or wiki reference) or is explicitly marked provisional.
4. **Given** the total code devoted to version dispatch and profiles, **When** measured against the pre-change baseline, **Then** it has decreased.

---

### User Story 5 - Era resolution stops guessing (Priority: P3)

A 1.0.0 M2 and a 1.12.1 M2 — which report the identical version field `0x100` but have different header layouts — are each routed to the correct reader by a deterministic decision, not by attempting one layout and catching failure.

**Why this priority**: The current trial-parse heuristic works today but is a latent correctness hazard: it silently misroutes any file whose wrong-layout parse happens to produce plausible offsets. It is P3 because it is not currently causing a known user-visible failure. It is in scope because a profile system that still guesses has not solved the problem.

**Independent Test**: Feed known 1.0.0 and 1.12.1 models to era resolution and assert correct routing with the fallback path disabled.

**Acceptance Scenarios**:

1. **Given** a 1.0.0 M2 and a 1.12.1 M2, **When** each is resolved, **Then** each routes to its correct reader without a failed parse attempt occurring first.
2. **Given** a file that matches no known profile, **When** it is resolved, **Then** it fails loudly and names what was unrecognized, rather than falling through to a different era's layout.

---

### Edge Cases

- **Key index runs past the animation's range.** The 1.0.0 client's final clamp tests the *total* key count, not the animation's end index. When the bracketing key is the last of a range and more keys follow, the client interpolates across the boundary into the next animation's first key. This is reproduced deliberately, not "fixed" — see FR-004. On well-formed data it is unreachable (time is clamped to the sequence duration, and the last key sits at the duration, yielding a zero interpolation factor).
- **An animation index addresses a range that does not exist** (index beyond the interpolation-range array).
- **A range is degenerate** (end ≤ start): a single key, no interpolation.
- **A track's key and value arrays disagree in length.**
- **A model reports a known version but its layout does not validate** — must fail loudly (US5 scenario 2), never silently reinterpret.
- **A creature's texture combo resolves to a negative (replaceable) index when it should be a hardcoded texture** — the P-Texture failure mode to watch: a mis-indexed combo turns a direct texture into a replaceable lookup with no runtime skin, yielding gray.
- **A referenced texture BLP is absent from the staged client** — must render a visible placeholder and log, not a silent gray that is indistinguishable from a resolution bug.
- **A build is staged that no profile names.** Whether this fails or falls back to a nearest profile interacts with the user's separate, queued direction on DBC schema fallback ("use the last version that fits"). Called out in Assumptions.

## Requirements *(mandatory)*

### Functional Requirements

**Texture resolution (P-Texture — sequenced first)**

- **FR-T1**: A 1.0.0 batch that references a hardcoded (Type-0) texture MUST resolve that texture's filename to a BLP from the staged client and bind it. The current gray outcome MUST be root-caused first: the fix depends on whether the failure is combo mis-indexing, BLP-not-found, or replaceable-with-no-skin, and the diagnosis MUST precede the fix (guessing caused prior bugs).
- **FR-T2**: A negative texture-combo entry MUST be treated as a replaceable skin id, resolved through the replaceable-texture path. When no runtime skin is available, the batch MUST render an explicit placeholder and log the unresolved id — it MUST NOT collapse every batch onto one shared gray fallback.
- **FR-T3**: A batch with multiple texture stages MUST bind each stage independently. A model MUST NOT be reduced to a single material because only the first (or a fallback) stage bound.
- **FR-T4**: A referenced BLP absent from the staged client MUST produce a visible placeholder and a log line, distinguishable from a resolution bug (FR-013's fail-loudly principle, applied to textures).

**Lighting (P-Lighting — confirm then correct)**

- **FR-L1**: The material-flag→`IsUnshaded` mapping MUST reflect the 1.0.0 material UNLIT bit (`0x01`) per the Ghidra evidence, not a hardcoded constant or a guess. It MUST be audited, since the material array is read but the flag semantics were previously heuristic.
- **FR-L2**: A 1.0.0 material without the UNLIT bit MUST render directionally lit (N·L); one with it MUST render fullbright — matching the client. Lambert already runs; this pillar is confirmation and correction, not new lighting.
- **FR-L3**: P-Lighting MUST be judged only after P-Texture lands, because the "flat" appearance is dominated by the missing texture and cannot be assessed on a gray model.

**Animation track addressing (the architecture proving slice)**

- **FR-001**: The animation track contract MUST be able to express both addressing modes: the 1.0.0 form (one flat key array per track, sliced per animation by an inclusive interpolation range) and the Wrath+ form (key arrays nested per sequence, no interpolation ranges). These are different addressing modes, not different strides — no offset adjustment can bridge them.
- **FR-002**: The sampler MUST select the addressing mode from the model's resolved profile, never by inspecting bytes at sample time.
- **FR-003**: An empty interpolation-range array MUST mean "the whole flat array is the range."
- **FR-004**: Sampling MUST reproduce the observed client behaviour recorded in the Ghidra evidence, including the total-count clamp described in Edge Cases. Any deviation MUST be justified by a reproduction case, not by an assumption that the client is wrong.
- **FR-005**: A track bound to a global sequence MUST take its time from the global timer, overriding the supplied animation time.
- **FR-005a**: The **sequence time base MUST be era-dependent.** 1.0.0 sequences carry a `start`/`end` pair describing a span of a **global timeline** shared by all animations, and the time handed to the sampler is absolute (`start + (elapsed mod (end - start))`). Wrath+ sequences instead carry a single `duration` and each sequence's keys live in their own array, making time sequence-local (`elapsed mod duration`). Our sampler currently implements only the Wrath rule, and `M2SequenceDefinition` models only `duration` — it cannot represent 1.0.0's `start`/`end` at all. This is a **second expressiveness gap of the same class as FR-001**, discovered while verifying the caller; it is why the flat key array and interpolation ranges exist in the first place. Sampling 1.0.0 with the Wrath time rule yields wrong keys even with FR-001 implemented.
- **FR-006**: The sampler MUST NOT carry per-track mutable state. The client's cached-key optimization is a performance device whose three search branches provably converge on the same key; a stateless bracketing search is equivalent. (Verified this session — see Assumptions.)
- **FR-007**: 1.0.0 bone tracks MUST NOT be populated until the flat-range sampling path is in place and passing. Wrong poses are worse than no poses, and every bug fixed in this area to date was caused by a guessed layout.

**The profile system**

- **FR-008**: Exactly one component MUST own version→layout decisions, and it MUST live where the core readers can consume it (Constitution I forbids core referencing the viewer; Constitution II requires one canonical owner per format surface).
- **FR-009**: The system MUST NOT introduce a third scheme. The two existing schemes MUST be reconciled: one absorbed into the other, with the redundant one deleted.
- **FR-010**: The inert `M2Profile` records MUST be deleted rather than migrated. They encode no varying information; carrying them forward would preserve the ceremony this spec exists to remove.
- **FR-011**: Every layout fact in the profile system MUST cite its evidence or be explicitly marked provisional. Provisional facts MUST be distinguishable from verified ones at the point of use.
- **FR-012**: Era resolution MUST be deterministic. Trial-parse-and-catch MUST NOT be the routing mechanism.
- **FR-013**: An unrecognized model MUST fail loudly, naming what was unrecognized. It MUST NOT fall through to another era's layout.
- **FR-014**: Adding a new build's support MUST require declaring its facts in one place, without editing dispatch logic.

**Not regressing what works**

- **FR-015**: The 3.x/4.x M2 path MUST be behaviourally unchanged. This is non-negotiable: it is the path the user reports working (including 4.0.0 data), and the shared contract types this spec changes are the ones it uses.
- **FR-016**: A no-regression gate MUST compare 3.x/4.x output against a pre-change baseline captured before any shared type is touched.
- **FR-017**: `WarcraftNetM2Adapter` and the legacy `MdxRenderer` fallback MUST remain untouched (user decision, this session). Recorded as tech debt; not addressed here.

**Format surface scope**

- **FR-018**: The profile system MUST cover the M2 format surface, and only the M2 format surface (user decision, this session). The ADT, WMO, and MDX surfaces MUST continue to use `FormatProfileRegistry` unchanged. Rationale: the M2 half of that registry is inert and can be deleted outright, whereas the ADT/WMO halves carry load-bearing varying constants inside the Constitution's declared Terrain Alpha Risk Area — materially different risk, and none of it is what is broken.
- **FR-019**: The profile system MUST be shaped so the ADT, WMO, and MDX surfaces can migrate later without redesigning it. "Shaped so they can migrate" means the design MUST NOT encode M2-only assumptions into the profile contract itself (e.g. it must not assume a single root magic, a single version field location, or that a version field alone is sufficient to identify a layout).
- **FR-020**: This feature MUST NOT be considered a licence to migrate the other surfaces opportunistically. A migration of ADT, WMO, or MDX requires its own spec and its own regression gate.

### Key Entities

- **Format profile**: The declared, evidence-cited layout facts for one format at one version or build range. Replaces both today's build-string-keyed viewer records and today's implicit per-era reader knowledge.
- **Era / version identity**: The resolved answer to "which layout is this file?" — currently derived from the file's version field plus a trial parse; must become a deterministic resolution, possibly needing build context that core cannot reach today.
- **Animation track**: A timed key sequence. Its addressing mode is a per-era property (flat + interpolation ranges vs nested per sequence), and today's contract can express only one of the two.
- **Interpolation range**: A `{first, last}` inclusive key-index pair, indexed by animation index, that slices a 1.0.0 flat key array into per-animation spans. Has no Wrath+ equivalent. Absent from the codebase today.
- **Evidence citation**: The provenance of a layout fact (Ghidra address, runtime trace, wiki reference) or an explicit provisional marker.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-000 (headline)**: A named 1.0.0 model (CentaurKhan) renders **textured, correctly lit, and animated** in the viewer — user-confirmed against a staged 1.0.0 client. This is the combined-spec success condition; SC-001/SC-T1/SC-L1 are its three components.
- **SC-T1**: A 1.0.0 model shows its real texture — texture detail visible, not a uniform gray surface; multi-stage batches bind more than one material. User-confirmed.
- **SC-L1**: A textured 1.0.0 model is directionally lit where its materials are not UNLIT, and fullbright only where they are — no batch wrongly forced to a single lighting mode. User-confirmed.
- **SC-001**: A named 1.0.0 model plays a named sequence with visibly moving bones, confirmed by the user against a staged 1.0.0 client. The reported "frame 2081/3333 but static" symptom is gone.
- **SC-002**: 3.x/4.x models produce output identical to the pre-change baseline — zero diffs across the named regression set, including 4.0.0 data.
- **SC-003**: Exactly one component answers "which layout does this M2 use?" — today two do, and they never communicate. Measured per format surface, not repo-wide: `FormatProfileRegistry` deliberately survives as the ADT/WMO/MDX owner (FR-018), so its continued existence is not a failure of this criterion.
- **SC-004**: Total code devoted to **M2** version dispatch and M2 profiles is **lower** than the pre-change baseline. Growth is a failure signal, regardless of design quality. (Deleting the inert `M2Profile` records alone removes roughly 80 lines; the bar is that the new system costs less than what it replaces.)
- **SC-005**: Zero trial-parse fallbacks remain in era resolution.
- **SC-006**: 100% of layout facts in the profile system either cite evidence or are marked provisional; a reader can tell which at the point of use.
- **SC-007**: Supporting a hypothetical new build requires edits in one place only, demonstrated on a real build.

## Assumptions

- **The 1.0.0 track and bone layouts are Ghidra-verified, not assumed.** The sampler algorithm (`FUN_0070f6d0`) and bone layout (`0x6C`, from relocator `FUN_0071f440`) were recovered in a prior session and **re-verified against WoW.exe 1.0.0.3980 during this session**; the prose was confirmed accurate on every point. Raw evidence: `specs/104-legacy-m2-rendering/evidence/1.0.0-ghidra/m2_track_sampler.c`. Newly established this session: the client's `out[0]` is an in/out cached-key seed selecting among forward-scan / binary-search / backward-scan; it is purely a performance cache and all branches converge on the same key, which is what licenses FR-006.
- **The animation index IS our sequence index — risk retired this session.** Traced `FUN_0070f960` (2,082 lines, 58 call sites into the sampler): the value passed as the interpolation-range index is the per-bone current animation id, which the same function uses to index the sequence array. So interpolation ranges run in lockstep with sequences — `interpRanges[i]` describes the key span of `sequences[i]`, and our existing `sequenceIndex` is the correct value. No alias translation sits at this layer. The caller trace also independently re-confirmed the bone stride (`0x6C`), `parentBone` at `+0x08`, and the rotation/scale track offsets, from a second site.
- **Two behaviours found in the client that we deliberately do NOT model** (noted rather than flattened away): 1.0.0 animation state is **per bone** — each bone carries its own `(time, animIndex)`, and `-1` means "inherit the parent bone's" — and a **second** `(time, animIndex)` pair drives cross-animation **blending** via a second sampler call. Our runtime models one sequence per model. Both are out of scope for a first pass; neither blocks the animation pillar (US2).
- Format facts for eras other than 1.0.0 are assumed to be captured accurately by the existing readers, since those readers work. This spec does not re-derive them.
- The 1.0.0 vs 1.12.1 disambiguation may require build context that core cannot currently reach. How that context arrives without violating Constitution I (core must not reference the viewer) is a design question for the plan, not a spec decision.
- Unknown-build fallback behaviour is assumed to be **fail loudly** for models (FR-013), consistent with Spec 104's "fail loudly rather than fall through." This deliberately differs from the user's queued DBC direction ("use the last version that fits"), because a wrong DBC schema yields visibly wrong data while a wrong model layout yields plausible-looking garbage. If the user wants these unified, it changes FR-013.
- All validation uses staged clients under `output/tmp/wowarchive-clients/` (Constitution III, VI).
- Per AGENTS Rule 0, the agent prepares commands; the **user** runs render validation and owns visual signoff. SC-001 cannot be self-certified.
- Phases follow the Constitution's "One Phase at a Time" and "Bite-Sized Plans" (max 10 steps per phase, one concern per step).

## Out of Scope

- `WarcraftNetM2Adapter` and the legacy `MdxRenderer` fallback path (user decision; tech debt noted).
- Building Warcraft.NET adapters — the premise is inverted; see Finding 1.
- Retiring the Warcraft.NET M2 dependency.
- Particle, ribbon, camera, and light track wiring for 1.0.0 (they share the track contract and will benefit, but are not proven here).
- Pre-`0x100` eras (0.11/0.12), which already work through a different path.
- **Non-M2 rendering (terrain, WMO, liquids).** The three pillars are M2-only.
- **Character customization / equipment textures.** P-Texture covers hardcoded creature textures and the replaceable-skin *fallback* path; full character skin/hair/armor composition is a separate, larger concern.
- **Full material/effect parity** (multi-pass combiners, animated UV/color/alpha as they affect shading). P-Texture binds textures and P-Lighting fixes the lit/unlit split; it does not claim shader-combiner fidelity with the 1.0.0 client.
