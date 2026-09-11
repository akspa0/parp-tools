# Feature Specification: Legacy MDX & M2 Rendering Correctness (1.0.0 Through 3.0.1) & Fuckported-Asset Compatibility

**Feature Branch**: `235-legacy-mdx-m2-rendering`

**Created**: 2026-09-10

**Status**: Draft — authored from operator directive plus reconciliation of prior specs; not planned

**Input**: Operator directive 2026-09-10 (verbatim intent): "our mdx and m2 understanding for
version 1.0.1 client data through 3.0.1, is very much non-functional, also. I thought that 1.0.0
used the same version of mdx, but I was wrong. Our MDX support up to the introduction of the M2
format really needs to be looked at and solved at some point. use speckit to plan for this
properly. We have no objects rendered, and no bounding boxes either, most of the time, for these
format files. Later format files that have had their chunks rewritten using non-standard means of
'fixing', also do not render properly (the so-called 'fuckported' assets) - we need to ensure that
everything that warcraft.net can natively read, is renderable. We also are missing visual effects
on torches and light-bearing objects in some cases in the mdx format."

## Supersession notice (found during authoring, 2026-09-10)

Before planning could start, checking `specs/epics/active-epics.md` Epic 4 and
`specs/registry.md` surfaced **two existing, unowned, substantially-overlapping specs that were
not showing up in default routing**:

- **Spec 104** (`104-legacy-m2-rendering`, "Legacy M2 model rendering, client 1.0.0–2.4.3") — has a
  full kit (plan, tasks, research, Ghidra trace notes, data model, contracts). **7 of 27 tasks
  checked.** Its own header claims `Status: Active`, but it is absent from every named bucket in
  `specs/registry.md` (not "current owner," "near-term planned," or "queued active") — it has been
  functionally cold despite the "Active" label. Its embedded-skin-profile root-cause finding
  (hardcoded to zero) and version-priority rollout order (well-documented versions first, then
  mid-range, then early alphas via dynamic tracing) are real, credible, and reused below.
- **Spec 154** (`154-m2-era-reader-parity`, "M2 Reader Era Parity 1.x–3.0.1") — has spec, plan,
  research, data-model, contracts, quickstart, but **no `tasks.md`** (planned, never task-broken).
  Its scope is nearly identical to this feature's core ask and its evidence is **more precise**
  than anything available when this spec was first drafted: a real build-by-build survey found the
  broken range is **exactly `0x100` through `0x107`** (not an approximate "1.x–3.0.1"), that
  `3.0.1` declares `0x107` while `3.3.0` declares `0x108` (the real known-good boundary — not
  "3.3.5"), and three named defects (bones discarded for the `0x100` route; a 20-byte-per-bone
  layout drift in the fallback path; an **unhandled** crash reading camera records on a 4.0.0 beta
  that contradicts the assumption "4.x already works"). It also proved that `0x100` alone does not
  identify the layout — a `2.0.0.5610` pre-release *also* declares `0x100`, distinct from both
  1.0.0 and 1.12.1.
- **Spec 105** (`105-format-version-profiles`, the 1.0.0-specific texture/animation/lighting
  pillar cited in this spec's original draft) remains valid, narrower prior art — it went deep on
  **one** model (CentaurKhan) on **one** build (1.0.0) rather than surveying the range, and
  explicitly deferred particle/ribbon/light-track wiring. Not superseded, just narrower.
- **Spec 193** (`193-benilla-112-client-reference`) documents an available external oracle — a
  modern, actively-maintained Rust 1.12.1 client (`samwhosung/benilla`) — as a second reference
  implementation for 1.x M2 parsing, alongside Warcraft.NET named in this feature's own input.
- **Spec 136** (M2 doodad rendering performance) is a genuinely separate, non-overlapping concern
  (FPS/batching, not correctness) and is not touched by this spec.

**This spec supersedes the unimplemented residue of Spec 104 and Spec 154** (consistent with this
project's standing rule: the newest spec supersedes older unimplemented ones — see
`specs/STATUS.md` "Status rules"). It inherits and reconciles their measured findings rather than
re-deriving the range from scratch, and adds the genuinely new scope from today's operator
directive that neither covered: the `FormatProfileRegistry` MDX-side resolver bug, "fuckported"
asset / Warcraft.NET parity, and MDX torch/light-emitter effects. Dated amendment notes pointing
here have been added to the top of `104/spec.md` and `154/spec.md`; neither directory's content is
deleted or archived — both remain readable prior art per this project's "archive, don't destroy"
discipline. **Spec 104's own 7 checked tasks have not been receipt-audited as part of this spec's
authoring** — that audit is planning work, not something to assume passed or failed here.

## Context

Grounding evidence, consolidated from this spec's own research plus specs 104/105/154:

- **The measured broken range (from Spec 154, real staged clients, 2026-08-15) is exactly `MD20`
  header version `0x100` through `0x107`.** `3.0.1` declares `0x107`; `3.3.0.10958` declares
  `0x108` and reads cleanly (151 bones, 155 sequences, geometry, 375 bone lookups) — `0x108` is the
  real known-good reference point, not "3.3.5 through 4.0.0." A `4.0.0.11927` beta *also* fails,
  with an **unhandled** crash reading camera records — contradicting any assumption that "4.x
  already works." The version word alone does not identify the layout: `0x100` covers at least
  three mutually incompatible cases (1.0.0, 1.12.1, and a `2.0.0.5610` pre-release), each requiring
  evidence-based disambiguation, never inference from another build's version word.
- **Two named, measured defects in the currently-broken range** (Spec 154 D1/D2): bone data is
  discarded entirely for the `0x100`-era route (reports zero bones); where a fallback path runs, it
  reads bones at the wrong stride (88 bytes, borrowed from the late-3.x reader) against the
  `0x100` era's own already-recorded 108-byte stride — a 20-byte drift that corrupts geometry
  starting at bone index 10. The correct layout was already recorded in the codebase and never
  connected to a bone parser.
- **The embedded-skin-profile root cause** (Spec 104): for every in-scope version (M2 format
  version ≤ 263, i.e. up through the `0x107`/2.4.3 boundary), skin profiles (submesh definitions,
  triangle indices, texture-unit bindings) are stored **inside** the `.m2` file itself
  (`nViews`/`ofsViews`), not externalized to `.skin` files (that only starts at WotLK, version
  264+). The current reader hardcodes the embedded view/skin count and offset to zero for these
  versions, which is the direct mechanism behind "no objects rendered" — there is no mesh data to
  render because it was never read, not because rendering itself is broken.
- **`FormatProfileRegistry.ResolveMdxProfile`/`ResolveModelProfile`**
  ([FormatProfileRegistry.cs](../../src/viewer/WoWViewer/Terrain/FormatProfileRegistry.cs)) has a
  related but distinct bug this spec adds to the picture: `ResolveMdxProfile` has no case for any
  build with major version `>= 1` and silently falls back to the `0.6.0`/`0.7.0` MDX profile;
  `ResolveModelProfile` (M2) has no case for `major == 1` at all and returns `null` for the entire
  Vanilla retail era. **Whether this registry is actually in the load path Spec 104/154's readers
  use, or a separate/legacy resolution mechanism that needs reconciling with `M2ModelReader100`
  (named in Spec 193) and whatever era-layout selection Spec 154's research already built, is an
  open research question for planning — do not assume either finding is "the" root cause of the
  other without checking.**
- **`docs/architecture/m2/consumer-cutover.md`** confirms the light/particle/ribbon gap is a named,
  tracked architectural absence ("shader backend wiring, particle/ribbon parser and simulation...
  remain future proof levels"), and names the vendored **Warcraft.NET** library
  (`libs/ModernWoWTools/Warcraft.NET`, `WarcraftNetM2Adapter.cs`) as an existing parity reference —
  the natural anchor for "everything warcraft.net can natively read must be renderable." Spec 193's
  Benilla reference is a second, independent oracle specifically for the 1.x era.
- **"Fuckported"-asset prior art exists for WMO** (`WmoV17ToV14Converter.cs`, two copies, no
  real-data validation). For M2/MDX there are **two existing, likely-drifted** converters —
  `WowViewer.Core.IO/M2/MdxToM2Converter.cs` + `M2ToMdxConverter.cs`, and a separate
  `M2ToMdxConverter.cs` under `src/viewer/WoWViewer/Terrain/Transfer/` — their relationship to
  "chunks rewritten using non-standard means of fixing" is unconfirmed and must be established
  during planning, not assumed.
- **MDX light parsing already has some representation** (`MdxLightSummary.cs`/`MdxLightType.cs` in
  `WowViewer.Core`) — whether the torch/light-bearing-object gap is a parsing absence or a
  runtime/effect-wiring absence must be confirmed before implementation, not assumed.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Know exactly which builds and models are broken, and how, before fixing anything (Priority: P1)

An operator (or the next implementer) gets a build-by-build, model-by-model survey across the
staged client library: for each client in the 1.0.0–3.0.1 range, which layout each model actually
uses, which route handles it, and per model, whether identity/skeleton/sequences/geometry/materials
were read correctly or exactly how reading failed.

**Why this priority**: Inherited directly from Spec 154 US1. Every other story depends on the true
mapping, and this project has already paid once for fixing readers against an assumed rather than
measured mapping (154's own "3.3.5 through 4.0.0" assumption was wrong; the real boundary is
`3.3.0`/`0x108`). It converts "MDX and M2 are very much non-functional" into a defect list with
owners, and is a concrete instance of this project's standing "verify detector power before null
results" discipline.

**Independent Test**: Run the survey across the staged client library and read the resulting table.
Complete when every client has a row per surveyed model, no row says "unknown," and any build
sharing a version word with another (e.g. `0x100`) has the evidence that distinguished them
recorded.

**Acceptance Scenarios**:

1. **Given** the staged client library, **When** the survey runs over a fixed set of representative
   models present across the 1.0.0–3.0.1 era, **Then** it emits one row per build/model with build
   identity, declared version, route taken, and per-section outcome.
2. **Given** a model that fails to read, **When** the survey records it, **Then** the row names the
   section and element position at which reading failed, not only that it failed.
3. **Given** two builds that declare the same version word, **When** they resolve to different
   layouts, **Then** the table records the evidence that distinguished them.
4. **Given** the survey's results, **When** compared against `FormatProfileRegistry`'s existing
   MDX/M2 resolution, **Then** the relationship between the two mechanisms (same system, competing
   systems, or one superseding the other) is explicitly documented.

---

### User Story 2 - Objects render with real geometry, and always show at least a bounding box (Priority: P1)

An operator loads a client anywhere in the 1.0.0–3.0.1 range and views its world or model objects.
Today most render as nothing at all, and even the bounding-box fallback other eras show when a full
mesh can't be built is usually also missing.

**Why this priority**: The operator's headline complaint; the direct blocker to using the toolkit
for this entire multi-year client era.

**Independent Test**: Using US1's survey as the map of what should now work, load a representative
model from each surveyed build and confirm visible geometry renders with correctly bound textures;
for any asset that still can't fully render, confirm an accurate bounding box appears instead of
nothing.

**Acceptance Scenarios**:

1. **Given** a model asset from a client build in the 1.0.0–3.0.1 range with embedded skin
   profiles, **When** it is loaded, **Then** the viewer renders its mesh triangles with correctly
   bound textures, not a bounding box and not nothing.
2. **Given** a model asset that cannot yet fully render for a specific, named reason, **When** it
   is loaded, **Then** the viewer still draws an accurate bounding box for its placement.
3. **Given** the same model already confirmed correct in another era (0.5.3, 3.3.0/3.3.5), **When**
   this feature's changes ship, **Then** that other era's rendering is unchanged.
4. **Given** an M2 whose embedded skin data is malformed or truncated, **When** it is loaded,
   **Then** the viewer degrades gracefully (renders what it can, or the bounding box) without
   crashing — no unhandled termination, per Spec 154's D3 finding.

---

### User Story 3 - Skeletons load completely for the `0x100`-era route (Priority: P2)

An operator loading a 1.x-era or early-2.x-era character model receives its full skeleton: every
bone, each bone's parent, and each bone's pivot — with no bone silently missing and no
non-finite value.

**Why this priority**: Inherited from Spec 154 US2 (its D1/D2 findings). This is both a rendering
prerequisite (a model with a discarded skeleton cannot pose correctly even if mesh appears) and the
foundation for any future animation work in this era; the correct layout is already recorded, the
work is connecting it.

**Independent Test**: Load known `0x100`-era models (Spec 154 names Blood Elf and Night Elf from
the `2.0.0.5610` pre-release) and confirm a complete, finite skeleton rather than zero bones or a
failure at a specific bone index.

**Acceptance Scenarios**:

1. **Given** a `0x100`-era character model, **When** it is read, **Then** the reported bone count
   is non-zero and matches the file's own declared count.
2. **Given** that model, **When** its bones are read, **Then** every pivot is finite and every
   parent index is either "no parent" or a valid in-range bone.
3. **Given** a model that previously failed at a specific bone index (Spec 154 measured index 10),
   **When** it is read, **Then** it reads to completion.

---

### User Story 4 - "Fuckported" assets render if Warcraft.NET (or the Benilla reference) can read them (Priority: P2)

An operator loads an asset whose chunks were rewritten by a non-standard third-party conversion
tool to backport it to an earlier client format. Today these render incorrectly or not at all, even
when an external, actively-maintained reference implementation can parse the same file natively.

**Why this priority**: Named directly by today's operator directive as a real, encountered
blocker distinct from the format-profile/embedded-skin gap.

**Independent Test**: Feed a known non-standard-rewritten asset that Warcraft.NET (or Benilla, for
1.x specifically) can successfully parse into the viewer and confirm it renders.

**Acceptance Scenarios**:

1. **Given** an asset with non-standard chunk rewrites that an external reference parser can read,
   **When** it is loaded in the viewer, **Then** it renders rather than failing or rendering
   incorrectly.
2. **Given** an asset that neither this project's reader nor any external reference can parse,
   **When** it is loaded, **Then** the specific parse failure is reported per-asset — never
   silently dropped.

---

### User Story 5 - Torches and light-bearing MDX objects show their light effects (Priority: P2)

An operator views an MDX-era object that defines a light node (a torch, brazier, or similar
light-bearing prop). Today the visual light/glow effect is missing in some cases even though the
object itself renders.

**Why this priority**: Named directly by today's operator directive; builds on Spec 105's
explicitly-deferred "light track wiring" boundary rather than reopening solved ground.

**Independent Test**: Load a torch-bearing MDX model and confirm its defined light/glow effect
renders, using the existing `MdxLightSummary`/`MdxLightType` parse-side data as the source of truth
for what the model actually defines.

**Acceptance Scenarios**:

1. **Given** an MDX model with a light node (e.g. a torch), **When** it is loaded and rendered,
   **Then** its defined visual light/glow effect is visible.
2. **Given** an MDX model whose light node type is not yet supported by the render effect
   pipeline, **When** it is loaded, **Then** the gap is named/logged explicitly rather than
   silently skipped.

---

### Edge Cases

- What happens when a build's model-version field is shared by two (or, per Spec 154's evidence,
  three or more) known-different layouts, e.g. `0x100` meaning 1.0.0, 1.12.1, or a `2.0.0.5610`
  pre-release? Must route deterministically via evidence in the file/build context — see US1
  acceptance scenario 3, US2 is not attempted until this is solved.
- What happens when an asset from an otherwise M2-era build (1.x+) turns out to still be literally
  MDX-formatted (a legitimate pre-release leftover, or a fuckported reintroduction)? Must route
  through the correct MDX profile for its actual content, not the M2 pipeline and not a wrong-era
  MDX guess.
- What happens when a model has zero bones legitimately (a static prop/doodad) versus one whose
  bones were silently dropped by a reader bug? These must not look alike (Spec 154 edge case) —
  the system must distinguish "no bones" from "bones not read."
- What happens when a bone parent index forms a cycle or points outside the array? Must be
  detected and rejected without corrupting a walk of the skeleton.
- What happens when an asset that an external reference parser (Warcraft.NET/Benilla) also cannot
  parse is loaded? Report the failure clearly; never crash, never silently vanish the object.
- What happens when a light-bearing MDX object's specific light type isn't one the current effect
  pipeline models? Name/log the gap explicitly rather than skipping it silently.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST produce a build-by-build, model-by-model survey of the 1.0.0–3.0.1 range
  (build identity, declared version, route taken, per-section outcome) before further reader
  changes are made against an assumed mapping.
- **FR-002**: System MUST NOT infer a build's layout from another build's version word alone;
  layout selection MUST rest on evidence observed in the file/build being read (inherited from
  Spec 154 FR-002, given the measured `0x100` ambiguity).
- **FR-003**: For every in-scope version (M2 format version ≤ 263, i.e. through the WotLK
  externalized-skin boundary) and for any MDX build in range, System MUST read the embedded/native
  skin/geometry data (submesh definitions, triangle indices, texture-unit bindings) instead of
  treating the count/offset as zero.
- **FR-004**: System MUST render placed model objects (M2 or MDX) for every surveyed build in the
  1.0.0–3.0.1 range with visible geometry and correctly bound textures, matching the render quality
  already achieved for other supported eras (0.5.3, `0x108`/3.3.0+).
- **FR-005**: When full mesh rendering cannot yet succeed for a specific asset, System MUST still
  render an accurate bounding-box placeholder for that object's placement rather than nothing.
- **FR-006**: System MUST read the complete bone set (count, parent linkage, pivot) for every
  `0x100`-era model it claims to support, and MUST distinguish "this model has no bones" from
  "this model's bones were not read."
- **FR-007**: No model in the staged library MUST cause an unhandled termination (Spec 154's D3,
  the 4.0.0 beta camera-record crash, is the concrete precedent) — every failure MUST be a
  reported, catchable error naming the section and position at which it occurred.
- **FR-008**: System MUST render assets whose chunks have been rewritten by non-standard
  third-party "fixing"/conversion tools ("fuckported" assets) whenever an external reference
  parser (Warcraft.NET, or Benilla for 1.x) can natively parse the same file.
- **FR-009**: When an asset cannot be parsed by either this project's reader or an external
  reference parser, System MUST report the specific parse failure per-asset rather than silently
  omitting the object.
- **FR-010**: System MUST render the visual light/glow effect for MDX light-bearing objects (e.g.
  torches) whose model data defines a light node, building on the existing MDX light-node parsing
  (`MdxLightSummary`/`MdxLightType`).
- **FR-011**: System MUST NOT regress rendering behavior for any client era or build already
  confirmed working (0.5.3 alpha, `0x108`/3.3.0+, WotLK 3.3.5, etc.) — per AGENTS.md §4 and Spec
  154 FR-009.
- **FR-012**: Every build-support claim MUST name the exact build and model it was verified
  against; support for one build MUST NOT be recorded as support for an adjacent or same-version-
  word build without its own verification (Spec 154 FR-011 — these are rolling releases where
  structurally significant changes land in patch increments without a version-word bump).
- **FR-013**: Where a layout is already recorded in the codebase or in Spec 104/154's research
  artifacts, the reader MUST consume that record rather than restating or re-deriving it.
- **FR-014**: The relationship between `FormatProfileRegistry`'s MDX/M2 resolution and whatever
  era-layout selection mechanism Spec 104/154's work already established (e.g. `M2ModelReader100`)
  MUST be explicitly reconciled — either the registry is extended to be the single resolution
  authority, or it is confirmed superseded/unused for this range and updated accordingly. Two
  competing, silently-diverging resolution mechanisms MUST NOT both remain in the codebase.
- **FR-015**: The two existing MDX↔M2 converter implementations (`WowViewer.Core.IO/M2/` and
  `src/viewer/WoWViewer/Terrain/Transfer/`) MUST be reconciled to a single owned implementation as
  part of or before this work, rather than left drifting in parallel.

### Key Entities

- **Client Build Profile / Layout Profile**: the resolved format+version identity for a given
  build, evidenced rather than inferred; the mechanism this spec must make singular and correct
  (FR-001, FR-002, FR-014).
- **Embedded Skin Profile (View)**: the per-file geometry description (submesh definitions,
  triangle indices, texture-unit bindings) stored inside the `.m2`/MDX file itself for every
  in-scope version — the data whose absence produces the "empty box" symptom.
- **Skeleton**: the bone set for one model — per bone, identity, parent, and pivot; must be
  complete and distinguishable from a legitimately-boneless model.
- **Survey Record**: one build/model row from US1 — identity, declared version, layout used,
  per-section outcome, and failure position where applicable.
- **Fuckported Asset**: a model asset whose chunks were rewritten by a non-standard third-party
  tool to backport it across client eras; parity target is "renders if an external reference
  parser can read it."
- **Light Emitter**: an MDX light node (torch, brazier, etc.) with a defined light/glow visual
  effect not currently wired to the runtime render/effect pipeline in all cases.
- **Bounding Box Fallback**: the placeholder rendering path used when full mesh rendering isn't
  available for an object; must remain functional across this entire era.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every client in the staged library covering 1.0.0–3.0.1 has a complete survey record
  (US1), with no "unknown" layout and no unexplained failure.
- **SC-002**: For a representative sample of models spanning the surveyed range, each renders with
  visible geometry and correctly bound textures — not a blank placement — user-confirmed against
  staged clients.
- **SC-003**: For that same sample, every placed object shows at minimum an accurate bounding box
  in any remaining case where full mesh rendering does not yet succeed.
- **SC-004**: 100% of `0x100`-era character models surveyed return a non-zero, complete, finite
  skeleton whose bone count matches the file's own declaration.
- **SC-005**: Zero models in the staged library cause an unhandled termination.
- **SC-006**: A "fuckported" asset that an external reference parser (Warcraft.NET or Benilla) can
  successfully parse is not silently invisible in the viewer for that same file.
- **SC-007**: Light-bearing MDX objects (e.g. torches) visibly show their defined light/glow effect
  where the source model data defines one.
- **SC-008**: Zero regressions in rendering for client eras/builds already confirmed working prior
  to this feature (0.5.3, `0x108`/3.3.0+, 3.3.5, etc.), verified by existing focused test suites
  plus operator spot check.
- **SC-009**: Every build the system claims to support names at least one model it was verified
  against; no build is claimed on the strength of a different build's result.

## Out of Scope

- **Full material/effect parity** (multi-pass combiners, animated UV/color/alpha shading) beyond
  what correct base rendering requires — remains Spec 105's stated boundary, not reopened here.
- **Ribbon/particle simulation** beyond the specific torch/light-glow effect named by the operator;
  full VFX parity is a larger, separately-scoped effort per `consumer-cutover.md`'s "future proof
  levels."
- **Motion export** (BVH, FBX, pose clips) — inherited exclusion from Spec 154; that tooling does
  not exist and this work does not create it.
- **Cross-era rig comparison** (Spec 154 US4, e.g. comparing a 0.5.3 High Elf rig to a Blood Elf
  rig) — a valid use case that this spec's US3 (skeleton completeness) makes *more* reachable, but
  it is not itself required by today's operator directive and is not re-added as a requirement
  here. Revisit as its own spec slice if the operator asks for it.
- **Client eras outside 1.0.0–3.0.1**: the 0.x MDX-only alpha profiles and the 3.3.5+/4.x+ M2
  profiles are separately covered and not re-litigated here, except where a resolver bug directly
  touches that boundary (FR-002, FR-014). Hard scope ceiling inherited from Spec 154: nothing at or
  beyond 4.0.0 is read, surveyed, or referenced by this work.
- **Building new fuckported-style converters**: this spec is about rendering whatever an external
  reference parser can already read, not authoring new asset-repair tooling beyond what parity
  requires.
- **Real-client boot/gameplay proof**: this spec covers the viewer's rendering of these assets, not
  running or validating an actual game server/client session.

## Assumptions

- The MDX→M2 format switch happened before WoW 1.0.0 shipped, consistent with the codebase's
  current profile boundaries and Spec 104's own framing (0.11/0.12 use MDX and already render
  correctly; 1.0.0+ is M2). FR-001's survey will confirm this holds for every staged build, per
  FR-009 in the original operator uncertainty ("I thought 1.0.0 used the same version of mdx, but
  I was wrong") — read as: the operator's assumption was about *version continuity*, not about
  which container format is used at all.
- Warcraft.NET (`libs/ModernWoWTools/Warcraft.NET`) and Benilla (external, Spec 193) are both
  legitimate parity references; which one is actually consulted per-format is a planning decision,
  not fixed here.
- "Torches and light-bearing objects" refers to MDX light nodes specifically; M2-era light effects
  are a related but separate concern, folded in only if planning finds shared plumbing.
- x64dbg with the automate MCP bridge (per Spec 104) remains the available dynamic-analysis tool
  for any early-alpha layout recovery still needed; Ghidra is a separate setup step if required.
- Real-client visual proof (a model looking right in a staged client of each era) remains
  operator-owned per AGENTS.md §6; this spec's completion is source/build/automated-comparison
  proof plus an operator witness, not a claim from either alone.
- Spec 104's 7 already-checked tasks are treated as **unverified** pending a receipt audit during
  planning (consistent with this project's 2026-09-10 governance audit practice), not as confirmed
  progress to build on blindly.

## Dependencies

- **Spec 104** (legacy-m2-rendering) and **Spec 154** (m2-era-reader-parity) — superseded
  (unimplemented residue absorbed here); both remain readable prior art, not archived.
- **Spec 105** (format-version-profiles) — narrower, valid prior art for the 1.0.0
  texture/animation/lighting pillar; not superseded.
- **Spec 193** (Benilla 1.12.1 reference) — available external oracle for 1.x M2 parity checks.
- `FormatProfileRegistry` (`wow-viewer/src/viewer/WoWViewer/Terrain/FormatProfileRegistry.cs`) —
  must be reconciled with whatever mechanism Spec 104/154 already built (FR-014).
- Warcraft.NET vendored library — parity reference for FR-008/FR-009.
- Existing `MdxLightSummary`/`MdxLightType` parse-side types — foundation for FR-010.
- AGENTS.md §4 (do not touch working format readers without a proven bug + operator instruction —
  this spec is that instruction for the specific, evidenced bugs named above) and §6 (real-client
  proof stays operator-owned).
