# Feature Specification: Cataclysm 4.x Renderer Evolution

**Feature Branch**: `138-cataclysm-renderer-evolution`

**Created**: 2026-08-08

**Status**: Draft

**Input**: User description: "Capture the Cataclysm 4.x renderer audit as a bounded epic for terrain format evolution, synthesized local data, visual fidelity, and rendering performance across supported client eras."

## Summary

The repository now has a substantial Ghidra-based reference dossier for World of Warcraft
Build 11792 (Cataclysm 4.0.0 internal development build), plus earlier-client archaeology.
This epic turns that material into an evidence-led, cross-era terrain foundation. The first
anchor is 4.0.0 because it exposes the renderer's modern signal and permutation boundaries;
the intended compatibility span is basic terrain support from 0.5.3 through 11.x, with
later-era features added as profile capabilities rather than as separate renderer rewrites.

The dossier identifies candidate improvements including more than the legacy four terrain
layers, `MCLV` vertex lighting, `MCTV` explicit UVs, `MCMT` surface material IDs, `MCLY`
height-blend and UV flags, `MD21` M2 wrapping, 4.x WMO material permutations, and GPU
instancing for dense detail doodads. These are reference findings until verified against
real local clients and representative files; the epic must not silently promote a claim or
sample pseudocode into a universal format contract.

## Cross-Era Foundation and Sequencing

The durable deliverable is a profile-gated terrain core with a small common path and explicit
optional signals. A client profile must determine file ownership (including monolithic versus
split ADT layouts), chunk availability, vertex layout, terrain-layer capacity, lighting inputs,
shadow inputs, liquid support, and material features before those capabilities reach the
renderer. Missing later chunks must degrade to the common terrain path rather than require a
new renderer branch.

The first foundation slice targets basic terrain across the era span: height/mesh construction,
MCNR-compatible normals, terrain texture layers, the available baked color/light inputs, and a
profile-owned shadow/minimap path. Full 0.5.3 parity is deliberately downstream of this
foundation. Later work can add MCLV/MCTV/MCMT variants, split-era files, richer liquids,
materials, objects, and modern client features independently.

Modern tools such as wow.export may be used as comparative reference material for 12.x-era
formats and behavior, but they are evidence sources only; the viewer remains repo-independent
and owns its own profile and rendering contracts.

## Archive and Reference Source Guidance

Archive access is an input boundary, not a terrain-renderer concern. The implementation should
extend the existing `IArchiveReader`/`IArchiveCatalog` seam in
`src/core/WowViewer.Core.IO/Files/` and keep every third-party archive implementation behind
that contract. A client profile selects the source adapter; callers must not select a library by
guessing from a directory shape.

| Source | Intended coverage | Role in this epic | Decision boundary |
|---|---|---|---|
| Existing `NativeMpqService` | 0.x through the original MPQ-era clients | Production adapter for legacy archives and Alpha wrapper archives | Keep as the canonical MPQ owner; do not replace it with CASC code |
| [ladislav-zezula/CascLib](https://github.com/ladislav-zezula/CascLib) and the [WoW-Tools/CascLib](https://github.com/WoW-Tools/CascLib) reference | WoW CASC generation beginning around 6.x | First compatibility fallback for early CASC builds; native reader behind a narrow managed adapter | Prove local-build opening, virtual-path lookup, FileDataID lookup, listfile behavior, and disposal before terrain use |
| [wowdev/TACTSharp](https://github.com/wowdev/TACTSharp) | 6.0+ local, online, and CDN-structured sources | Preferred managed adapter candidate for later CASC builds when its build coverage is proven | Its README still calls out encrypted products and all-build testing as unfinished; keep capability-scoped and swappable |
| [wowdev/TACT.Net](https://github.com/wowdev/TACT.Net) | TACT repository/distribution protocol | Reference or isolated acquisition tool for build/CDN metadata | GPL-3.0 requires a license review before any shipped dependency or linked code |
| [Marlamin/WoWTools.Minimaps](https://github.com/Marlamin/WoWTools.Minimaps) | Modern minimap extraction and MPQ/CASC workflow | End-to-end comparative reference for map/minimap discovery, extraction, and resolution handling | Do not copy its tool ownership into the viewer; use its submodule map and behavior as evidence |
| [Kruithne/wow.export](https://github.com/Kruithne/wow.export) | Retail/Classic CASC plus legacy MPQ browsing and modern map export | Modern terrain, overhead-map, object, and file-discovery comparator | Reference only; no runtime or source-path dependency |
| [wowdev/wow-listfile](https://github.com/wowdev/wow-listfile) | Filename/FileDataID metadata | Existing shared listfile input and provenance evidence | Treat community names as mutable; prefer verified names for stable contracts |
| [wowdev/WoWDBDefs](https://github.com/wowdev/WoWDBDefs) through the existing DBCD integration | Database definitions from 7.3.5 onward and ongoing older coverage | Existing project authority for build, map, area, light, and later DB2 discovery | Reuse the established DBCD/WoWDBDefs path; do not duplicate database parsing in the archive adapter |
| [wowdev/pyCASCLib](https://github.com/wowdev/pyCASCLib) and [wowdev/pywowlib](https://github.com/wowdev/pywowlib) | Python CASC bindings and cross-era format reference | Research, fixture generation, and comparison only | Do not make the C# viewer depend on the Python path |

The initial source matrix is therefore: loose files and `NativeMpqService` for the legacy MPQ
line; Zezula CascLib as the early-CASC compatibility baseline; TACTSharp as a later-CASC managed
candidate; and wow.export/WoWTools.Minimaps as modern behavioral references. The matrix must be
validated by build identity and capability probes, not inferred solely from the nominal expansion
number. DBCD, WoWDBDefs, and the existing wow-listfile integration are already-established
project authorities and are consumed by the new profile work rather than re-planned as new
dependencies.

Every adapter must emit the same provenance record: source kind, configured root or remote mode,
product/build identity, adapter and dependency version, listfile source, locale/install-tag
selection, and a content probe hash. A failed probe is a profile failure, not permission to fall
back silently to another archive reader.

## Current Renderer Baseline

The viewer already has partial 4.0.0 support: most terrain and world content loads, and WMO
and M2 assets render in basic form. This is useful compatibility, not renderer parity. The
known baseline gaps are:

- 4.x shaders and several visual-effect paths are missing or incorrect.
- Some lava-effect models do not render correctly.
- Fog behavior is incomplete or visually wrong.
- The renderer lacks a proper lighting model, including useful point-light support for WMO/M2
  scenes.
- World submission has no mature batching or broad optimization strategy and is substantially
  CPU-bound.

Spec 138 therefore starts from an operating but old renderer. The first implementation slices
must close one proven visual or performance gap at a time, with separate format, shader,
lighting, and frame-time proof. “Loads” or “WMO/M2 sorta works” is not accepted as visual
signoff.

## Reference Dossier

The source bundle is preserved at [`.reference_data/4.0.0.11792`](../../../.reference_data/4.0.0.11792/README.md).
Its README indexes 19 modules covering engine, graphics, terrain, M2/WMO, scene graph,
and ADT evolution. The first review set for this epic is:

- [`02_GRAPHICS_AND_RENDERING.md`](../../../.reference_data/4.0.0.11792/02_GRAPHICS_AND_RENDERING.md)
- [`03_MAP_AND_TERRAIN_FORMATS.md`](../../../.reference_data/4.0.0.11792/03_MAP_AND_TERRAIN_FORMATS.md)
- [`13_MCNK_CATACLYSM_CHUNKS_DEEP_DIVE.md`](../../../.reference_data/4.0.0.11792/13_MCNK_CATACLYSM_CHUNKS_DEEP_DIVE.md)
- [`14_MCLY_MCAL_CATACLYSM_DEEP_DIVE.md`](../../../.reference_data/4.0.0.11792/14_MCLY_MCAL_CATACLYSM_DEEP_DIVE.md)
- [`15_M2_WMO_RENDERER_4X_DEEP_DIVE.md`](../../../.reference_data/4.0.0.11792/15_M2_WMO_RENDERER_4X_DEEP_DIVE.md)
- [`19_ADT_SPLIT_VS_MONOLITHIC_EVOLUTION_DEEP_DIVE.md`](../../../.reference_data/4.0.0.11792/19_ADT_SPLIT_VS_MONOLITHIC_EVOLUTION_DEEP_DIVE.md)

The dossier's monolithic-ADT conclusion for build 11792 is important: 4.x renderer work
must not assume that every Cataclysm-era client uses the later split `_obj0`/`_tex0`/`_lod0`
layout. File ownership and chunk availability must be discovered per build profile.

## Binary Evidence Captured — Build 11792

The live Ghidra project for `WOW-11792patch4.0.0_Alpha-INTERNAL.exe` was queried on
2026-08-08. These are binary observations, not universal contracts for every 4.x build.

- `FUN_005084c0` (`MapChunk.cpp`) recognizes and stores `MCVT`, `MCCV`, `MCLV`, `MCNR`,
  `MCSH`, `MCLY`, `MCAL`, `MCRF`, and `MCSE`; this build's parser did not show `MCTV` or
  `MCMT`.
- `FUN_005096b0` builds the extended terrain vertex stream: `MCVT` supplies height,
  `MCNR` supplies three normalized normal bytes, `MCCV` supplies one vertex field, and
  `MCLV` supplies a separate baked-lighting field. Fallback vertex formats omit `MCLV`.
- `FUN_004b0300`/`FUN_004b03d0` register `Terrain*`/`Terrain*_pcf` permutations. The
  assertion at `00a80c10` names vertex color, shadows, PCF, layer count, point lights,
  environment mapping, and tessellation as independent terrain axes.
- `FUN_004e41c0` toggles the `mapShadows` bit (`DAT_00c1196c & 0x40`).
  `FUN_004e0070`/`FUN_004dffa0` gate `MCSH` on that bit and the MCNK shadow flag, and use
  `0x40 >> shadowLevel` as the base shadow texture dimension. `FUN_004dff00` creates a
  separate `TerrainBlend` shadow render target using the same 64-based resolution family.

Implications: 4.x terrain appearance cannot be treated as normal-only Lambert shading.
`MCLV`, `MCCV`, and `MCSH` are separate profile-scoped signals. The CPU vertex builder
preserves MCNR byte order, so the renderer's current normal-axis transform remains
unproven until the Terrain shader input/constant path is traced. Dossier claims about
`MCTV`/`MCMT` remain unconfirmed for this build.

## Binary Evidence Captured — 0.5.3.3368

The loaded `WoWClient.exe` for 0.5.3.3368 was queried in Ghidra on 2026-08-09. The detailed
dataset audit is in [research.md](./research.md). The evidence changes the acceptance order for
the initial real-data transfer sample:

- Native terrain minimap BLP loading, terrain MCSH/LIT rendering, and object/icon overlays are
  separate paths. A minimap BLP is not a native terrain-shadow target.
- Alpha `MAIN` indexing is row-major (`y * 64 + x`); MCVT is 145 absolute samples and MCNR's
  client component order is `(-b2, -b0, +b1) / 127`.
- The Alpha reader currently discards `MCLY.offsAlpha`, treats valid absolute zero heights as
  missing during gap filling, and resamples raw MCSH into a channel that is too broadly named.
- The current Alpha shadow helper forces synthetic cast shadows without native `lights.lit`
  data, and Alpha object masks are placement heuristics without MCRF or screen-space visibility.
- External WDL is a distinct 545-`int16` MARE signal; the current MCVT-derived lattice is only a
  proxy. Exact MCLQ per-vertex field semantics remain an open proof task.

Until those reader and provenance gates close, v60 may use the synthetic controls and diagnostic
real/synthetic comparisons, but it must not claim to contain an accepted 0.5.3 real training
corpus.

## User Scenarios & Testing

### User Story 1 - Establish a Trusted 4.x Renderer Evidence Base (Priority: P1)

As a renderer developer, I want the new audit material indexed by claim, build, file format,
and proof level so that future implementation work starts from verified evidence rather than
from copied assumptions.

**Why this priority**: The dossier is broad and includes both binary findings and proposed
implementations. A provenance ledger is the safety gate for every later renderer change.

**Independent Test**: Review the evidence ledger against all 19 reference modules and confirm
that every planned 4.x capability has a source link, build scope, proof level, and a falsification
or validation path.

**Acceptance Scenarios**:

1. **Given** the 4.0.0.11792 reference bundle, **when** a developer opens the epic's evidence
   index, **then** each candidate terrain, asset, lighting, and performance capability is
   classified as observed, inferred, synthesized, or unverified.
2. **Given** a claim that differs between client eras, **when** a renderer change is proposed,
   **then** the claim is attached to a build profile instead of being applied globally.

### User Story 2 - Render 4.x Terrain Fidelity Signals (Priority: P1)

As a viewer user, I want 4.x terrain to use the visual information present in its files,
including all validated terrain layers and additional vertex or UV data, so that maps do not
look artificially flattened, incorrectly blended, or darker than the client reference.

**Why this priority**: Terrain is the largest visible surface and the current viewer does not
fully render the 4.x map representation.

**Independent Test**: Open representative local 4.x maps containing the candidate signals,
compare them with a signal inventory and reference captures, and verify that no valid layer or
supported chunk is silently discarded.

**Acceptance Scenarios**:

1. **Given** a tile with more than four validated `MCLY` layers, **when** it is loaded, **then**
   the viewer preserves and renders every supported layer or reports an explicit unsupported
   capability; it never silently truncates the tile to four layers.
2. **Given** a tile containing validated `MCLV`, `MCTV`, `MCMT`, or height-blend metadata,
   **when** the corresponding rendering capability is enabled, **then** the viewer uses the
   signal with the correct build profile and exposes enough diagnostics to distinguish absent,
   unsupported, and malformed data.
3. **Given** a 4.x tile using monolithic ADT storage, **when** it is loaded, **then** terrain,
   texture, and placement data are resolved from the correct owning file without requiring a
   split-ADT naming convention.

### User Story 3 - Make Dense 4.x World Scenes Responsive (Priority: P1)

As a viewer user, I want maps with dense grass, clutter, M2 doodads, WMO materials, and effects
to remain responsive, so that the fidelity improvements do not make the viewer unusably
sluggish.

**Why this priority**: The audit describes the same class of draw-call and state-change costs
that already caused severe slowdowns in the viewer. The current renderer is CPU-bound and has
no mature batching or optimization layer, so fidelity and performance must be planned together.

**Independent Test**: Capture a repeatable frame-time baseline on a dense 4.x scene, apply one
bounded optimization slice, and compare CPU, render-thread, GPU, draw-call, and visible-instance
measurements against the baseline and an older-client regression scene.

**Acceptance Scenarios**:

1. **Given** many compatible instances of the same static model, **when** the scene is rendered,
   **then** compatible instances share a batch or instanced submission without changing visible
   transforms, materials, lighting, or fog behavior.
2. **Given** a model or material that requires unique animation, particles, ribbons, or a
   non-batchable shader path, **when** it is rendered, **then** it takes an explicit fallback
   path and remains visually correct.
3. **Given** a dense 4.x scene, **when** performance is measured before and after a slice,
   **then** the report identifies the limiting stage and does not claim a frame-rate win from
   a test that was not witnessed on a real scene.
4. **Given** a 4.x scene using fog, point lights, shader effects, or lava-effect models, **when**
   it is rendered, **then** each unsupported or incorrect path is identified in the diagnostic
   record and the scene does not silently receive a false visual-parity claim.

### User Story 4 - Build Provenance-Preserving Synthesized Data (Priority: P2)

As a renderer and data-tool developer, I want to synthesize missing or normalized render
artifacts from the client builds already on disk, so that the viewer can benefit from verified
4.x signals without inventing unsupported semantics or shipping proprietary source data.

**Why this priority**: Local builds can provide useful supervision and compatibility fixtures,
but synthesized artifacts are only trustworthy when their source lineage remains explicit.

**Independent Test**: Generate a small opt-in artifact set from named local clients and inspect
its manifest, hashes, build identity, map/tile identity, transformations, and observed-versus-
synthesized labels before any renderer or model consumes it.

**Acceptance Scenarios**:

1. **Given** a configured local client root and an approved map/tile selection, **when** an
   artifact is synthesized, **then** its manifest records the exact source build, source path,
   input signals, transform, generator version, and provenance status.
2. **Given** an absent or unverified signal, **when** synthesis is requested, **then** the result
   is marked as synthesized or blocked and is never relabeled as observed ground truth.
3. **Given** a generated artifact from one client era, **when** a different era consumes it,
   **then** compatibility is checked explicitly rather than inferred from a filename or map name.

### User Story 5 - Preserve Older-Era Rendering While Adding 4.x Capabilities (Priority: P2)

As a viewer developer, I want the 4.x improvements to coexist with the 0.12-1.10.0 and later
profiles already under investigation, so that a capability learned from one client does not
regress terrain or model rendering in another.

**Why this priority**: The new Ghidra work spans many eras, and the viewer is intended to be
useful across the local client library rather than only for one build.

**Independent Test**: Run focused format/profile tests and real-file smoke scenes for at least
one early client, one 1.x client, one 3.x client, and the 4.0.0.11792 reference client.

**Acceptance Scenarios**:

1. **Given** a client profile without a 4.x chunk or flag, **when** its map is rendered, **then**
   the viewer uses the established fallback and does not require fabricated payloads.
2. **Given** a malformed or truncated optional chunk, **when** the map is loaded, **then** the
   viewer reports the problem and continues only through a documented safe fallback.

### Edge Cases

- A tile advertises more layers than its alpha payload can safely contain.
- A 4.x-looking chunk appears in a build with a different header or owning-file layout.
- `MD21`, WMO material, or instancing metadata is present but the current graphics backend
  cannot support the corresponding path.
- A source client contains multiple revisions of the same map or asset and the selected build
  fingerprint is ambiguous.
- A synthesized field would make a visual preview look better but has no demonstrated client
  lineage.
- A dense scene improves GPU time while becoming CPU- or streaming-bound.

## Requirements

### Functional Requirements

- **FR-001**: The epic MUST maintain an evidence index covering the 19-module 4.0.0.11792
  dossier, with source link, build scope, claim, proof level, and validation status for each
  renderer-relevant finding.
- **FR-002**: The viewer MUST represent client capabilities by build/profile scope, including
  file ownership, terrain layer capacity, optional chunks, model headers, material features,
  and rendering fallbacks.
- **FR-003**: The 4.x profile MUST NOT impose a universal four-layer limit on `MCLY`; it MUST
  preserve all observed layer records that the profile validates and MUST report unsupported or
  malformed excess data explicitly.
- **FR-004**: The terrain path MUST inventory and preserve the presence, absence, and validity
  of `MCLV`, `MCTV`, `MCMT`, `MH2O`, and relevant `MCLY`/`MCAL` flags before deciding whether
  to render, synthesize, or fall back for each signal.
- **FR-005**: Height-driven blending and explicit terrain UVs MUST be treated as profile-gated
  capabilities whose visual behavior is validated against real files and reference captures;
  the audit's `0x200` and `0x100` interpretations MUST NOT be treated as universal without
  that validation.
- **FR-006**: The asset path MUST detect and route 4.x M2/WMO format and material variants,
  including `MD21`, documented 4.x WMO shader/material cases, lava-effect model variants, and
  visual-effect dependencies, without breaking established legacy profiles.
- **FR-007**: The world renderer MUST provide a bounded batching or instancing path for
  compatible dense detail and doodad instances, with an explicit fallback for animated,
  effect-bearing, transparent, or otherwise incompatible content.
- **FR-008**: Each performance slice MUST report a repeatable baseline and post-change frame,
  CPU, GPU, draw-call, instance, and streaming measurements for a named real scene.
- **FR-009**: Opt-in synthesized artifacts MUST include source-client fingerprint, map/tile or
  asset identity, generator/version, input fields, transforms, hashes, and observed-versus-
  synthesized status.
- **FR-010**: The viewer and data tools MUST fail visibly when a required capability is absent,
  ambiguous, malformed, or unsupported; they MUST NOT silently relabel legacy or synthesized
  data as 4.x observed ground truth.
- **FR-011**: Every completed implementation phase MUST include focused automated checks and
  real-file validation appropriate to its proof level; a parser/build pass alone MUST NOT be
  reported as renderer signoff.
- **FR-012**: Heavy client harvesting, corpus generation, GPU training, and long-running
  performance captures MUST remain user-run operations with exact PowerShell commands handed off
  after the code and validation surface are prepared.
- **FR-013**: The lighting path MUST define and validate the ownership and contribution of
  ambient, directional, fog, and point-light inputs for terrain, WMO, M2, and effect-bearing
  content; absence of a light source MUST be distinguishable from an intentionally unlit asset.
- **FR-014**: Shader, fog, visual-effect, and lava-effect failures MUST be diagnosable as missing,
  unsupported, malformed, or visually unverified rather than hidden behind a generic fallback.

### Epic Phase Gates

1. **Evidence and profile gate**: index the dossier, resolve the authoritative source paths,
   and define the first 4.x capability matrix.
2. **Terrain gate**: validate layer counts, alpha encodings, optional chunks, and monolithic
   versus split ownership against real local files before changing shared terrain contracts.
3. **Asset and lighting gate**: validate 4.x M2/WMO headers, material permutations, vertex light,
   and scene-lighting behavior with focused fixtures.
4. **Performance gate**: measure dense scenes, then land one independently verifiable batching,
   instancing, shader, visibility, or streaming slice at a time.
5. **Synthesis gate**: produce only opt-in, provenance-complete artifacts after observed data
   coverage and compatibility are established.

The first implementation slice is the evidence/profile gate. This epic does not authorize a
broad renderer rewrite or a training run by itself.

### Coordination with Spec 142 full-map runtime recovery

Spec 142 owns the immediate full-map runtime recovery: it has production evidence that normal
full-residency materialization takes 66.4 seconds and that an opaque overlay owner stalls frames
for 40-44 seconds. That spec first establishes overlay attribution/work admission, then index-first
budgeted tile residency. This epic consumes those stabilized contracts for Cataclysm dense-scene
work: shared immutable asset buffers, instance data, texture/material grouping, and capability-gated
multi-draw or indirect submission. Do not start modern GPU submission work while a CPU overlay owner
can still block a frame for seconds; preserve all 4.x material/effect fallbacks throughout.

The first bounded submission slice is recorded in
[`wmo-doodad-batching-slice.md`](./wmo-doodad-batching-slice.md). It adds conservative opaque WMO
shell instancing for portal-free, manually visible groups and retains per-placement handling for
transparent, liquid, portal-sensitive, and WMO-internal doodad content. Its real-scene performance
status remains pending user-run capture.

### Key Entities

- **4.x capability profile**: A build-scoped description of file layout, chunks, flags, asset
  headers, material features, lighting inputs, and safe fallbacks.
- **Evidence claim**: A renderer-relevant statement linked to a reference module, build, source
  location, proof level, and validation result.
- **Terrain signal**: A layer, alpha encoding, vertex-light, UV, material, liquid, or related
  payload that may be observed, unsupported, malformed, or synthesized.
- **Performance scene record**: A named real scene with camera/setup, client fingerprint,
  visible content, frame metrics, and comparison baseline.
- **Synthesized render artifact**: A derived local artifact with complete source lineage and an
  explicit observed/synthesized status.

## Success Criteria

### Measurable Outcomes

- **SC-001**: The evidence index links all 19 audit modules and assigns a proof status to 100%
  of the 4.x renderer claims selected for implementation planning.
- **SC-002**: On a validated sample of at least three 4.x map tiles containing more than four
  terrain layers, zero valid layers are silently dropped; each unsupported field is surfaced in
  diagnostics.
- **SC-003**: At least three representative 4.x scenes render terrain, placements, models,
  liquids, and available lighting signals with no newly introduced missing-content failure,
  and at least one scene has a reference comparison record.
- **SC-004**: Every performance change has before/after measurements, and the first dense-scene
  optimization demonstrates a measurable reduction in its identified limiting cost without a
  greater-than-10% regression on the selected older-client comparison scene.
- **SC-005**: 100% of synthesized artifacts accepted by downstream tooling carry a complete
  source fingerprint, identity, transform, hash, and observed-versus-synthesized label.
- **SC-006**: Focused profile, format, and renderer checks pass for the 4.0.0.11792 fixture plus
  one early and one 1.x/3.x comparison fixture before the epic advances beyond its corresponding
  phase gate.
- **SC-007**: Before renderer implementation begins, a witnessed 4.0.0 baseline records the
  current shader, effect, lava-model, fog, lighting, point-light, draw-call, and CPU-bound gaps
  for at least one representative world scene.

## Assumptions

- Build 11792 is an internal Cataclysm 4.0.0 reference point, not proof that every 4.x build
  has identical files, flags, or renderer behavior.
- The reference dossier and local clients remain read-only evidence inputs; production code and
  tests belong under `wow-viewer`.
- The reported 8+ layer capability is a hypothesis to measure against real files, not a reason
  to fabricate eight layers for older or sparse tiles.
- Existing format readers and Spec 136's narrow M2 batching work remain separate ownership
  surfaces; this epic may extend them only through bounded, evidence-backed slices.
- `H:\CLIENTS` and other explicitly configured client roots are the source of real-file proof;
  machine-local paths must not be baked into source or portable configs.
- The user runs client harvests, heavy corpus generation, GPU work, and long performance captures.

## Out of Scope for the Initial Epic Note

- Reimplementing all 19 audited subsystems in one phase.
- Treating decompiled or illustrative C++/C# snippets in the dossier as drop-in production code.
- Shipping proprietary client files, harvested corpora, trained weights, or generated outputs.
- Replacing existing readers without a concrete, profile-scoped compatibility gap and focused
  round-trip or real-file proof.
- Starting training, broad client harvesting, or a long-running renderer benchmark in this note.
