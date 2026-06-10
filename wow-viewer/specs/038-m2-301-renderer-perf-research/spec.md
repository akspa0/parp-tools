# Feature Specification: M2 3.0.1 Renderer Performance Research Slice

**Feature Branch**: `038-m2-301-renderer-perf-research`
**Created**: 2026-06-02
**Status**: Research Slice (Draft, no implementation)
**Input**: User description: "Dig deep with Ghidra on WoW.exe 3.0.1, uncover all aspects of the engine's renderer in this version, as it includes many improvements that our renderer does not handle properly. We must improve our renderer's performance by baking in stuff that the real engine does that we don't do, especially LOD and lighting."

## Scope Note: Research Only

This feature is a **research slice** that documents native renderer behavior in the loaded `WoW.exe 3.0.1.8303` Ghidra binary and other build evidence already present in the repo, identifies concrete gaps in `wow-viewer`'s renderer, and recommends a prioritized implementation roadmap.

This spec explicitly **does not** write any new `wow-viewer` code. Implementation work is intentionally deferred to a follow-on spec (likely `039+`) that will land each recommended slice one at a time under the `036-renderer-improvements` convergence plan.

This spec is **build-agnostic**: 3.0.1.8303 is the loaded Ghidra binary and the primary source of structural evidence, but the conclusions and recommended first slice are written to apply across 3.0.1, 3.3.5, and 4.0.0 builds, consistent with the existing `wow-viewer` renderer convergence plan (spec 036).

## Cross-References

- `wow-viewer/specs/036-renderer-improvements/spec.md` — convergence plan owner; defines library-first phases, the runtime-controls inventory, and the live terrain/world performance lane that this research feeds.
- `wow-viewer/specs/035-m2-render-parity-recovery/` — M2 render parity ownership; this research assumes spec 035 remains the owner of the underlying M2 runtime representation and does not re-litigate it.
- `wow-viewer/specs/037-m2-301-embedded-views-adapter/` — 2.x + 3.0.1 embedded-views M2 parser slice; this research assumes spec 037 owns the model parser layer and does not duplicate per-record stride decisions.
- `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md` — 3.3.5/4.0.0 static evidence baseline; this research extends (does not replace) that note with deeper 3.0.1 anchors.
- `gillijimproject_refactor/specifications/3.0.1.8303/Contracts/M2_MDX_Implementation_Contract_3.0.1.8303.md` — MdxViewer-side 3.0.1 contract; reference only.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Single Native Renderer Behavior Research Pack (Priority: P1)

As a renderer engineer, I can read one research document that consolidates the deep Ghidra evidence on the native WOW engine's renderer in 3.0.1, so I do not have to re-derive per-frame cull, batching, lighting, and detail-doodad behavior from raw decompilation output scattered across multiple notes.

**Why this priority**: Without a consolidated research pack, every future renderer slice starts from zero and re-decompiles the same `FUN_00705230`, `FUN_00788fb0`, `FUN_006ee8e0`, and `FUN_006f2a00` evidence. The research has already been done in this session; the value of capturing it now is that future slices can build on it instead of repeating it.

**Independent Test**: Read `specs/038-m2-301-renderer-perf-research/research.md` and confirm it documents the 3.0.1 graphics options registry, the per-batch alpha-cull algorithm, the scene-side draw-list builder, the master render flag word, the 3-tier SmallCull ladder, and the detail-doodad subsystem, with each finding traceable to a specific 3.0.1 function address and string anchor.

**Acceptance Scenarios**:

1. **Given** a renderer engineer starts a new lighting or LOD slice, **When** they open this research pack, **Then** they can identify the exact 3.0.1 Ghidra function that owns the behavior they need to mimic, with no further decompilation required.
2. **Given** an existing 3.3.5 anchor is referenced from `m2-native-client-research-2026-03-31.md`, **When** the engineer cross-references it, **Then** this research pack explicitly notes whether the 3.0.1 binary confirms, extends, or contradicts the 3.3.5 finding.
3. **Given** a finding is documented here, **When** the engineer reads the corresponding 3.0.1 Ghidra address, **Then** the function in the binary still matches the documented behavior (verified at spec finalization time).

---

### User Story 2 - wow-viewer Renderer Gap Inventory (Priority: P1)

As a maintainer, I can read a per-finding gap statement that lists exactly what `wow-viewer`'s renderer is missing relative to the native engine, with a recommended implementation owner and a one-paragraph rationale for each gap, so the next implementation slice can be chosen based on evidence rather than intuition.

**Why this priority**: The native engine has multiple distinct improvements (per-batch cull, distance cull, max-lights cap, projected-texture path, detail-doodad subsystem, scene-side draw-list). Without a gap inventory, the first implementation slice is chosen by what is "easy" rather than what the native engine actually does.

**Independent Test**: Read `specs/038-m2-301-renderer-perf-research/research.md` and confirm each finding has a "wow-viewer gap" subsection, a "recommended slice" callout, and a "build-agnostic vs build-specific" note.

**Acceptance Scenarios**:

1. **Given** a finding is documented, **When** the maintainer reads the gap subsection, **Then** they can point to a specific file or absence-of-file in `wow-viewer/src/core/WowViewer.Core.Runtime/` that should change.
2. **Given** a gap is recommended as the "first implementation slice", **When** the maintainer checks the rationale, **Then** the rationale cites a measurable performance criterion (e.g. "without per-batch alpha cull, scene FPS in dense world areas drops by N% per spec 036 SC-009 baseline").
3. **Given** a finding is build-specific (e.g. only present in 3.0.1, not 3.3.5), **When** the maintainer reads the build-agnostic note, **Then** the implementation is gated behind a `M2BuildProfile` check rather than enabled unconditionally.

---

### User Story 3 - Per-Batch Alpha-Cull as the First Recommended Slice (Priority: P1)

As a renderer engineer, I can read a focused recipe for the recommended first implementation slice (per-batch alpha-cull + 3-tier SmallCull + DistCull) with the Ghidra evidence already mapped, so I can start coding the slice without re-reading the binary.

**Why this priority**: The user explicitly requested "especially LOD and lighting" focus, and the per-batch alpha-cull combined with the 3-tier SmallCull + DistCull is the most direct match: it is the only finding that simultaneously improves both visibility/cull (LOD-adjacent) and per-batch lighting/material evaluation (lighting-adjacent), and it has the most authoritative 3.0.1 evidence (decompiled + unrolled loop + clear cull threshold constant).

**Independent Test**: Read the "First Recommended Slice" section in `specs/038-m2-301-renderer-perf-research/research.md` and confirm it contains: (a) the per-batch alpha-cull formula, (b) the 3-tier SmallCull ladder formula, (c) the DistCull clamp formula, (d) the list of M2 runtime state fields to add, (e) the test fixture plan against `3_0_1_8303` staged client.

**Acceptance Scenarios**:

1. **Given** a renderer engineer starts the first implementation slice, **When** they read the recommended-slice section, **Then** they can identify every Ghidra function that the slice must mimic and every state field the slice must add without further decompilation.
2. **Given** the slice is implemented, **When** validated against the staged `3_0_1_8303` client, **Then** the validation route runs at least 3.0.1 and 3.3.5 worlds with measurable scene-FPS improvement documented in spec 036's `quickstart.md`.
3. **Given** the slice changes per-batch visibility, **When** the validation route is run, **Then** the visible batch count in dense outdoor scenes decreases by a documented factor (target: ≥30% reduction in culled-frame batch count for dense 3.3.5 routes, per spec 036 SC-009 framing).

---

### User Story 4 - Cross-Build Evidence Mapping (Priority: P2)

As a researcher, I can read a build-by-build map that shows which renderer findings are present in 3.0.1, which are present in 3.3.5, and which are present in 4.0.0, so the recommended implementation slice is grounded in cross-build evidence rather than single-binary archaeology.

**Why this priority**: 3.0.1 is the loaded Ghidra binary for this session, but the `wow-viewer` convergence plan (spec 036) is 3.3.5-focused. A research pack that is 3.0.1-only would be of limited use to a renderer engineer working on 3.3.5 worlds.

**Independent Test**: Read the "Cross-Build Evidence Map" section in `specs/038-m2-301-renderer-perf-research/research.md` and confirm every finding has columns for "3.0.1 (this binary)", "3.3.5.12340 (research note)", and "4.0.0.11927 (research note)", with at least one positive evidence anchor per build where the feature exists.

**Acceptance Scenarios**:

1. **Given** a finding is documented as "3.0.1 only", **When** the maintainer cross-references 3.3.5, **Then** the cross-build map shows the 3.3.5 cvar name (if different) or marks it as absent.
2. **Given** a finding changes between 3.0.1 and 3.3.5, **When** the renderer engineer reads the map, **Then** the implementation is gated behind a `M2BuildProfile` check or version-specific field rather than enabled unconditionally.

---

### Edge Cases

- The loaded Ghidra binary is `WoW.exe 3.0.1.8303` (32-bit, base `0x00401000`); some 3.0.1 functions are decompiled from this binary and have not been re-verified against live x64dbg captures.
- 3.0.1 has a frozen `waterLOD` cvar (FUN_006edfb0 forces it to 0); the recommended slice must not assume `waterLOD != 0` works in 3.0.1 staging.
- The per-method timing gate (FUN_00786b20) logs to a debug string; wow-viewer may not surface this kind of internal timing gate directly, so the recommended slice's telemetry should be observable from outside the engine.
- The 25-cvar graphics options list in 3.0.1 is larger than the 3.3.5 list in `m2-native-client-research-2026-03-31.md`; some 3.0.1 cvars (e.g. `groundEffectDensity`, `objectFade`, `footstepBias`) may be 3.0.1-specific and not exist in 3.3.5.
- The master render flag word `DAT_00edfae0` (initialized to `0x7104b73`) is a single 32-bit bitmask; this is the cleanest single-source-of-truth state model and is recommended for `wow-viewer` to mirror.
- The per-batch alpha-cull formula in `FUN_00788fb0` uses a global cull threshold `_DAT_009455a8`; if `wow-viewer` adopts this, the threshold value must be exposed as a tunable cvar (not hard-coded) so the runtime behavior is observable.
- The 3.0.1 bones are 0xb4 stride at runtime, not 0x70 on-disk; the recommended first slice does NOT need to handle this transformation (it is a spec 037 adapter concern), but the gap inventory should note that downstream slices must.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The research pack MUST document the 3.0.1 graphics-options registration function `FUN_006ee8e0` with the full 25-cvar inventory, default values, callback function addresses, and per-cvar range validation.
- **FR-002**: The research pack MUST document the master render flag word `DAT_00edfae0` (init value `0x7104b73`), with each known bit mapped to a cvar toggle (e.g. bit `0x4` = `terrainLOD`, bit `0x8000000` = `specular`).
- **FR-003**: The research pack MUST document the per-batch alpha-cull algorithm from `FUN_00788fb0` (GetNumBatches) and `FUN_00789440` (GetNumPrimitives), including the formula `model_alpha × transparency_weight × color_weight` and the cull threshold constant `_DAT_009455a8`.
- **FR-004**: The research pack MUST document the 3-tier SmallCull value ladder from `FUN_006edb30` and the squared-distance precompute from `FUN_006f2a00`, with all four `_DAT_00c75e6c/0x70/0x74/0x78` constants and the resulting 12-entry global precompute table.
- **FR-005**: The research pack MUST document the DistCull clamp from `FUN_006ee4a0` (range `[1.0, _DAT_009858e8]`, stored at `_DAT_00c75e3c`).
- **FR-006**: The research pack MUST document the scene-side draw-list builder `FUN_00705230` with the per-model 10-int (40-byte) entry layout, the composite-model tree walk (`+0x68` children, `+0x70` siblings), and the recursive cost-estimation.
- **FR-007**: The research pack MUST document the per-method timing gate `FUN_00786b20` and the 31+ CM2Model methods that call it, with at least 4 method names confirmed from string-anchored callers.
- **FR-008**: The research pack MUST document the detail-doodad subsystem strings (`CDetailDoodad_idx`, `CDetailDoodad_vtx`, `visDetailDoodadList`, `mapDetailDoodadUpdateList`, per-chunk `detailDoodadInst == 0` gate).
- **FR-009**: The research pack MUST document the projected-texture render path `FUN_0088ff30` (RenderModelBatchesForProjectedTexture) and the 4 combiner family strings (`Projected_FadeAdd`, `Projected_FadeOpaque`, `Projected_ModAdd`, `Projected_ModMod`).
- **FR-010**: The research pack MUST provide a per-finding "wow-viewer gap" subsection, listing the current absence of the feature in `WowViewer.Core.Runtime/M2/` (or noting partial coverage if it exists).
- **FR-011**: The research pack MUST provide a per-finding "recommended slice" callout, identifying whether the gap is a candidate for the first implementation slice, a later slice, or research-only.
- **FR-012**: The research pack MUST provide a "Cross-Build Evidence Map" table with one row per finding and columns for 3.0.1, 3.3.5, and 4.0.0, citing the evidence anchor in each.
- **FR-013**: The research pack MUST identify the "first recommended implementation slice" with: the per-batch alpha-cull formula, the 3-tier SmallCull ladder formula, the DistCull clamp formula, the new `M2BuildCullPolicy` runtime state shape, and a test fixture plan against the staged `3_0_1_8303` and `3_3_5_12340` clients.
- **FR-014**: The research pack MUST update `wow-viewer/specs/036-renderer-improvements/research.md` with a cross-reference to this research so future readers of 036's research note see the deeper 3.0.1 evidence.
- **FR-015**: The research pack MUST list every assumption it makes about the staged `3_0_1_8303` client and the `WoW.exe 3.0.1.8303` binary being the loaded Ghidra image, and the failure mode if those assumptions are wrong.

### Key Entities *(research entities, not data entities)*

- **NativeRendererFinding**: A discrete renderer behavior in the WOW engine (e.g. "per-batch alpha cull", "3-tier SmallCull", "scene-side draw-list builder") that has been recovered from Ghidra, mapped to a function address and string anchor, and documented with a wow-viewer gap statement.
- **GraphicsCvarRegistry**: The 3.0.1 set of 25 graphics options registered by `FUN_006ee8e0`, with default, range, callback, and the runtime storage global the callback writes to.
- **MasterRenderFlagWord**: The single 32-bit bitmask at `DAT_00edfae0` (init `0x7104b73`) that 3.0.1 consults per-frame to gate major render features; recommended as the wow-viewer equivalent state model.
- **PerBatchCullFormula**: The decompiled `model_alpha × transparency_weight × color_weight < _DAT_009455a8` test from `FUN_00788fb0` and `FUN_00789440`, including the hand-unrolled 4-iteration loop and the 0x18-stride batch iteration.
- **SmallCullThreeTierLadder**: The 3-step value ladder in `FUN_006edb30` with thresholds at `_DAT_00985318`/`_DAT_00985310`/`_DAT_00985308` and values at `_DAT_00937f88`/`_DAT_00976c04`/`0x3f800000`, plus the `FUN_006f2a00` squared-distance precompute at 12 globals (`_DAT_00c75e80`..`_DAT_00c75ec8`).
- **DistCullClampPolicy**: The `[1.0, _DAT_009858e8]` clamp in `FUN_006ee4a0` with storage at `_DAT_00c75e3c` and the `DistCull must be in range 1.0 - %f` validation message.
- **SceneDrawListEntry**: The 10-int (40-byte) record produced per-model by `FUN_00705230`: `[GetFileName_result, 0, world_x, world_y, world_z, length×scale, model_index, GetNumBatches, GetNumPrimitives, boneCount]`.
- **PerMethodTimingGate**: The `FUN_00786b20` "stalled" assertion that every expensive CM2Model method calls with its name string; emits `Model2: CM2Model::%s stalled: %s\n` and asserts `m_loaded` bit `0x10 & 1`.
- **DetailDoodadSubsystem**: The 3.0.1 detail-doodad subsystem anchored on `DetailDoodad.cpp`, with geometry classes `CDetailDoodad_idx`/`CDetailDoodad_vtx`, per-frame lists `visDetailDoodadList`/`mapDetailDoodadUpdateList`, and per-chunk `detailDoodadInst == 0` gate.
- **ProjectedTextureRenderPath**: The 3.0.1 dedicated projected-texture draw path at `FUN_0088ff30` (`RenderModelBatchesForProjectedTexture`) with 4 combiner families `Projected_FadeAdd`/`Projected_FadeOpaque`/`Projected_ModAdd`/`Projected_ModMod`.
- **M2BuildCullPolicy** (recommended, not yet implemented): A proposed `wow-viewer` runtime service that would own the per-batch alpha-cull, the 3-tier SmallCull, and the DistCull, gated behind the existing `M2BuildProfile` enum from spec 037.
- **CrossBuildEvidenceRow**: A row in the cross-build evidence map with columns for 3.0.1 (this binary), 3.3.5.12340 (research note), and 4.0.0.11927 (research note), and a free-text "notes" column.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: `specs/038-m2-301-renderer-perf-research/research.md` exists and contains the 25-cvar inventory with default/range/callback/storage for each cvar, the master render flag word with each known bit documented, the per-batch alpha-cull formula, the 3-tier SmallCull ladder formula, the DistCull clamp, the scene-side draw-list entry layout, the per-method timing gate caller set, the detail-doodad subsystem, and the projected-texture render path.
- **SC-002**: Each finding in the research pack has a "wow-viewer gap" subsection that names the absent or partial `wow-viewer` file or runtime state, and a "recommended slice" callout.
- **SC-003**: The cross-build evidence map covers every finding with one row per finding and three columns (3.0.1, 3.3.5, 4.0.0) plus notes.
- **SC-004**: The "first recommended implementation slice" subsection contains the 5 required deliverables (per-batch cull formula, SmallCull ladder, DistCull clamp, `M2BuildCullPolicy` state shape, test fixture plan) without any implementation.
- **SC-005**: The research pack cross-references `wow-viewer/specs/036-renderer-improvements/research.md` and `wow-viewer/specs/037-m2-301-embedded-views-adapter/spec.md` so the 036 convergence owner and the 037 parser owner can be located.
- **SC-006**: The research pack does not write any code, create any `wow-viewer/src/...` files, or modify any existing `wow-viewer/` file outside of the cross-reference update in 036's `research.md` and the spec artifacts under `specs/038-m2-301-renderer-perf-research/`.
- **SC-007**: A renderer engineer can read this research pack in a single sitting and identify the Ghidra function for every finding, the wow-viewer gap for every finding, and the recommended first slice's deliverables, without opening the binary.
- **SC-008**: The spec is marked as a research slice (not an implementation feature) in its `Status` field and explicitly states "this spec does not write any new wow-viewer code" in the Scope Note.

## Assumptions

- The loaded Ghidra binary `WoW.exe 3.0.1.8303` remains available for the duration of any future implementation slice that references the function addresses in this research.
- The staged `3_0_1_8303` and `3_3_5_12340` clients under `I:\parp\parp-tools\output\tmp\wowarchive-clients/` are the validation targets for any future slice; `H:\CLIENTS` is not used (per AGENTS.md RULE 9).
- The existing `wow-viewer` M2 representation produced by spec 037's adapter (a normal `M2ModelDocument`) is the input to any future implementation slice; this research does not modify the document model.
- The existing `M2BuildProfile` enum in `wow-viewer` (proposed in spec 037) is the right place to gate any build-specific cull / lighting behavior; cross-build cvars (3.0.1's `groundEffectDensity` vs 3.3.5's absence) are gated here, not duplicated as separate code paths.
- The 3.0.1 graphics-options list is mostly stable across 3.0.1 builds (no 3.0.1 patch client is staged, so the list is treated as a 3.0.1.8303 snapshot).
- 3.0.1 frozen `waterLOD` (FUN_006edfb0) is real; any future slice that adds `waterLOD` to wow-viewer must keep it at 0 for 3.0.1 staging.
- The 25 cvars in 3.0.1 are a superset of 3.3.5's smaller set; future implementation should adopt the union where possible and gate build-specific cvars behind `M2BuildProfile`.
- The native engine's per-batch cull threshold `_DAT_009455a8` is a sensible default but is not the only valid choice; wow-viewer should expose it as a tunable cvar.
- The 3.0.1 0xb4-stride runtime bones are a runtime expansion from 0x70-stride on-disk bones; this is owned by spec 037 and is not duplicated here.
- The 31+ CM2Model methods that call `FUN_00786b20` cover all expensive model operations; not all of them have been decompiled, but the call sites cluster around the per-frame animation/draw hot path.

## Out of Scope

- Implementation of any of the recommended slices (deferred to follow-on specs `039+`).
- New code in `wow-viewer/src/core/` or `wow-viewer/src/viewer/` (this spec creates only spec artifacts under `specs/038-m2-301-renderer-perf-research/`).
- Re-litigation of spec 035's M2 render parity ownership or spec 037's 2.x/3.0.1 adapter ownership.
- Spec 036's plan.md or phase changes (this research only updates spec 036's `research.md` with a cross-reference; phase changes require their own spec/plan edit).
- Live x64dbg capture confirmation of the decompiled functions (static decompilation only; runtime proof is owned by future implementation slices).
- 4.0.6a.13623 (Cata) or later cross-build mapping beyond the 4.0.0.11927 anchors already in `m2-native-client-research-2026-03-31.md`.
- 2.0.0.x render parity beyond the spec 037 adapter scope.
- WMO interior lighting (MdxViewer already has WMO lighting, but a cross-build WMO lighting research is out of scope; the 3.0.1 DBC-only lighting model is what wow-viewer should target first).
- Detail-doodad implementation (noted as a follow-on slice, not implemented in this spec).
- Projected-texture combiner implementation (noted as a follow-on slice, not implemented in this spec).

## Open Questions

- **OQ-1**: Does the staged `3_0_1_8303` client contain extractable `.mdx` sample files for end-to-end per-batch cull validation, or must the test fixture be built from a synthetic MD20? (Spec 037 OQ-1 is the same question; the answer blocks this research's recommended slice too.)
- **OQ-2**: Does the staged `3_0_1_8303` client have full 0x104..0x108 sub-version range, or just one version? (Spec 037 OQ-2; relevant because some cull behavior may differ between 3.0.1 build variants.)
- **OQ-3**: The 3.0.1 `0xb4`-stride in-memory bone layout — is this identical between on-disk and in-memory forms after a known transformation, or is it a runtime-only expansion that requires a per-model stride? (Spec 037 owns this; relevant for downstream slices that touch bones.)
- **OQ-4**: The 31+ callers of `FUN_00786b20` include methods whose names we have not recovered (the format-arg is the method's own name string); should the first implementation slice also instrument wow-viewer to mirror this per-method timing gate, or is the gate out of scope?
- **OQ-5**: Does the `DAT_00edfae0` master render flag word have bits beyond `0x4` and `0x8000000` that we have not yet mapped? (Recovered bits: `0x4` = terrainLOD, `0x8000000` = specular; 30 other bits unknown.)
- **OQ-6**: Does the projected-texture render path (`FUN_0088ff30`) require per-projector setup that we have not yet recovered, or is it a single function call from the main render loop?
- **OQ-7**: The detail-doodad subsystem strings exist but we have not yet decompiled the per-chunk detail-doodad update list code; is the implementation gated by chunk visibility or always-on per-frame?
- **OQ-8**: Should the recommended first implementation slice ship as a self-contained `M2BuildCullPolicy` service, or as in-line cull checks in `M2StaticRenderModelBuilder` and `M2SkinnedRenderModelBuilder`? (Service is cleaner but adds an indirection; in-line is faster but scatters the logic.)

## Notes for the Future Implementation Slice

The recommended first implementation slice is **per-batch alpha-cull + 3-tier SmallCull + DistCull**. The slice is expected to:

1. Add a new `M2BuildCullPolicy` runtime service in `wow-viewer/src/core/WowViewer.Core.Runtime/M2/M2BuildCullPolicy.cs` that owns:
   - `CullThreshold` (default `_DAT_009455a8` value, exposed as a tunable cvar)
   - `SmallCullTier1`/`SmallCullTier2`/`SmallCullTier3` (the 3-tier ladder values)
   - `DistCullMin`/`DistCullMax` (the clamp range)
2. Modify `M2StaticRenderModelBuilder` to consult `M2BuildCullPolicy` for each batch and skip the section if the cull test fails.
3. Add a new `M2BuildProfile`-gated `M2BuildCullPolicyFactory` that returns the right policy for 3.0.1 vs 3.3.5 vs 4.0.0.
4. Add a per-batch alpha-cull telemetry counter that increments on cull, exposed via `wow-viewer` telemetry so validation routes can measure.
5. Validate against staged `3_0_1_8303` and `3_3_5_12340` clients using spec 036's `quickstart.md` validation routes.

The slice is intentionally narrow: it touches one runtime service, one builder, one factory, and the telemetry layer. The remaining findings (scene-side draw-list, per-method timing gate, detail-doodad subsystem, projected-texture render path) are follow-on slices, each its own spec.
