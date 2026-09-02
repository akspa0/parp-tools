# Feature Specification: Object Draw-Call Reduction

**Feature Branch**: `207-object-draw-call-reduction`

**Created**: 2026-09-02

**Status**: Draft

**Input**: Operator ablation, 2026-09-02: "If I turn off doodads, framerates stabilize to around 17fps
with just wmo's, and over 300 without any objects" — plus "we don't have proper LOD in our renderer
calculations, we just fade stuff in and out, but it still loads behind the culling, doesn't it?!"

## The measurement this is built on

The operator's ablation decomposes the frame cleanly, because frame time is additive:

| configuration | FPS | frame time | delta |
|---|---|---|---|
| terrain only | >300 | <3.3 ms | — |
| + WMOs | ~17 | ~58.8 ms | **WMOs ≈ 55.5 ms** |
| + doodads | ~10 | ~100 ms | **doodads ≈ 41.2 ms** |

**Terrain is ~3% of the frame. Objects are ~97%.** Neither lever alone is sufficient: removing all
WMO cost still leaves ~23 FPS; removing all doodad cost still leaves ~17 FPS.

### The cost is per-draw CPU overhead, not fill rate

This is derived, not assumed. Doodads cost **41.2 ms across 8,989 submitted MDX instances = 4.6 µs
each**. Fill rate cannot produce a constant per-*item* cost across two subsystems whose pixel
coverage differs by orders of magnitude, and a terrain-only pass runs at >300 FPS at the same
resolution, so the GPU is not struggling to fill. **Draw call count is the currency**, and every
requirement below is ordered by draw calls eliminated.

Corroborating stage table (2048-frame flight, Valley of the Four Winds, MoP 5.0.1):

```
WmoSubmission          median 30.00  p99 64.08  max 76.5
MdxOpaqueSubmission    median 10.66  p99 36.17  max 54.9
MdxVisibility          median  6.90  p99 14.22  max 17.3
SceneMaintenance       median  0.02  p99 15.68  max 103.7
DeferredAssetLoads     median  0.07  p99  0.08  max  4.1
frames over 33.3 ms: 2025 of 2048;  median frame 67.58, p99 114.51
```

Two things to read off it. **`DeferredAssetLoads` is no longer the bottleneck** — max 4.1 ms against
spec 204's 442.9 ms baseline, so 204's urgency does not hold for this scene. And the distribution is
**flat**: p99/median ≈ 1.7 with 98.9% of frames over budget. That is a sustained throughput problem,
not a hitch problem, so scheduling fixes do not apply.

## Three defects found by source inspection

### 1. Faded doodads cost a draw call and render nothing

`WorldObjectVisibilityCollector` added every instance to `VisibleMdx` unconditionally — there was no
zero-fade check. An instance at `opaqueFade == 0` was submitted, drew one draw call, and the shader
multiplied it to zero pixels.

**Fixed 2026-09-02** ahead of this spec: instances below 1/255 fade are dropped and counted in
`FullyFadedMdxCount`. The skip is suppressed under `IgnoreDistanceCulling` /
`IgnoreVisionConeCulling`, because capture flights set those precisely so distance cannot remove
objects — a regression the existing test suite caught.

### 2. The instancing gate excludes the population instancing helps most

The instanced path admits only `OpaqueFade >= 0.999`, at two independent gates
(`WorldScene.cs:11158` and `ModelRenderer.QueueGpuInstance:815`). Everything in the fade band
therefore renders **one draw call per instance**.

Fade begins at 80% of cull distance, so by area the band is `1 − 0.8²` = **36% of the visible disc**.
With roughly uniform doodad density that is about a third of visible doodads forced off the batched
path — and it is exactly the population instancing helps most: distant, numerous, repeated.

**The capability already exists.** Per-instance fade is plumbed end to end:
`GpuInstanceData(ModelMatrix, FadeAlpha)` → attribute location 10 with `VertexAttribDivisor(10, 1)`
→ `layout(location = 10) in float aInstanceFade` → `finalAlpha = outputAlpha * uColor.a *
vInstanceFade`.

The gate is nonetheless **not simply removable**: one instanced draw shares one blend state, and
partially-transparent geometry in the opaque pass needs blend on and depth-write off. Deleting the
gate without addressing that trades draw calls for sorting artifacts.

`QueueGpuInstance`'s gate is additionally a latent defect in its own right: it **silently returns**,
so an instance reaching it past the first gate would never be drawn at all. It is safe today only by
accident.

### 3. WMO groups are admitted wholesale

Operator instrumentation, this session: **80 groups considered, 80 admitted, 0 rejected**; admitted
by rule — conservative fallback 50 (62.5%), gpu-instanced shell 26 (32.5%), **portal traversal only
0**. Portal culling runs and rejects nothing. 215 of 664 WMOs visible.

Spec 151 measured the same shape in Stormwind at a different scale: 7,512 visible groups and 80,484
draw calls, ~10.7 draws per group. Two zones, same defect, so it is not Stormwind-specific.

### Not a defect, but the reason the ceiling is low: there is no LOD

Fade is a **visibility** ramp, not a **cost** ramp. Nothing gets cheaper as it recedes; it gets more
transparent while paying full geometry, full draw call, and (defect 2) worse batching.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Doodads submit a bounded number of draw calls (Priority: P1)

Flying through a doodad-dense zone submits draw calls proportional to the number of **distinct
visible models**, not to the number of placements.

**Why this priority**: Best ratio available, on code that is already instrumented and already behind
a runtime toggle, and it needs no new asset pipeline.

**Independent Test**: Record opaque MDX draw calls and distinct-visible-model count on the same
camera before and after; the ratio must collapse toward a small multiple of distinct models.

**Acceptance Scenarios**:

1. **Given** many instances of one model are visible, **When** the frame submits, **Then** they cost
   a small constant number of draws for that model rather than one per instance.
2. **Given** instances inside the distance-fade band, **When** they submit, **Then** they are
   instanced rather than excluded from instancing.
3. **Given** faded instances are instanced, **When** the scene is compared against the previous
   build, **Then** there is no new popping, z-fighting or sorting artifact in the fade band.
4. **Given** an instance would not be drawn, **When** it is skipped, **Then** it is counted and
   reported, never silently dropped.

---

### User Story 2 - WMO groups are rejected when they cannot be seen (Priority: P1)

Groups facing away from the camera, outside the frustum, or subtending a negligible number of pixels
are rejected before submission, whether or not the WMO has a usable portal graph.

**Why this priority**: The largest absolute number in the frame — 55.5 ms — and the current
rejection rate is literally zero.

**Independent Test**: Stand where 0 of N groups are currently rejected and confirm a non-zero, stable
rejection count with no visible geometry loss.

**Acceptance Scenarios**:

1. **Given** a placement is admitted, **When** its groups are evaluated, **Then** each is admitted or
   rejected on its own bounds rather than inheriting the placement's admission.
2. **Given** a WMO with no portal data in the file, **When** it is evaluated, **Then** it is still
   group-culled by frustum and projected size.
3. **Given** the conservative fallback fires, **When** the panel reports it, **Then** the reason is
   attributed specifically, and "no portal data in the file" is distinguished from "portal data
   present, graph not built".
4. **Given** group culling is active, **When** the camera moves through an interior, **Then** no wall,
   floor or ceiling pops in or out.

---

### User Story 3 - Distance reduces geometry cost, not just opacity (Priority: P2)

Distant models draw fewer triangles by selecting a lower-detail skin profile the client already
ships, and the fade band stops carrying full-detail geometry.

**Why this priority**: Multiplies the win from US1 rather than competing with it, but depends on
skin-profile extraction and is the largest new surface. P2 because US1/US2 are pure wins that need
no new asset path.

**Independent Test**: Compare submitted triangle counts at a fixed camera before and after, with no
visible silhouette change at distance.

**Acceptance Scenarios**:

1. **Given** a model with multiple skin profiles, **When** it is distant, **Then** a lower-detail
   profile is selected by projected size.
2. **Given** a model with only one skin profile, **When** it is distant, **Then** it renders exactly
   as before.
3. **Given** LOD selection is active, **When** the camera approaches, **Then** the transition does not
   visibly pop.

---

### Edge Cases

- A model whose local lights are transformed per instance — spec 202 holds these out of instancing.
- Instances straddling the fade threshold across frames, causing batch churn.
- WMOs with a single enormous group, where group culling can reject nothing useful.
- Capture and validation flights that intentionally bypass culling.
- Models with no bounds resolved yet, whose placeholder AABB must not permanently exclude them.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Opaque doodad draw calls MUST scale with distinct visible models, not with placements.
- **FR-002**: Distance-faded instances MUST be eligible for instancing.
- **FR-003**: Instanced rendering of faded geometry MUST preserve current visual output — blending
  and depth behaviour correct for partially-transparent instances.
- **FR-004**: An instance that is not drawn MUST be counted and attributable. Silent drops are
  prohibited, including inside `QueueGpuInstance`.
- **FR-005**: The distinct **visible** model count MUST be measured per frame — it is the floor
  FR-001 is measured against, and the existing panel figure is a *resident* count.
- **FR-006**: WMO groups MUST be admitted or rejected individually, on their own bounds.
- **FR-007**: Group culling MUST NOT require a portal graph.
- **FR-008**: Fallback admissions MUST be attributed per reason, separating "no portal data in file"
  from "portal data present, graph not built".
- **FR-009**: Distant models SHOULD select a lower-detail skin profile where the asset provides one,
  and MUST render unchanged where it does not.
- **FR-010**: Every change MUST be switchable at runtime so a before/after can be captured without
  rebuilding.
- **FR-011**: No change may rely on a measurement the frame panel cannot show; each requirement is
  paired with a counter.

### Key Entities

- **Submission**: one item handed to the renderer. The unit the 4.6 µs constant is measured against.
- **Draw call**: one GL draw. The currency.
- **Instance batch**: submissions of one model collapsed into one draw. Now split by fade state.
- **Group**: one WMO sub-mesh; the unit admission must operate on.
- **Skin profile**: an authored LOD level already present in the asset.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Opaque MDX draw calls fall by **≥10×** on the operator's Valley of the Four Winds
  camera, from a measured baseline of 8,989 submissions.
- **SC-002**: `MdxOpaqueSubmission` median falls from 10.66 ms to **≤3 ms**.
- **SC-003**: Zero instances are dropped without appearing in a counter.
- **SC-004**: WMO group rejection rate is **>0%** where it is currently 0/80, with no visible
  geometry loss on a pixel-compared capture.
- **SC-005**: `WmoSubmission` median falls from 30.00 ms to **≤15 ms**.
- **SC-006**: Frames over 33.3 ms fall from 2025/2048 to **<50%** of frames.
- **SC-007**: Doodads-off and objects-off ablations still bracket the result — terrain-only stays
  >300 FPS, confirming no regression was introduced below the object layer.
- **SC-008**: Submitted triangle count at a fixed distant camera falls measurably once US3 lands,
  with no silhouette change detectable by pixel comparison.

## Out of Scope

- **Off-thread asset decode** — spec 204. `DeferredAssetLoads` max is 4.1 ms here; it is not the
  bottleneck in this scene and re-measuring is a precondition for reopening it.
- **`SceneMaintenance`** — median 0.02, max 103.7 ms. A 5000× spread is one rare expensive operation,
  not a distribution; it needs diagnosis before it can be planned and does not belong in a
  draw-call spec.
- **Generating LOD meshes.** US3 uses profiles the client already ships; it does not decimate.
- **Portal traversal correctness** — spec 200/151 keep that. This spec only requires that group
  culling not *depend* on portals.
- **Terrain rendering.** It is 3% of the frame.

## Dependencies

- **Spec 202** — instancing exists and is reachable; this spec removes the gate that keeps it from
  reaching the population that matters.
- **Spec 201** — per-path submission attribution is the instrumentation every SC is read from.
- **Spec 200 / 151** — own WMO portal admission. US2 takes the group-bounds slice.
- **Spec 193** — skin profile extraction (`nViews`/`ofsViews`) that US3 needs.

## Assumptions

- **Doodad density is roughly uniform** in the fade annulus, so the 36% area figure approximates the
  affected share of instances. The exact share is measured by FR-005's counters rather than assumed.
- **Distinct visible models is well under 8,989.** If it is not, US1's ceiling is lower than
  estimated and the measurement will say so before the work is built toward.
- **The 4.6 µs per-submission constant holds** across zones. It is derived from one flight; a second
  zone with a very different constant would mean re-deriving the ordering.
- **MoP-era M2s ship multiple skin profiles.** Where they do not, US3 is a no-op for that asset.
