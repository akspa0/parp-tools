# Phase 0 Research: Skybox Rendering

**Spec**: [spec.md](./spec.md) | **Date**: 2026-08-18

All Technical Context unknowns are resolved below. No `NEEDS CLARIFICATION` markers remain.

---

## R1: LIT authors sky colours but never a skybox model name

**Decision**: Resolve the **sky gradient source** and the **skybox model source** independently,
each carrying its own provenance. They are not two fields of one record.

**Rationale**: The two sources do not cover the same ground.

| Source | Sky band colours | Skybox model name |
|---|---|---|
| Map-scoped LIT (`.lit`) | Yes — colour tracks 2-6 | **No** |
| `Light*` DBC chain | Yes — via `LightIntBand` | Yes — `LightSkybox.Name` |
| WMO `MOSB` | No | Yes |

`lit-draft.md` shows LIT's only string field is a `char[32]` group `name` at 0x20, and its
sky-related scalar is `highlightSky`, an integer index — there is no model path anywhere in the
format. So on a LIT-era build the gradient is fully authored while the model has **no declaration at
all** and must come from discovery or `MOSB`.

A single "sky profile record with a model field" would therefore be null-model on exactly the era
this branch targets, and would invite filling that null from the other source — which FR-002
forbids. Splitting the resolution keeps FR-002 enforceable: a LIT gradient plus a discovered model
is two independently-provenanced facts, not a silent mix.

**Alternatives considered**: One combined `SkyProfile` with an optional model field — rejected, it
makes the common alpha case look like a partial failure and creates the exact cross-source
fill-in-the-blank pressure FR-002 exists to prevent.

---

## R2: The five-band gradient is a shader change, not a geometry change

**Decision**: Keep `SkyDomeVertexBuilder` as-is. Extend the fragment shader to sample an ordered
band set by height.

**Rationale**: The dome already carries what a five-band gradient needs. `SkyDomeVertexBuilder`
writes a per-vertex `heightFactor` (`ring / rings`, 0 at horizon → 1 at zenith) as vertex attribute
1, and the vertex shader forwards it as `vHeight`. The current fragment shader throws that
resolution away on a single `mix(uHorizonColor, uZenithColor, t)`. Adding bands means uploading an
ordered colour array and interpolating between adjacent bands by `vHeight` — no new vertices, no new
buffers, no change to `Build()`.

**Consequence to watch**: the dome is a **hemisphere** (`phi` sweeps `0 → π/2`), so it has no
geometry below the horizon. Bands the source authors as below-horizon have nowhere to land. The
existing shader already blends toward fog colour under `vHeight < 0.15`; band mapping must be
defined against that same convention rather than assuming a full sphere.

**Alternatives considered**: Rebuilding the dome with a ring per band — rejected, it couples mesh
topology to a source-specific band count and would need a rebuild whenever the source changes.

---

## R3: Skybox model animation free-runs on wall-clock time

**Decision**: Drive the active skybox model's animation from the world time-of-day clock by setting
`CurrentFrame` directly, instead of accumulating real-time deltas.

**Rationale**: `ModelRenderer.UpdateAnimation()` computes its delta from `DateTime.UtcNow` against
the last frame's wall-clock timestamp. For ordinary world models that is correct. For a skybox it is
wrong twice over: scrubbing the time-of-day control does not move the sky, and a paused or
time-frozen world still has a sky animating forward on its own.

Both animators expose a settable `CurrentFrame` (`M2RuntimeAnimator.cs:44-48`,
`MdxAnimator.cs:47-51`), so mapping world time onto the sequence needs no new animator surface. Note
the two clamp differently — `M2RuntimeAnimator` clamps through `ClampFrame(value, _sequenceIndex)`
while `MdxAnimator` assigns raw — so wrap behaviour at the midnight boundary must be handled at the
call site, not assumed from the setter.

**Alternatives considered**: Scaling `PlaybackSpeed` to match the day length — rejected, it stays
delta-driven, so it drifts and cannot jump when the user scrubs time.

---

## R4: The interior-skybox trigger exists but has zero callers

**Decision**: Wire `WmoCameraVisibility.IsInsideRootOrGroup` at the point where WMO instance bounds
are already resident; do not introduce a second interior test.

**Rationale**: A repo-wide search returns exactly one hit — the declaration itself. The helper is a
pure function over `(localCameraPosition, rootMin, rootMax, groupBounds, padding)`, already written
and already correct in shape; it was simply never called. This mirrors `MOSB`: parsed but unwired.
US4 is therefore two connections rather than new logic — preserve the name through the read path,
and call the existing interior test.

**Consequence**: because it takes a **local** camera position, the caller owns the world→local
transform per WMO instance, and padding choice is the caller's. Both belong with the code that
already holds instance transforms and bounds.

**Alternatives considered**: A fresh AABB test at the sky call site — rejected under Constitution II
(one canonical owner); it would duplicate a helper that already exists.

---

## R5: The renderer read path discards the MOSB name

**Decision**: Carry the skybox name on the summary the renderer consumes, rather than routing the
renderer to the standalone reader.

**Rationale**: There are two `MOSB` read paths. `WmoSkyboxSummaryReader` preserves the name but has
no renderer consumer. `WmoSummaryReader` — the one the renderer actually uses — reads the chunk and
then reduces it to `hasSkybox: mosb is { Length: > 0 }` (`WmoSummaryReader.cs:54-72`), so
`WmoSummary` exposes only `bool HasSkybox` (`WmoSummary.cs:94`). The name is read and thrown away at
the point of parse.

Routing the renderer to the second reader would mean opening and re-parsing each WMO root a second
time purely for a string that the first parse already had in hand. Preserving it on the existing
summary costs one field.

**Alternatives considered**: Have the renderer call `WmoSkyboxSummaryReader` on demand — rejected,
it is duplicate I/O per WMO on a path that specs 151/153 are actively trying to make cheaper.

---

## R6: Frame-cost budget measures against two stages that are already instrumented

**Decision**: Express the FR-022 budget against the existing `WorldRenderStage.Sky` (3) and
`WorldRenderStage.SkyboxBackdrop` (4) distributions. Add no new profiler.

**Rationale**: `WorldRenderFrameHistory` already records all 19 stages per frame with p50/p99/max
distributions, hitch attribution, and — importantly — `CameraMovedDuringWindow` /
`CanDemonstrateMovementBehavior` gating, which is the fix for the known static-camera blind spot in
earlier profiling. Sky and skybox are already separate stages, so the before/after comparison is a
direct read of two existing series and needs no instrumentation work.

**Budget**: set from the measured pre-change baseline in Phase 0, not asserted here. The gate is
stated as a delta against that baseline plus an absolute ceiling, both recorded in
`contracts/frame-budget.md` once measured.

**Alternatives considered**: A new sky-specific timer — rejected, the stage already exists;
asserting a fixed millisecond number now — rejected, FR-022 explicitly requires a measured baseline,
and an invented number would be unfalsifiable.

---

## R7: Source selection is currently a manual user toggle

**Decision**: Keep an explicit selection, make it *resolved* rather than *toggled*, and report the
choice. Preserve the existing manual override as an override.

**Rationale**: Today LIT wins only when `_useLitFogOverride` is true — a user-settable boolean
(`WorldScene.cs:3765-3772`), consulted at `:10552`. So which source drives the sky is a UI state, not
a property of the loaded build. Under FR-001 the source must resolve from what the build provides.

The existing code already honours the no-mixing rule for fog, with a comment recording that mixing
DBC colours with LIT fog "produced a profile that no client file actually authored". FR-002 extends
that same discipline to sky. Automatic resolution prefers the map-scoped source over the global one
when both resolve, because map-scoped is the more specific authority — and records that it did, so
the choice is visible rather than assumed.

**Alternatives considered**: Removing the toggle — rejected, it is useful for A/B comparison during
validation and its removal is not required by any FR; it becomes an explicit override whose use is
recorded in provenance.

---

## Cross-cutting: what this changes about the spec's phase order

R1 splits what the spec treated as one resolution into two independent ones. This does **not**
reorder the user stories — US1 still gates everything visible — but it does mean US1 (gradient
colours) and US2 (model visibility) touch **different** resolution paths and can proceed in
parallel after the shared provenance scaffold lands. That is reflected in the plan's phase
dependencies.
