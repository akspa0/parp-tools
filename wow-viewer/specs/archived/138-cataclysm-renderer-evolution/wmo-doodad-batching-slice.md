# WMO and Doodad Batching Slice

**Parent**: Spec 138 Cataclysm Renderer Evolution
**Status**: implemented; real-client WMO-shell smoke proof complete; doodad/performance proof pending
**Evidence anchor**: `.reference_data/4.0.0.11792/15_M2_WMO_RENDERER_4X_DEEP_DIVE.md`

## Goal

Reduce CPU submission work for repeated opaque WMO placements while preserving the existing
ordered and visibility-sensitive paths. The 4.0.0 dossier identifies `mapObjRender2` material
batch sorting and GPU instancing for dense detail doodads as the relevant native-era guidance.
Those findings guide the shape of this slice; the dossier's pseudocode is not treated as a
universal client contract.

## Current baseline

- `WmoRenderer` already groups opaque WMO-internal doodads by `IModelRenderer` and uses the
  existing GPU instance contract for compatible M2/MDX doodads.
- `WorldScene` still calls the WMO renderer once per visible WMO placement for opaque shell,
  even when the same WMO asset is repeated.
- Portal-sensitive WMO visibility, transparent shell, liquid, animated/effect-bearing content,
  and manual group isolation must remain on explicit fallback paths.

The implementation now groups eligible visible placements by model key, uploads their model
matrices to a renderer-owned instance buffer, and submits each opaque WMO material/group with
`DrawElementsInstanced`. WMO-internal doodads are replayed per placement after the shared shell
draw so their existing runtime visibility and M2 fallback behavior remains intact.

The first user-run Cata 4.0.0.11927 AOI capture completed five warmup frames and one measured
frame with three visible WMO placements. It reported `WmoOpaqueBatchInstanceCount: 3`,
`WmoBatchDrawCallCount: 16`, and `WmoDrawCallCount: 16`, proving the real-client opaque shell
batch path executed. That sample reported zero WMO doodad submissions and is not yet visual or
performance signoff for the doodad path.

The first interactive viewer run against `C:\WoW4-data\WoW-12025` then exposed a native access
violation at `WmoRenderer.DrawBatch -> GL.DrawElements` during world load. The range guard did not
fire on the rerun, so the batch's numeric index range was valid. That rerun also exposed concurrent
ADT parser mutation (`non-concurrent collections`), because background tile loads shared adapter
state. All `TerrainManager` adapter load paths now serialize parser entry while retaining async
stream scheduling, and normal/instanced WMO draws explicitly rebind both their VAO and EBO at the
draw site. The viewer project builds with zero errors; interactive stability remains pending rerun.

The next user rerun no longer showed managed ADT corruption and briefly reached a healthy-looking
`~67 FPS`, but still exited with the same native `DrawBatch -> GL.DrawElements` access violation.
The draw guard now also checks every batch source vertex index against the uploaded vertex count and
includes the WMO model path in any skip diagnostic; interactive stability is still unproven.

The next user run sustained `70+ FPS` while stationary but crashed immediately after camera motion,
again at `DrawBatch -> GL.DrawElements`. This makes camera-driven visibility/streaming admission the
current trigger. The draw boundary now also verifies live VAO/EBO handles and the driver's reported
EBO byte size before submission; no camera-movement stability proof exists yet.

The following user run regressed to an immediate load-time access violation at the same draw call,
before any visible frame. To remove the remaining offset-indexed draw boundary, each WMO batch now
uploads a compact batch-local EBO and submits with a zero index offset; the full-group EBO remains
only for fallback groups. The crash investigation also found that every WMO VAO enabled the
divisor-one instance-matrix attributes even for ordinary non-instanced draws, while the shared
instance VBO could still be zero bytes before the first instanced submission. The renderer now
seeds that buffer with one identity matrix during WMO buffer initialization, so those enabled
attributes always have valid storage. The viewer builds with zero errors; real-client stability is
pending.

## Bounded contract

1. Add a renderer-owned opaque WMO instance-batch seam carrying model matrices and the existing
   fog/lighting uniforms.
2. Enable it only for WMOs that have no portal graph and no manually hidden groups. The batch
   draws all manually visible opaque group batches with `DrawElementsInstanced`; object-level
   frustum admission has already happened in `WorldScene`.
3. Keep transparent WMO batches, WMO liquids, WMO-internal doodads, and any ineligible WMO on
   the existing per-placement path. No visual feature is silently dropped to obtain a batch.
4. Expose batch and instance counts in the existing world render statistics so a real capture
   can prove whether the path was used (`WmoBatchDrawCallCount` and
   `WmoOpaqueBatchInstanceCount`).
5. Add focused contract tests for eligibility and grouping. A build is library/compile proof;
   only a user-run real-client capture can establish frame-time or visual parity.

## Implementation steps

1. [x] Add the narrow viewer-layer WMO instance-batch interface.
2. [x] Add instance-buffer attributes and opaque instanced shell submission to `WmoRenderer`.
3. [x] Group eligible visible WMO placements in `WorldScene` and preserve the fallback path.
4. [x] Add focused planner tests and update the report counters.
5. [x] Build the viewer and hand off a user-run Azeroth capture proving real-client opaque WMO shell batching.
6. [ ] Capture a WMO placement with loaded internal doodads and compare visual/performance behavior.

## Explicit non-goals

- Rewriting WMO readers or MOGP/MOMT parsing.
- Batching transparent geometry, liquids, portal-visible groups, particles, ribbons, or animated
  model content.
- Claiming that a 4.0.0 audit pseudocode sample proves exact shader or CVar semantics for every
  client profile.
- Running the user's full-map capture or any long GPU benchmark in this implementation pass.
