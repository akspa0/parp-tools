# Feature Specification: Renderer Polish — Wireframe Consistency & MDX Lighting (Spec 226)

**Feature Branch**: `226-renderer-polish`
**Created**: 2026-09-06
**Status**: Draft — authored verbatim from operator feedback; not diagnosed or implemented
**Input**: Operator directive 2026-09-06: "the wireframe stuff is inconsistent and fails most of the
time, based on what angle the camera views the terrain or objects, lighting on MDX objects seems
wonky or not really specular."

## Context

Two standing rendering-quality complaints, recorded here so they are not lost:

1. **Wireframe reveals fail depending on camera angle.** The ghost wireframe overlays (Spec 211's
   dual-pass fill + line renders for terrain, WMOs and M2/MDX) disappear or break up from many view
   angles. Suspected class of cause: depth-fighting between the fill pass and the offset line pass
   (slope-scaled bias is absent; a constant polygon offset wins only at some slopes), or line-pass
   state leaking from preceding passes.
2. **MDX/M2 lighting looks wrong and has no real specular.** The model shaders evaluate Lambert +
   ambient only; there is no specular term and per-vertex lighting makes curved surfaces read
   flat/wonky compared to the client.

## Operator evidence 2026-09-06 (second report — narrows US1 considerably)

- **Terrain wireframe only renders on multiple-textured tiles.** Single-texture / array-texture
  tiles show no wireframe. Hypothesis to verify first: the wireframe line pass exists in one terrain
  shader program but not the other (the array-texture "tile terrain" shader vs the per-texture
  shader), or the pass is gated behind a layer-uniform condition.
- **MDX/M2/WMO wireframes do not work at all.** The Spec 211 dual-pass ghost wireframe is fully dead
  on models. Hypothesis to verify first: the batching consolidation (Spec 202) routes models through
  paths that never execute the wireframe pass — `RequiresUnbatchedWorldRender` was narrowed to
  `_wireframe` for the legacy renderer, but native-route renderers have no wireframe path at all.
  WMO wireframe regression suspects: the GPU-instancing/wireframe unbind path in `WmoRenderer`.
- These symptoms supersede the earlier "fails most of the time by angle" framing: angle-dependence
  may have been an artifact of which tiles/objects were under the camera.

## User Stories

### US1 — Wireframe visible from every angle (P1)
The operator toggles a wireframe reveal on any terrain chunk, WMO or model; the wireframe is visible
as a clean line overlay from every camera angle, at every slope, without fill-pass z-fighting.

**Acceptance criteria**:
- Slope-independent line visibility: steep faces show lines as reliably as flat ones.
- No depth flicker between fill and line passes at any distance within fog range.
- The fix is in the shared line-render path (polygon offset / depth-bias policy), not per-call-site
  tweaks.

### US2 — Believable MDX/M2 lighting (P2)
Models shade with a specular response appropriate to their materials, and lighting does not read as
broken from common viewing angles.

**Acceptance criteria**:
- A specular term exists in the model shader path and is material-gated (not a global sheen).
- Per-pixel lighting (or an explicitly justified per-vertex equivalent) replaces the current flat
  read on curved geometry.
- Alpha 0.5.3 clients and later-era models do not regress; the era-split lighting rules (Spec 152)
  are respected.

## Constraints

- Do not break the working ghost-wireframe dual-pass structure from Spec 211; fix the depth/offset
  policy within it.
- Per AGENTS.md §9: diagnose with measurements/screenshots before changing shader math; both items
  need before/after captures as receipts.

## Success Criteria

- **SC-1**: Wireframe reveal survives a full 360° orbit of a steep mountain face and a dense WMO.
- **SC-2**: A side-by-side capture of one model before/after shows the specular and shading delta
  without changing terrain lighting.
