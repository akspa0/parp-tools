# Phase 0 Research: Precise Object Selection

## 1. Pickable mesh data is parsed, then discarded — MEASURED (2026-08-16)

**Decision**: Pickable geometry must be either newly retained or re-read on demand. It is **not** already
sitting in memory for regular placed objects, contrary to the working assumption spec.md deliberately
flagged as unverified.

**Rationale — measured from source**:

| Type | Retains | Does NOT retain |
|---|---|---|
| `WmoMeshSummary` (`WorldAssetManager.cs:28-41`) | Version, group/vertex/index/triangle/batch **counts**, bounds, `FootprintSampleVertices` (samples), per-group summaries | Full vertex array; any index array |
| `WmoGroupMeshSummary` (`:43-53`) | Per-group counts, bounds, footprint samples | Full vertex/index arrays |
| `MdxCollisionMeshSummary` (`:55-64`) | Counts, bounds, `FootprintSampleVertices` | Full vertex/index arrays |

`BuildWmoMeshSummary` (`:1655-1680`) *does* walk real `group.Vertices` / `group.Indices` at load time —
the data exists during parsing, and is then summarized away. `ComputeWmoGeometryBounds` (`:1707`) is the
same story: it iterates every vertex, keeps only min/max.

**Consequence for the plan**: Phase 2 (US1) is materially larger than Phase 1 (US2). PM4 objects keep
their real triangles (`Pm4OverlayObject.Triangles`), so US2 is a pure test-swap; regular objects have no
retained triangles at all, so US1 must first solve data availability. This is why the plan inverts their
order.

**Alternatives considered**: Assuming the render path's GPU buffers could be read back. Rejected —
a GPU readback per pick would be far more expensive than either retaining or re-reading CPU-side, and
this codebase has no existing readback path to reuse.

## 2. There is already a re-read-on-demand-and-cache precedent

**Decision**: Follow `TryGetMdxCollisionSummary`'s existing pattern rather than inventing a new one.

**Rationale**: `WorldAssetManager.cs:471-500` already does exactly this shape — check a cache, and on
miss, resolve the canonical model path, `ReadFileData(...)`, parse, cache the derived result (including
caching a **negative** result as `null` so a repeat miss doesn't re-read). That negative-caching detail
matters directly for FR-002: "geometry unavailable" must be a cheap, repeatable, normal answer, not an
exception path or a repeated file read on every pick.

**Alternatives considered**: Eagerly retaining full geometry for every loaded model. Not rejected —
deferred to a measurement (Phase 0 step 3), since the tradeoff is memory-vs-latency and this project has
existing frame-pacing sensitivity (Specs 152/153) that makes a per-pick file read worth scrutinizing.

## 3. Not every model kind can supply pickable geometry

**Decision**: Record which model kinds fall back to bounding-volume picking under FR-002, rather than
assuming precise picking is universally available.

**Rationale**: The existing MDX collision path explicitly bails on MD20/MD21 models
(`WorldAssetManager.cs:492`: `if (data == null || data.Length < 4 || WarcraftNetM2Adapter.IsMd20(data) || WarcraftNetM2Adapter.IsMd21(data))`
→ caches `null`, returns false). Whatever Phase 2 builds must expect the same asymmetry: some models will
have pickable geometry and some will not, and FR-002's fallback is the normal, expected path for the
latter — not a defect.

**Alternatives considered**: Treating a model kind without pickable geometry as a bug to fix in this
spec. Rejected — extending model-format coverage is Spec 154's lane, not this one; this spec's job is to
degrade correctly, not to widen format support.

## 4. PM4 geometry is genuinely already available

**Decision**: US2 consumes `Pm4OverlayObject.Triangles` directly, with no new assembly, no re-read, and
no new parser.

**Rationale**: Confirmed during Spec 156's own drafting research — `BuildPm4TileObjects` fan-triangulates
MSUR surfaces and adds MSLK/MSPV wall quads, and the resulting object retains real `Triangles`/`Lines`
with bounds computed from that geometry. The picker (`TryPickPm4ObjectByRay`, `WorldScene.cs:12670`)
currently ignores all of it in favor of an AABB test. This is the cheapest real win in the spec.

**Alternatives considered**: None — the data is present and unused; there is nothing to weigh.

## 5. SC-004's performance budget is unquantified and must be pinned

**Decision**: Pin a real number from the existing frame-stats system before Phase 2 claims completion.

**Rationale**: spec.md's own checklist flags SC-004 as its single non-numeric success criterion, left
qualitative deliberately because no picking-specific frame budget had been measured and inventing one
would have been fabrication. This project already has the instrument to fix that
(`WorldRenderFrameStats`/`WorldRenderFrameHistory`, with percentile and hitch analysis), so the number
should come from measurement during Phase 0 step 5, not from a guess written into the spec.

**Alternatives considered**: Leaving SC-004 qualitative through implementation. Rejected — "no
perceptible added lag" is unfalsifiable as written, and this project's own history (Spec 153, "the
profiler is blind") shows how easily an unmeasured performance claim goes wrong.

## Open Research Boundaries

- Retention-vs-re-read for pickable geometry is deliberately **not** decided here — it is a measurement
  (Phase 0 step 3), not an opinion.
- The confirmed-match library's on-disk format is a Phase 3 decision; the constraint that binds it now is
  only that it stores identifiers/paths/provenance and never client asset bytes (FR-016).
- Whether the world cursor marker should reuse `BatchPin`, `BatchOctahedron`, or a new dedicated marker
  shape is a Phase 4 cosmetic decision, not an architectural one — all three submit through the same
  already-correct depth-tested pipeline.
