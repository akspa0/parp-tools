# Research: Phase Layer Rotation Tools

**Feature**: [spec.md](spec.md) · **Plan**: [plan.md](plan.md)
**Created**: 2026-09-03

## R1 — How should rotation be represented on a phase layer?

**Decision**: `RotationDegrees` (float) + `RotationOriginTileX/Y` (float, tile space) on
`PhaseLayerSettings`, with zero degrees as the universal short-circuit.

**Rationale**: A single float covers 90/45/free uniformly, so the preset buttons are just
shortcuts over one code path. Tile-space origin keeps the math in the same coordinate system the
offsets already use (the 2026-09-03 fix established that offsets are in tiles and placements
translate by `TileSize`). Zero-degree short-circuit is what makes FR-011 (0.5.3 / unrotated
unchanged) provable rather than hoped for.

**Alternatives considered**:
- *Quaternion/matrix on the layer* — overkill: composition happens per-tile at parse time, and a
  matrix hides the quadrant-exactness the 90° path depends on.
- *Enum of {None, CW90, CCW90, CW45, CCW45}* — cannot express US3's free angle without a second
  field anyway; the float subsumes it.

## R2 — What approximation does 45° use? (FR-009)

**Decision**: **Free-rotate with grid-snapped tile lookup; placements transformed exactly; no
terrain resampling in Phase 1.** The applied mode is reported per layer.

**Rationale**: Resampling heightfields (bilinear across rotated chunk lattices) is the only option
that produces genuinely rotated terrain, but it is destructive to data fidelity, expensive, and —
critically — it invents heights the client never authored, which violates the repo's
"never present a guess as a measurement" discipline. Re-sourcing whole donor chunks by rotated
coverage keeps every height value authentic; the cost is that the rotated result shows the donor's
chunk grid at 45°, i.e. visible chunk seams along the rotation direction. That is an honest,
stated approximation the operator can see and judge. Placements (objects) are point transforms and
rotate exactly, so objects stay seated on the donor terrain they came from.

**Alternatives considered**:
- **Grid-snap the whole layer to the nearest 90°** — dishonest: the operator asked for 45° and
  would get 90° with a report saying "45°". Rejected.
- **Bilinear height resampling** — visually smoothest, but fabricates terrain data and is the
  expensive path. Deferred as an opt-in later mode if the operator wants it after seeing the
  free-rotate result.
- **Free-rotate the rendered mesh (shader-level)** — would rotate terrain visually but break the
  composition seam (channel gates, placement re-chunking, minimap) and cannot be unit-tested in
  Core. Rejected for Phase 1.

## R3 — Fixed transform order for composition (FR-007)

**Decision**: **rotate about origin → apply tile offset → per-tile placement overrides.**

**Rationale**: Rotation is defined about content (the layer's occupied bounds), so it must run
first while the content is still in its authored frame. The offset then places the rotated content
in the grid. Per-tile placement is an explicit operator decision about specific tiles, so it is
the last word for any target tile it claims — this also gives a natural conflict rule: per-tile
mapping beats offset+rotation for the tiles it names. The order is fixed in
`PhaseCompositionPolicy` and reported in logs; both adapters consume the same resolver so the two
eras cannot drift.

**Alternatives considered**: offset-then-rotate (makes the rotation origin move with the offset,
which is harder to reason about when aligning a donor by eye). Rejected.

## R4 — Per-tile placement data model and conflict rule (US5)

**Decision**: a list of `PhaseTilePlacement { DonorTileX, DonorTileY, TargetTileX, TargetTileY }`
on `PhaseLayerSettings`. Target lookup consults mappings first (last mapping wins on duplicate
target, conflict reported), then falls back to offset+rotation.

**Rationale**: A list (not a 64×64 array) keeps the common case cheap — the operator maps a
handful of tiles, not the whole grid. Last-wins is deterministic and matches the existing
"later layer wins" convention in 203's stack. Conflicts are reported, never silent (FR-015).

**Alternatives considered**: a dense 4096-entry array — wasteful and implies a UI grid the
operator did not ask for; a dictionary keyed by target — equivalent to the list for lookup but
loses insertion order, which the conflict report wants.

## R4 — Which donor tile does a straddling placement belong to?

**Decision (default, to be verified against real data in Phase 2)**: an object belongs to the
donor ADT that listed it; it moves with that tile, even if its bounds cross the tile edge.

**Rationale**: it is the parse-time fact — the placement was read from that file — and it keeps
the transform a pure function of parse output. The alternative (re-assigning by rotated position)
would split one object's geometry from its terrain when the terrain re-sources by a different rule.

**Open**: Phase 2's real-data gate (T-gate) loads a real donor map and checks whether straddling
objects visibly detach from their terrain. If they do, the fallback rule is "assign by rotated
position of the object's origin", and research.md is updated with the observed evidence.

## R5 — Rotation origin default

**Decision**: centre of the layer's occupied tile bounds, computed by the adapter from the donor
WDT's existing-tile set; overridable via `RotationOriginTileX/Y`.

**Rationale**: the operator's goal is fitting a donor onto a base map; centre-of-content is where
that alignment is judged, and it is computable without asking the operator for coordinates they
should not have to know.

## R6 — Performance: rotation must not force full-layer loads

**Decision**: rotation changes only the *lookup* (which donor tile fills a target); the existing
AOI-driven lazy tile load is untouched. A rotated layer loads exactly the donor tiles its visible
targets resolve to.

**Rationale**: the spec's edge case ("must not require loading every tile") is satisfied by
keeping composition per-target-tile, which is how 203 already works. No prefetch, no full scan.

## R7 — Minimap under rotation

**Decision**: draw the layer's minimap tiles at their rotated target positions using the existing
per-layer minimap path; if the minimap renderer cannot express a 45° quad cheaply, fall back to
offset-only minimap with the limitation stated in the panel rather than drawing a wrong visual.

**Rationale**: the minimap is a navigation aid; a wrong tile is worse than an absent one. 90°
rotations are exact quads and always drawn.

## R8 — Mirroring: exact-grid, involution, handedness (FR-018, operator clarification 2026-09-03)

**Decision**: horizontal and vertical mirrors are first-class exact-grid transforms alongside the
90° turns. Heights mirror in place; the normal component along the mirrored axis negates; texture
alpha maps mirror; hole/shadow/vertex-colour maps mirror; liquid surfaces follow the heights; and
placement positions mirror with orientation adjusted for the flip (a mirror reverses handedness,
so a placement's rotation must be negated about the mirrored axis, not merely offset).

**Rationale**: a mirror is an exact grid operation like the 90° turns — no resampling, no
approximation, every value authentic. Mirror is an involution (mirror twice = identity), which
gives a free round-trip test (SC-009) exactly like the 90° CW/CCW pair. The handedness flip on
placement orientation is the one subtle part: mirroring position without flipping orientation
leaves objects facing the wrong way, which is the same class of "correct transform, wrong result"
defect as the 2026-09-03 name-table bug.

**Alternatives considered**: treating mirror as a 180° rotation + flip (equivalent math but
obscures the handedness rule); shader-level mirroring (breaks the composition seam and cannot be
unit-tested in Core). Rejected.

## R9 — The transform seam is phase-agnostic (FR-019, operator clarification)

**Decision**: the rotate/mirror/offset math lives in `WowViewer.Core/Maps/TileContentTransform.cs`
as pure functions over `TerrainChunkData` + placement lists, with no phase-system type in scope.

## R10 — Translation precision is the terrain-cell lattice (FR-020/021)

**Decision**: Store translation canonically as signed integer terrain cells: 8 cells per chunk,
128 cells per ADT axis. Tile and chunk offsets are exact UI views over that one state.

**Rationale**: Whole-tile lookup cannot align pasted terrain whose source and target boundaries do
not coincide. A terrain cell is the finest regular authored cell unit that can be translated
without inventing heights. Moving by a non-multiple of 8 requires re-slicing source neighborhoods
at target chunk/ADT boundaries, but does not require interpolation. Arbitrary sub-cell/world-float
translation would require resampling and must be a separate, explicitly stated future mode.

## R11 — WDL magnetization is an optional snap oracle, not the transform (FR-022)

**Decision**: Reuse Spec 196's `WdlLatticeMagnetizer.SampleWdlHeight` contract to score a bounded
search of integer cell offsets and propose the best macro-lattice alignment. Report the candidate
offset and fit score; apply only after operator acceptance.

**Rationale**: WDL can identify broad elevation alignment when local ADT boundaries disagree, but
`MagnetizeChunkHeights` blends/rebases heights and is therefore not an exact translation. Spec 219
must not silently modify the donor terrain to make a fit look better. The WDL path assists choosing
the authoritative integer offset; it does not duplicate Spec 196 or replace exact re-slicing.

## R12 — Existing Chunk Manipulator is not the selection surface (FR-023–032)

**Measured source audit**: the existing panel is a fixed 280-pixel dark grid centered on the camera,
draws no minimap or heightmap, handles only mouse-click selection despite Spec 195 claiming drag
selection, and pastes at the camera tile's chunk origin. The exposed texture and placement options
are not applied by its destination replacement loop; only heights/normals/holes reach the rebuilt
chunks. Its `ChunkSelectionRegion` rectangle/global-chunk helpers and editor operation concepts are
potentially reusable, but the active UI/paste pipeline is not a correctness base.

**Decision**: Build one global tile/chunk/cell selection contract and project it into a full-map
orthographic canvas and the 3D renderer. The old plugin redirects to this workbench and loses its
independent selection, clipboard, target, and paste code after migration gates pass.

## R13 — Base is a channel-gated first layer; phase controls stay on phase cards (FR-028–030)

**Decision**: Base uses the same channel vocabulary as overlays and is filtered before phase merge.
It cannot be removed or reordered. Each phase card directly exposes exact presets, mirrors, free
angle, origin, approximation, and reset. Selected-content transforms remain separate operations.

**Rationale**: Renderer-only visibility would not produce the same preview that save materializes,
and a detached rotation editor would hide the most important per-layer controls from the layer stack.

## R14 — Save materializes the preview and reuses format owners (FR-033–035)

**Decision**: A workbench Save Transformed Map action creates output copies after a complete
representability/conflict preflight. Supported LK-v18 ADT and Alpha WDT routes consume existing
writers/converters. Native split MoP is refused until Spec 197 lands a target-specific writer;
LK output is never labeled as native MoP. The frozen Alpha writer is consumed, not modified.
The phase layer is the first consumer; later editing tooling (beyond the phase map system) calls
the same primitives directly.

**Rationale**: the operator stated this tooling is preparation for later editing work beyond the
phase system. Factoring the seam now means the future editor reuses tested primitives instead of
re-deriving them, and it keeps the phase composition layer thin (adapters call policy + seam, no
math inline). A Core unit test exercises the seam with no phase type in scope (SC-010), which is
the structural proof that the seam is genuinely reusable.
