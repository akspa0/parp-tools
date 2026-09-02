# Next Day of Work — Ordered Implementation Plan

**Written**: 2026-09-02
**Purpose**: One ordered pass through the open specs so they can be implemented back-to-back without
re-deriving context or tripping over each other's dependencies.

Read [STATUS.md](./STATUS.md) for per-spec detail. This file is only the **order and the reasons**.

---

## Ordering rationale

Three constraints drive the order:

1. **Diagnosis before construction.** Spec 208's Phase 0 is mostly *reading* spec 195, and it decides
   how much of the transplant tool already exists. Doing it first can remove most of 208's work.
2. **Blocked work waits.** Anything needing the datastore waits on the TensorStore migration, which is
   operator-run.
3. **Cheap verification first.** Several fixes are code-complete and need only an operator look; those
   are grouped so one session in the viewer clears them all.

---

## Block 0 — Operator verification sweep (~30 min in the viewer, clears 6 items)

None of these need code. One pass through the viewer confirms or reopens all of them.

| What to check | Expect | If wrong |
|---|---|---|
| MoP/Cata terrain lighting | Terrain lit correctly, not dark | Re-run `inspect adt terrain-shading` |
| 0.5.3 loads without crashing | No AV in `DrawElements` | Toggle off "GPU instancing for opaque models" to bisect |
| Fade band (doodads at distance) | No popping/sorting artifacts | Sort faded batch back-to-front by centroid |
| Near objects load before far ones | Near first | Report `_priorityMdxLoads` backlog size |
| 0.5.3 phase layer loads tiles | `[AlphaADT] Phase patch (…) patched=N` in log | **Send the `[AlphaADT]` / `[TerrainManager]` log lines** |
| Frame panel: `of which distance-faded` | Non-zero; `distance fade below 0.999` now 0 | Instancing not engaging |

**Capture while you are there** (spec 207 T004): opaque MDX submissions, instanced / state-hoisted /
unbatched, the per-gate breakdown, WMO groups considered/admitted/rejected, and stage medians.

---

## Block 1 — Spec 209 liquid convergence: finish what was started

**Mechanism A is fixed** (presence-weighted MCLQ upsample, 7 tests). **Mechanism B is not measured.**

- **209-T1** Build `inspect adt liquid-convergence --client <dir> --map Azeroth`, reporting per tile:
  cells MCLQ-only / WL-only / both / neither, the height-difference **distribution** where both, and
  the count of **partially-present MCLQ quads** (Mechanism A's population, so the fix's reach is
  measured rather than assumed).
- **CONFOUND — do not skip.** `ReadWlFiles` runs only in `AdtTensorPackBuilder.Build(adtPath, …)`;
  `BuildFromBytes` passes `null, null` for WL*. A scanner built on archive bytes reports **zero WL
  coverage regardless of the truth**. Use the path that loads WL*.
- **209-T2** Run on the Wetlands coast (operator-supplied subject: unchanged in 23 years, so
  reproducible).
- **209-T3** If gaps remain, they are Mechanism B (`KeepOnlyAboveTerrain` culling WL* at the
  waterline). If the report finds cells covered by a source but **absent from unified**, both
  hypotheses are wrong — the merge is a union and cannot drop coverage, so the fault is outside
  `BuildUnifiedLiquid`, most likely the renderer.

**Why first**: small, self-contained, and it closes a loop already opened. Also unblocks 208, whose
transplants carry liquid and would otherwise inherit the defect and be blamed for it.

---

## Block 2 — Spec 208 Phase 0: audit before building

**Do not write transplant code before this.** Spec 195 (complete) already ships the engine:
`ChunkTranspositionOptions` with `RotationDegrees` / `MirrorX` / `MirrorY` / relative-height
anchoring / per-attribute includes, `ExtractPayload` → `TransformPayload`, on a global chunk lattice
giving **chunk-granular (1/16 tile)** offsets, with `EditorSession` undo/redo.

- **208-T001** Where would map identity have to enter for source ≠ target? That is the entire delta.
- **208-T002** Does 195's rotate/mirror transform **normals and placement rotations**, or only vertex
  positions? FR-005 depends on it. If not, fix it in 195, not around it.
- **208-T003** Map `PhaseDataChannel` onto `ChunkTranspositionOptions`. **Two channel models exist
  today; do not create a third** (Constitution II).
- **208-T004** **Operator question**: does "partial tile offsets" mean chunk granularity (already
  available) or sub-chunk (needs interpolation — materially larger)?
- **208-T005** Does a transplant apply to the live map, a staged artifact, or the datastore?

**Highest-risk silent corruption in all of 208**: MCLY texture indices are per-map. Copying them
across maps paints the wrong textures with no error (FR-010 / T103).

---

## Block 3 — Spec 208 Phases 1–2: cross-map sourcing + tile picker

Only after Block 2. Phase 1 is the transplant itself; Phase 2 is the 64×64 minimap picker.

Note for the picker: 0.5.3 has **no loose minimap directory** (only `md5translate.txt`), so
"tile with no preview" must be visually distinct from "no tile" — measured, not hypothetical.

---

## Block 4 — Spec 207 Phase 2: WMO group admission

The largest single frame cost (~55 ms of a ~100 ms frame) and rejection is currently **0 of 80
groups**.

- **207-T201** Split the conservative-fallback counter by reason **first** — one counter hides two
  defects with opposite fixes ("no portal data in file" vs "portal data present, graph not built").
- **207-T202/203** Per-group frustum + projected-size rejection, independent of any portal graph.
- **207-T205** Behind a toggle, default off until the interior walk-through passes.

---

## Blocked — needs the operator's environment

**TensorStore migration (spec 206).** `pyproject.toml` needs `tensorstore`; `zarr_store.py`,
`zarr_io.py` and `v22_zarr_io.py` use `zarr.codecs.BloscCodec` directly. Needs a dependency install
and a run against real stores.

**Constraint, non-negotiable**: Python owns the datastore. **C# does not implement Zarr or
TensorStore.** C# emits `ARRY/ENDS` blobs via `RawArraySerializer`; Python's
`harvester.raw_reader.read_tile_blob` ingests them. A C# Zarr reader was written on 2026-09-02 and
deleted the same day. See `feedback_python_owns_the_datastore`.

Everything downstream of the store — merged-tile export, alpha WDT / LK ADT converters — waits here.
**3.3.5 ≡ 4.0.1** for MCNR order (operator-confirmed), so the LK writer is otherwise unblocked.

---

## Standing rules that keep biting

- **Measure before fixing.** Spec 205: the obvious field reading was wrong, the wiki threshold did not
  hold, and the first measurement came back *inverted* due to a reader bug.
- **Prove the detector.** `BuildFromBytes` reporting zero WL coverage is a null result from a detector
  that cannot see the thing.
- **Never silently drop.** Every skip gets a counter and a reason.
- **`DBCDRow.ID` is positional** for MoP WDB2 tables. Key on the `ID` column.
- **No test project references the viewer.** Logic that must be tested belongs in `WowViewer.Core*`.
- **Close the viewer before running tests** — it locks the build output.
