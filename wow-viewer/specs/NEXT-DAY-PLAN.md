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

---

# Addendum 2026-09-02 — New spec pack (212–217)

Six specs were drafted on 2026-09-02. **None are planned yet**; each needs `speckit-plan` before
implementation. This addendum is the recommended order and the reasoning, for a fresh session.

## Recommended order

### 1. Spec 217 — audio lifecycle *(start here)*

Highest value per unit of work, and the only one with a defect the operator hits every session.
The diagnosis is already made from client evidence: a sound is a state machine over six explicit
lists and one that never reaches the delete list never stops. **Planning's first task is to confirm
that diagnosis against our own audio code** — the spec asserts it from the binary plus the symptom,
not from an audit of our implementation. If it is wrong, that is the finding.

Most of it is unit-testable in `WowViewer.Core*` without a sound device — lifecycle transitions,
repeat-mode classification, priority, weighted selection — which matters because no test project
references the viewer. Autoplay is the *outcome*, not the starting point.

### 2. Spec 212 US6 — selection outlines

Small, self-contained, and a standing daily annoyance. Independent of the rest of 212 by design.
The box overlay's inflation was already made proportional on 2026-09-02, which treated the symptom;
US6 removes the cause by tracing the object's silhouette.

### 3. Spec 212 US7 — museum profile

The largest experience change available for the least new machinery: a camera-locked HUD over a
full-window scene, with panels and readouts hidden. Independently shippable — it needs neither US3's
context rig nor US4's authored shells. **FR-032 is the constraint that matters**: a HUD element must
invoke the *same* underlying action as its full-shell equivalent. Two implementations of one
operation drift apart, and the profile silently becomes a fork with its own bugs.

### 4. Spec 216 — model cursor as a scene light source

Do this after 212 US7, because the museum profile is the setting the torch scene wants. The real
work is **not** the cursor: `UploadMdxLights` uploads a model's lights into that model's own shader
program, so a torch lights only itself. Making a model's light reach terrain, WMOs and other models
is a change across three render paths, and it is the whole feature. Plan it as such, or it will be
scoped as a day of wiring and produce a torch glowing alone in an unchanged dark room.

Spec 212 US8 (the clock) pairs naturally here — reaching 3am is the concrete task the clock exists
for — but neither blocks the other; the existing linear slider satisfies 216's requirement.

### 5. Spec 214 — physics

Now much smaller than first drafted: the solver is **licensed in, not written**. Planning's first
deliverable is library selection with license verification — and **cloth must be a selection
criterion, not a discovery**, because US4 (flags moving) is the visible payoff. Domino is decoded as
a contract only: data layouts and observable behaviour, never transcribed algorithms.

### 6. Spec 215 — weather

Depends on interfaces from 143/147/160 that may not exist yet. Coordination work with those specs
comes first. **FR-015 forbids a parallel lighting or fog model** — the failure mode is building one
because the interface was inconvenient. Wind is the single join with 214, and is useful with no
solver behind it.

### 7. Spec 213 — MCP tooling harness

Independent of everything above and can be done at any time. Its load-bearing requirement is FR-007:
the MCP schema and the CLI parser must derive from **one shared definition**, with the build failing
on divergence. A hand-maintained parallel schema does not satisfy it and would reproduce the
documented CLI-drift defect with an extra surface to keep in sync.

## Standing constraints for all of these

- **Era-gate everything.** 5.0.1 is decoded because it is the *complete* implementation; it is not
  evidence about 0.5.3. Domino does not exist there, and the audio backend is entirely different.
  Carry provenance, flag unknown builds. This project has paid for era-blind decoding twice.
- **Logic that must be tested goes in `WowViewer.Core*`.** No test project references the viewer.
- **Close the viewer before running tests.** It locks the build output.
- **Baseline: 9 pre-existing test failures.** Any other failure is new.
- **Ghidra sessions are read-only** unless the operator says otherwise.
