# Research: PM4-Guided Object Transfer and Museum Placement Repair

> **Dependency checkpoint (2026-08-25, Phase 0).** The gate specs 166/167/168/173/175 now have their
> library-first core with passing focused tests, so Phase 0 was unblocked and Phases 1–2 were executed.
> Implemented seams: [`WowViewer.Core.Editor`](../../src/core/WowViewer.Core.Editor/) holds the plugin
> host, the editor↔runtime bridge contracts (`EditorSceneSnapshot`, operations-as-data), the session, and
> the asset-integrity gate; [`AdtPlacementEditor`](../../src/core/WowViewer.Core.IO/Maps/AdtPlacementEditor.cs)
> is the placement mutation surface (move/rotate/scale/add/delete + non-colliding ID allocation +
> MMDX/MMID/MWMO/MWID name-table merge); [`Pm4ReconciliationEngine`](../../src/core/WowViewer.Core.PM4/Reconciliation/Pm4ReconciliationEngine.cs)
> is the deterministic, side-effect-free proposal engine (align/substitute/clone with
> `ReviewRequired`/`Conflict`). Remaining deps are viewer-shell adapters (bridge adapter, PM4/Museum
> overlay preview, save/reload provenance UI) and the user-owned real-client/corpus gates; those are not
> claimed from compilation.

**Date:** 2026-08-25
**Feature:** [176-object-transfer](spec.md)
**Evidence policy:** PM4 claims below are inherited from the measured PM4 workstream and Specs 184/185;
they are not re-derived by this plan. Runtime and visual claims remain validation gates.

## Decision summary

### 1. Treat PM4 as a read-only guide, not as the authoring target

**Decision:** Convert PM4 observations into placement-space guidance and use them to propose edits to the
Museum ADT/WDT placement set. Never mutate or regenerate PM4 in this feature.

**Rationale:** The established coordinate contract is `Pm4CoordinateService.Pm4LocalToAdtPlacement`,
which maps `placement = (MapOrigin - MSVT.X, MapOrigin - MSVT.Y, MSVT.Z)`. The same placement space is
used by `AdtPlacementReader` and the current writer. The PM4 source must remain a stable reference while
the user evaluates the visual alignment.

**Alternatives considered:**

- Rebuild PM4 from the edited ADT: rejected; it changes the guide and makes the operation impossible to
  compare against the original evidence.
- Add a second PM4-to-world conversion in the viewer: rejected; the workstream explicitly records the
  existing transform as settled and warns against another coordinate convention.

### 2. Reuse the existing matching and synthesis pipeline, then add a reviewed reconciliation layer

**Decision:** Extend the existing `Pm4AssetMatchScorer`, `Pm4ReplacementPlacementSynthesizer`, and
`Pm4*MatchingModels` contracts rather than creating another object reader or matcher. Add a separate
reconciliation service that compares PM4 guide observations against the actual ADT placement catalog.

**Rationale:** The scorer already emits ranked candidates, status (`Matched`, `Ambiguous`, `Unresolved`,
`Ineligible`), score breakdowns, and rationale. The synthesizer already emits a replacement proposal with
position/rotation/scale and provenance. The missing seam is matching those proposals to existing Museum
placements, calculating a correction, and requiring a review decision before authoring.

**Alternatives considered:**

- Replace the 2021/Implave-style matching work: rejected; the existing corpus signals and saved choices
  are valuable evidence and must remain usable.
- Add an ML model before the editor workflow exists: rejected; deterministic, explainable candidates are
  sufficient to deliver a reviewable first pass, and model quality cannot be hidden behind a UI score.
- Infer identity from nearest position alone: rejected; proximity is a proposal signal only, never proof.

### 3. Use a two-stage candidate process with explicit ambiguity

**Decision:** First filter by compatible placement kind/tile/bounds and available PM4 geometry. Then rank
using existing asset signals, placement height, footprint/containment, asset path metadata, and confirmed
matches. Emit `Matched` only when the existing scorer's score floor and separation rules are met; otherwise
emit `Ambiguous` or `Unresolved` with competing candidates and missing evidence.

**Rationale:** `Pm4AssetMatchScorer` currently uses a minimum score of `0.45` and an ambiguity window of
`0.03`, while also marking low-confidence PM4 segments for review. Those values are useful starting
defaults, not permission to auto-edit. A review record keeps the score, each component, and the reason
visible so thresholds can be recalibrated against real Museum pairs.

**Alternatives considered:**

- Always take rank one: rejected; this would turn an ambiguous match into silent corruption.
- Require a perfect match before showing a candidate: rejected; it would discard useful human-in-the-loop
  repair work and the existing ranking tool.

### 4. Distinguish alignment, substitution, and cloning

**Decision:** The reconciliation output has three different proposal actions:

1. **Align:** move/rotate/scale an existing placement while retaining its asset reference.
2. **Substitute:** change a placement's asset reference only after the user selects an existing corpus
   candidate and the target name table can resolve it.
3. **Clone:** create a new normal ADT placement from a selected donor asset/placement with a fresh ID.

Each action has its own validation and provenance. Missing or ambiguous matches produce no action.

**Rationale:** A correct model in the wrong position is a different repair from a missing model or a
wrong model. Separating these actions makes the preview understandable and lets the user accept alignment
without accidentally changing the asset identity.

### 5. Keep authoring library-first and preserve bytes outside edited placement data

**Decision:** Put reconciliation models and pure matching/transform logic in `WowViewer.Core.PM4` or
`WowViewer.Core`; put placement catalog/name-table/transaction integration in `WowViewer.Core.IO`; expose
the workflow through the editor bridge/plugin only after the core contract exists. Use the existing
`AdtPlacementWriter`, `LkAdtWriter`, `AlphaWdtWriter`, and converters according to the target format. Add
no ADT/WDT serializer.

**Rationale:** `AdtPlacementWriter` currently provides byte-preserving position transactions for existing
MDDF/MODF rows, while `AdtPlacementReader` exposes the placement catalog. Full cloning/name-table edits
must use the established format writers and be audited for each supported era rather than smuggled into a
translation-only writer. The epic's measured constraint is that authoring code already exists; the missing
work is the bridge and operation composition.

**Alternatives considered:**

- Serialize ADTs inside the viewer: rejected by the library-first and single-owner rules.
- Mutate the loaded renderer arrays and export them directly: rejected; renderer state is not the format
  authority and would bypass raw-chunk preservation.

### 6. Make preview side-effect free and save atomic

**Decision:** A preview is an immutable plan over source fingerprints and placement identities. Accepted
decisions become one editor operation/batch. Before writing, stage all target bytes in memory, validate
IDs, names, tile bounds, supported era, and source fingerprints, then commit all output files together.
Write only loose ADT/WDT content to the configured output directory and write a provenance report beside it.

**Rationale:** The user needs immediate visual feedback without risking the source Museum map. Atomicity is
required when a repair spans tiles; source fingerprint checks also prevent applying a stale preview after
another edit.

**Alternatives considered:**

- Write each accepted row immediately: rejected because a later tile failure would leave a partial repair.
- Save into the game client: prohibited by the editor-platform hard constraints.
- Put provenance in MPQ/CASC: prohibited; Blizzard containers are read-only inputs.

## Existing implementation facts

The source inspection for this plan found the following usable seams:

- `AdtPlacementCatalog` carries source path, file kind, model/WMO name tables, and model/WMO placement
  records with position, rotation, scale, bounds, flags, and unique IDs.
- `AdtPlacementEditTransaction` and `AdtPlacementWriter` already validate placement kind/index/unique ID
  and preserve unrelated bytes for position moves; they do not yet represent rotation, scale, name-table
  merge, or new placement rows.
- `Pm4AssetMatchScorer` already ranks WMO/M2 asset references with score breakdowns and explicit status.
- `Pm4ReplacementPlacementSynthesizer` already creates review-aware position/rotation/scale proposals and
  records fallback provenance.
- `Pm4CoordinateService` already resolves PM4 placement space and the correctly unpadded `_obj0.adt`
  companion path; callers must handle a missing companion rather than choose an arbitrary ADT.
- The viewer already has a PM4 object-match workbench and selected-placement staging, but the epic records
  that the app-side staging is a second implementation with no core/editor owner. The plan therefore
  connects those surfaces through the editor bridge instead of adding more `ViewerApp` staging state.

## Evidence and validation gates

- PM4/ADT coordinate evidence: 55,978/60,560 placement positions contained by paired PM4 footprints under
  the canonical transform; the unswapped alternative scored 412/60,560.
- PM4-to-asset evidence: the existing PM4 object-library maps 904 PM4 objects to 243 source assets; this
  is a candidate corpus, not blanket ground truth for every object.
- Placement-height evidence: the surface end value equals the producing ADT placement Z bit-exactly for
  844/950 objects in the measured corpus; use it as a weighted signal and report exceptions.
- Asset matching evidence: the current scorer's status and rationale must be preserved in every proposal;
  do not collapse a ranked list to one opaque confidence number.
- Real-data validation must identify the configured client root, build/fingerprint, PM4 guide files, Museum
  ADT files, output directory, and hashes. The user owns the real-client visual proof; build/test proof is
  not a substitute for seeing the corrected objects in the viewer or an independent reader.

## Resolved planning unknowns

- **Coordinate frame:** resolved by the existing PM4 coordinate service; no new convention.
- **Matching source:** resolved to the existing PM4 object-library/scorer plus actual ADT placement catalog;
  no duplicate reader or new model is required for the first implementation.
- **Persistence:** resolved to loose edited ADTs/WDTs plus a machine-readable sidecar provenance report;
  no MPQ/CASC output and no client asset bytes in the repository.
- **Undo boundary:** resolved to the editor session operation, with one reviewed batch covering all accepted
  decisions and source-fingerprint checks for stale previews.
- **Failure policy:** resolved to explicit `Ambiguous`, `Unresolved`, `Unsupported`, or `Conflict` results;
  no best-effort writes.
