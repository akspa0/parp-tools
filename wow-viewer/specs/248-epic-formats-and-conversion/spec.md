# Epic 248 — Formats, Readers, Writers & Conversion

**Created**: 2026-09-23 (spec reconciliation) · **Branch**: `v0.6.0-dev` · **Status**: Triage pending

> Primary successor of 21 archived specs (split specs also contribute items; see the ledger). **No new scope**: every backlog item below is
> scope a source spec already stated, cited by id (§9.1). Item wording is condensed; the archived
> source spec is the detail. Evidence: [reconciliation ledger](../archived/reconciliation-2026-09-23/README.md),
> audits [D1](../archived/reconciliation-2026-09-23/audit/batch-D1.md) ·
> [D2](../archived/reconciliation-2026-09-23/audit/batch-D2.md) ·
> [G2](../archived/reconciliation-2026-09-23/audit/batch-G2.md) · [E](../archived/reconciliation-2026-09-23/audit/batch-E.md).

## Goal

Proven, era-faithful read and write for every format the project touches — legacy (0.5.3–5.x) and
modern (FileDataID/CASC, DAT v22/v23/v26) — plus one-way conversion that makes modern and
pre-release terrain usable in legacy clients.

## Delivered baseline (verified in code 2026-09-23 — do not re-plan)

| Capability | Where | Source |
|---|---|---|
| Legacy M2 `MD20` `0x100`–`0x107` routed through `M2Era100ModelReader` (refusal wall removed); 108/112-byte bones, embedded skins, bbox fallback, quaternion/UV/material/animation fixes (8 receipts) | `Core.IO/M2Era100/`, `M2ModelReaderDispatcher` | 235 (+104, 154 evidence) |
| MH2O LiquidObject → LiquidVertexFormat DBC chain, one decoder for `Mh2oChunk` + `AdtLiquidReader` | `Core.IO/Dbc/LiquidVertexFormatChain.cs` + tests | 205 |
| Liquid convergence measurement (`inspect adt liquid-convergence`); root cause known (Mechanism B) | `LiquidConvergenceAnalyzer` | 209 |
| Converter Phase 0 baseline; 3 real defects fixed (MCLY flags, MCAL/MCSH header strip, validator) | `evidence/phase0-baseline.md` | 221 |
| MoP 5.0.1 split-ADT read (`MopAdtChunkParser`, `AdtTileFamilyResolver`); PM4 click selection; multi-client staging | `Core.IO/Maps/`, `Core.Editor/Staging/` | 197 |
| Asset reference inventory sweep (US1) | inspect tooling | 155 |
| LIT documentation draft (wiki submission is an operator action) | `docs/` | 157 |
| DAT v22/v23/v26 (AHDR family) read + render + byte-identical v26 rewrite; `adt-ahdr check\|objects\|roundtrip\|export-lk` | `AdtAhdrReader/Writer/TileSlicer/Alpha`, `AhdrTerrainAdapter` | 237 |
| DAT → LK v18 ADT/WDT export with loss manifest; DAT folders as Cartography layers (`dat:<folder>`) | `AdtAhdrCommandSupport`, `DatLayerSource` | 247 US3/US5 |
| Local CASC install (+ optional CDN fill) via vendored TACTSharp; `inspect casc …` verbs | `Core.IO/Casc/`, `ViewerApp_CascAhdr.cs` | 238 US1 |
| FileDataID-era reads: MAID, MDID/MHID/MTXP, SFID/TXID, GFID/MODI, MD21, DB2 by id | `CascDataSource`, `FileDataIdPaths`, `StandardTerrainAdapter` | 239 |
| WMO MOBA 16-bit material ids, GFID per-LOD groups, shader-23 base texture, 8 terrain layers, `casc wmo-survey/map-survey` | `WmoV17ToV14Converter`, CASC verbs | 240 T000a–f |

## Backlog (spec-stated residue — each item awaits operator triage in [TRIAGE.md](../TRIAGE.md))

### A. Modern data (v0.6 line)

| ID | Item | Source | Today |
|---|---|---|---|
| F-01 | v22 `AMAP` alpha codec (v22 currently blends layer 0 only) + relax the `AhdrTerrainAdapter` 4096-byte all-layers guard so partial alpha sets blend | 247 US1/FR-001–002; activeContext 2026-09-20 | oracle for scored search confirmed (`ACNK`+0x12); codec unidentified |
| F-02 | Real-rendered top-down capture per DAT tile + stitched overview | 247 US2/FR-005–009 | not started |
| F-03 | Modern → legacy map conversion (LK v18 + Alpha 0.5.3), deterministic multi-layer merge + per-tile report, batch mode, low-touch UI, provenance/no-overwrite, optional asset bundle, pre-write validation | 243 FR-001–008 | plan/contracts/data-model authored; zero code |
| F-04 | Modern chunk completeness survey + per-chunk legacy build-in feasibility | 245 FR-001–009 | one scoping note only |
| F-05 | Conformance survey library (chunk inventory + conformance report) — **one** component serving both asks | 240 US1 T003–T013; 239 FR-008/US4 | absent |
| F-06 | WDT `MAI2` liquid-flow decode → viewer liquid context + shared flow datum + legacy disposition | 244 FR-001/003/004/006/010 | absent |
| F-07 | CASC CDN-only remote streaming; historical builds by explicit identity | 238 US2/US3 | local only shipped |
| F-08 | Single file-reference resolver (FDID/path lookups consolidated) | 239 FR-001 | scattered |
| F-09 | Tier A/B build coverage (6.x–8.3); only `wow_classic_beta` 1.60.1 proven | 239 SC-001/003; 238 SC-003 | unverified |
| F-10 | WMO shader-correct materials + split-group portals | 240 US2 T014–T020 | absent |
| F-11 | Terrain high-res holes + MTXP height blending | 240 US3 T022–T026 | absent |
| F-12 | M2 chunk-consumption audit; BLP/WDT companion decisions; wiki-correction write-up | 240 US4/US5, T027–T034+ | absent |
| F-13 | Synthesized minimap for DAT folders (AHDR → tensor-pack builder + DAT-folder input mode) | activeContext 2026-09-19 (237 follow-up) | absent |

### B. Legacy formats

| ID | Item | Source | Today |
|---|---|---|---|
| F-20 | Single M2↔MDX converter (two drifted converters exist) | 235 FR-015 | both still present |
| F-21 | "Fuckported"-asset parity (Warcraft.NET-written assets) | 235 US4/FR-008–009 | no tasks, no code |
| F-22 | MDX light-emitter visual effects (torch/glow) | 235 US5/FR-010 | light uniforms exist; effect absent |
| F-23 | `FormatProfileRegistry` M2 profile unification + delete inert profile | 105 FR-008–014; 235 FR-014 | inert profile still present |
| F-24 | Flat-key/interpolation-range animation addressing model | 105 FR-001–007 | absent |
| F-25 | Native M2 route: remove M2→MDX conversion fallback in `WorldAssetManager` | 110 US2/FR-006–008 | fallback still active |
| F-26 | Published WMO v14/v17 + M2→MDX capability tables with fixture evidence | 110 US4/FR-011–012 | absent |
| F-27 | Liquid shoreline-culling fix (WL* cells culled by terrain) + re-measure | 209 T4–T6 | root cause known |
| F-28 | Converter harness: object round-trip validator, corpus-gate aggregation, oracle cross-check, real-client harness | 221 Phases 1–4 | absent |
| F-29 | MoP native semantics (Ghidra) + sparse merger + native MoP split writer (kept disabled) | 197 T117/T117a/T117b, T122/T123 | writer disabled |
| F-30 | MCAL alpha-map decode correctness (client consumer identification, corpus sweeps) | 199 US1–US4 | absent |
| F-31 | Asset reference three-set comparison, candidates, chronology, repair, conversion survey | 155 US2–US6 | absent |

### C. Parked (spec-stated, explicitly deferred by their own authors)

| ID | Item | Source |
|---|---|---|
| F-90 | Cross-era rig comparison (0.5.3 High Elf vs Blood Elf) | 154 US4 |
| F-91 | Benilla 1.12.1 side-by-side oracle methodology | 193 T101–T303 |
| F-92 | DAT as project interchange format (operator-deferred) | 241 |

## Operator verification owed on shipped code (not implementation work)

- 235 SC-002/003/008: visible geometry + bbox fallback across 1.0.0–3.0.1; no-regression spot check.
- 205 T206/T304/T501/T502: real-client liquid confirmation.
- 237/247: load an exported LK ADT in a 3.3.5 client or Noggit; compose a DAT Cartography layer on
  screen (both answer whether DAT tile axes need `--transpose`).
- 238 SC-001/002: hash-match CASC reads against an independent extractor.
- Alpha2 shipped unverified: M2 texture-wrap fix (all M2 eras); `LkAdtWriter` chunk-completeness fixes (17 call sites).

## Constraints

AGENTS.md §4 reader freeze applies (MPQ/ADT/WMO/M2/MDX readers, `AlphaWdtWriter`); reader changes
need a verified format bug. New tooling layers above proven readers (§4 "Layer new tooling").
