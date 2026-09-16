# Implementation Plan: ADT/v22 Terrain Reading and Rendering

**Branch**: `v0.5.4-dev` (v0.6 release line) | **Release**: v0.6 | **Date**: 2026-09-16 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/237-adt-v22-terrain/spec.md`

> The `setup-plan.ps1` script refuses non-numbered branches, so this plan was authored from the
> template by hand. To run speckit scripts against this spec without touching `.specify/feature.json`,
> set `$env:SPECIFY_FEATURE_DIRECTORY = 'specs/237-adt-v22-terrain'`.

## Summary

Add a measured, evidence-first reader for the `AHDR`-family terrain tiles (ADT/v22, and v23 through the
same code), expose it through the inspect CLI, and render it in the viewer via a new `ITerrainAdapter`.

The work is ordered so nothing is built on unverified layout. **Phase 0** fixes version detection and
builds a byte-accounting inventory of the real corpus. **Phase 1** decodes each channel, and every
ambiguous interpretation (vertex order, normal component order, height frame, alpha encoding,
placement frame, `ACDO` record size) is settled by a detector that is first proven able to tell the
candidates apart. **Phase 2** maps the decoded tile into the viewer's existing `TerrainChunkData` /
`TileLoadResult` shape, so the unchanged terrain renderer draws it.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: existing `WowViewer.Core.IO` chunk infrastructure (`ChunkedFileReader`, `MapFileSummaryReader`, `MapSummaryReaderCommon`), `AdtMcalDecoder` (called, not modified), viewer `ITerrainAdapter` / `TerrainChunkData` / `TerrainRenderer`, Silk.NET.OpenGL

**Storage**: Loose files on disk (read-only), in `wow-viewer/test_data/v22_adts/` by repo convention (like `test_data/0.5.3/`). The CLI takes `--root`. Tests resolve `GetWowViewerRoot()/test_data/v22_adts`, which `WOWVIEWER_AHDR_CORPUS` can override, and skip when it is absent.

**Testing**: xUnit in `tests/WowViewer.Core.Tests`: synthetic-buffer unit tests, plus real-corpus tests that skip when the corpus root is not configured

**Target Platform**: Windows desktop (viewer), cross-platform core library and CLI

**Project Type**: Library + CLI + desktop viewer

**Performance Goals**: A tile decodes fast enough not to be felt during viewer streaming (target: at or below the existing LK ADT tile load time on the same machine). The corpus inventory is a batch job with no interactive target.

**Constraints**: No change to existing MCAL decode/alpha packing (Terrain Alpha Risk Area); no change to existing format detection outcomes except the `AHDR` branch; no writer

**Scale/Scope**: One tile format family, about 10 chunk types. Corpus size is unknown until Phase 0 inventory.

**Open inputs (operator)**:
- Corpus: **on hand**, 700 files (699 unique) in `wow-viewer/test_data/v22_adts/unknown/` (git-ignored). No WDT, map table or listfile names exist for them. Extensionless FileDataID names. **Revision 26** (MVER 26 + AHDR). See evidence/phase0-first-look-2026-09-16.md
- **Standalone**: a never-before-seen engine version. There are no external companions or lookups, so the tile files are the only source (research R10). No dependency on Specs 238/239.

**Dependency order**: Fast path F first, then Phase 0 → 1 → 2. Everything runs on the corpus already on disk; nothing waits on other specs.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|---|---|---|
| I. Repo independence | PASS | All code lives under `wow-viewer/src/core`, `tools/inspect`, `src/viewer`. |
| II. Library-first | PASS | Reader and inventory live in `WowViewer.Core.IO/Maps`; the CLI and viewer adapter are thin consumers. The existing `AdtV23SummaryReader` is extended, not duplicated. |
| III. Real-data validation | PASS (gated) | Every phase gate is measured on the real corpus with command, root, and hashes recorded under `evidence/`. Synthetic tests are not sign-off. |
| IV. Model architecture | N/A | No model. |
| V. Streaming dataset pipeline | N/A | No dataset emission in scope. A later datastore ingest would follow "Python owns the datastore". |
| VI. No client path assumptions | PASS | The corpus root is a CLI argument / viewer folder pick. Tests use `test_data/v22_adts` with an env override. |
| VII. Containers are inputs | PASS | Read-only; no writer; no container output. |
| Format reader ownership | PASS | No existing reader decodes AHDR payloads (verified: `AdtV23SummaryReader` reads the header only). |
| Terrain alpha risk area | PASS | `AdtMcalDecoder` is called with the encoding inferred per map; no MCAL, edge-fix or shader-blend code is modified. |
| One phase at a time / bite-sized | PASS | 3 phases, each 10 steps or fewer, each ending in a real-data gate. |
| Spec docs source of truth | PASS | The measured format layout is written to `docs/architecture/adt-v22-format.md` in Phase 1. |

**Post-design re-check**: PASS. The design adds no project, no new dependency and no container or
writer surface.

## Project Structure

### Documentation (this feature)

```text
specs/237-adt-v22-terrain/
├── spec.md
├── plan.md              # this file
├── research.md          # decisions + the measurement each one needs
├── data-model.md        # typed model + viewer mapping
├── quickstart.md        # operator commands (PowerShell)
├── contracts/
│   └── cli-contract.md  # inspect CLI surface
├── checklists/
│   └── requirements.md
└── evidence/            # Phase gate receipts (created during implementation)
```

### Source Code

```text
src/core/WowViewer.Core/
├── Files/WowFileKind.cs                  # + AdtV22, AdtV22Error, AdtV26, AdtAhdrUnknownVersion
├── Maps/MapFileKind.cs                   # mirror kinds; update the ADT-family predicate
├── Maps/MapChunkIds.cs                   # + Alyr, Amap, Ashd, Acdo, Aloc, Aoch, Adst
└── Maps/AdtAhdr/                         # NEW: typed model (see data-model.md)
    ├── AdtAhdrTile.cs
    ├── AdtAhdrChunk.cs
    ├── AdtAhdrLayer.cs
    ├── AdtAhdrPlacement.cs
    └── AdtAhdrInventory.cs

src/core/WowViewer.Core.IO/
├── Files/WowFileDetector.cs              # AHDR branch reads version
├── Maps/MapFileSummaryReader.cs          # kind mapping
├── Maps/AdtV23SummaryReader.cs           # accept v22 kinds (becomes AHDR summary)
├── Maps/AdtAhdrInventoryReader.cs        # NEW: byte-accounting chunk walk
├── Maps/AdtAhdrReader.cs                 # NEW: full decode, never throws on malformed
└── Maps/AdtAhdrTileSlicer.cs             # NEW: whole-tile grids -> per-chunk 145-vertex order

tools/inspect/WowViewer.Tool.Inspect/Program.cs   # adt-ahdr inventory|dump|layout-probe
docs/CLI-TOOLS.md                                 # document the above (diffed against the real parser)
docs/architecture/adt-v22-format.md               # NEW: measured layout, the source of truth

src/viewer/WoWViewer/Terrain/
├── AhdrTerrainAdapter.cs                 # NEW: ITerrainAdapter over a loose folder
└── (ViewerApp.cs)                        # open-folder entry point, beside the Rosetta datastore open

tests/WowViewer.Core.Tests/
├── WowFileDetectorTests.cs               # version-correct AHDR tests replace "all AHDR = v23"
├── AdtAhdrInventoryReaderTests.cs        # synthetic
├── AdtAhdrReaderTests.cs                 # synthetic, including malformed inputs
├── AdtAhdrTileSlicerTests.cs             # synthetic, known-answer grid mapping
└── AdtAhdrRealDataTests.cs               # uses test_data/v22_adts (or WOWVIEWER_AHDR_CORPUS); skips when absent
```

**Structure Decision**: The code is named for the `AHDR` family rather than "V22", because the repo
already overloads "V22" for an unrelated dataset lane (`V22Enrich`, `V22ModelPayload`, specs 086–088).
`AdtV23SummaryReader` keeps its name to avoid churning its callers; its guard widens to accept all
`AHDR` kinds.

## Phases

### Fast path F: wireframe first look (US0; FR-017), runs before Phase 1

Only facts already measured in `evidence/phase0-first-look-2026-09-16.md` are used. The phase gates below still apply to the full decoder.

1. Detection: MVER-then-AHDR recognition and `AdtV26` kind (Phase 0 steps 1–3, pulled forward).
2. Minimal reader: `AHDR` dims, `ALOC` X/Y, `AVTX` outer+inner float arrays, and nothing else. It never throws, and uses a diagnostic on short chunks.
3. Slicer: outer row-major + inner second block → per-chunk 145-entry 9-8-9 heights (the Phase 2 step 1 slicer, pulled forward). Normals are computed from heights, not read from `ANRM`.
4. `AhdrTerrainAdapter` minimal: content-sniff the folder, place tiles by `ALOC`, empty layers/placements, and a "provisional" label.
5. Viewer entry "Open AHDR terrain folder…" rendered through the existing terrain wireframe mode.
6. **Gate**: `evidence/fastpath-wireframe.md` with screenshots showing SC-009 (no cracks on ALOC-adjacent edges), and a note on whether the inner grid looks right.

Each phase ends with a real-corpus gate recorded in `evidence/`. Per the constitution, a phase is
not started until the previous gate passes.

### Phase 0: Detection + corpus inventory (US1, US4; FR-001–003)

1. Add `AdtV22`, `AdtV22Error`, `AdtV26`, `AdtAhdrUnknownVersion` to `WowFileKind` and `MapFileKind`. Grep every `AdtV23` switch/predicate site and extend each explicitly.
2. `WowFileDetector`: recognize `AHDR` as the first chunk **or** the chunk right after `MVER` (observed revision 26), content-only with no filename or extension reliance. Pick the kind from `AHDR.version` (22, 23, 26, else unknown), keeping the `.error` split.
3. Replace the synthetic detector/summary tests that assert v23 for every AHDR file with version-parametrized tests (22, 23, 99).
4. Widen `AdtV23SummaryReader`'s guard to all AHDR kinds; add v22 to the inspect `map` output.
5. `AdtAhdrInventoryReader`: recursive chunk walk that accounts for every byte. Top-level chunks, `ACNK` header plus nested chunks, `ALYR` fixed part plus nested `AMAP`. Record id, offset, size, parent, and gaps/overruns. Probe both padded and unpadded sub-chunk walks, and report which one accounts for all bytes.
6. Documented-size table (AHDR 0x40, ALYR ≥0x20, ASHD 0x200, ACDO 0x38, AFBO 0x48, AVTX/ANRM derived from header) with per-chunk disagreement reporting.
7. CLI `adt-ahdr inventory <root>`: per-file line plus corpus aggregate (versions, chunk-occurrence table, unknown chunks, size disagreements, unaccounted bytes, failed files), with JSON output.
8. Synthetic tests: unknown chunk surfaced, truncated file flagged, gap detection.
9. **Gate**: run the inventory on the real corpus and write `evidence/phase0-corpus-inventory.md` (command, root, file count, SHA-256 of the file list, version histogram, full chunk table, every disagreement). **SC-001 and SC-002 must hold.** Any unknown chunk becomes a named Phase 1 research item before Phase 1 begins.

### Phase 1: Decode + layout resolution (US2; FR-004–011)

1. `AdtAhdrReader` skeleton: builds `AdtAhdrTile` from the inventory walk. Per-channel `Diagnostics`, never throws (FR-010).
2. Name tables: `ATEX` and `ADOO`. Measure first whether each is one chunk of NUL-separated names or one chunk per name, since the existing summary counts ATEX *chunks*.
3. `AVTX` and `ANRM` raw decode into outer/inner arrays sized from the header.
4. **Layout probe** (`adt-ahdr layout-probe`): score each candidate vertex order (row/column transpose × flip) by cross-tile seam agreement, and each ANRM component permutation × sign by agreement with height-derived normals. **Detector power check first**: deliberately permute a correct-candidate grid and show the score separates it. Record the winner with its margin.
5. `ACNK` header decode (v22 layout vs v23 layout, gated by version), plus `ALYR`, `ASHD` (reuse MCSH 1-bit expansion semantics) and `ACDO` raw records. Measure `ACDO` record size across the corpus before fixing the trailing fields.
6. Alpha: infer each map's encoding from payload size and layer flags. Decode through `AdtMcalDecoder`, then validate that layer weights stay in range and that non-base alpha masks are not all zero or all 255 at implausible rates.
7. Placement frame probe: for each candidate position frame (raw, origin-minus as MDDF, axis swaps), measure placement Z against decoded terrain height at the placement XY. The winner is the frame where doodads sit on the ground. Rotation units are checked the same way against WMO bounds where available.
8. v23-only `AFBO` and `ACVT` decode (validated only if real v23 files exist; otherwise synthetic only and marked so).
9. `adt-ahdr dump <file>`: human and JSON dump (FR-011). Write `docs/architecture/adt-v22-format.md` from the measured results; update `docs/CLI-TOOLS.md` and diff it against the real argument parser.
10. **Gate**: `evidence/phase1-decode.md` must show **SC-002–SC-006** met, with every probe's candidate scores and margins.

### Phase 2: Viewer rendering (US3; FR-012–014)

1. `AdtAhdrTileSlicer`: whole-tile outer/inner grids become per-chunk 145-entry interleaved heights/normals, using the Phase 1 winning order. Known-answer tests use an indexed ramp.
2. Resolve height frame (absolute vs chunk-relative) from the Phase 1 evidence, and set `TerrainChunkData.WorldPosition` from tile/chunk grid math shared with the existing adapters.
3. `AhdrTerrainAdapter` tile discovery: content-sniff every file in the folder (any name or extension), and place each tile by `ALOC[1]` = X, `ALOC[2]` = Y (measured). Flag missing `ALOC` and duplicate tiles. There is no WDT; the folder of tiles is the whole map.
4. `LoadTileWithPlacements`: fill `TerrainChunkData` (heights, normals, layers → `TileTextures` indices, 64x64 alpha, shadow, area id). Map `ACDO` to `MddfPlacement` or `ModfPlacement` by the referenced name's extension, using the Phase 1 frame.
5. Per-tile failure isolation: a failed tile is logged and surfaced in the tile list; the rest of the map loads (US3 scenario 4).
6. Viewer entry point: "Open ADT/v22 folder…" beside the Rosetta datastore open, wired through `TerrainManager`. Phasing, placement writing and cartography members return the documented "unsupported" values.
7. No external asset lookup: textures render as per-layer flat colours or checker (layer index visible), and placements render as markers labelled with the `ADOO` name. Real asset loading is a later, separate decision, not part of this spec.
8. Regression pass (US4): open one Alpha 0.5.3 map and one LK map; confirm the existing test suite is green (SC-008).
9. **Gate**: `evidence/phase2-render.md`. The operator loads the corpus and confirms seams, textures and placements by eye (SC-007), with screenshots saved to evidence.

## Complexity Tracking

No constitution violations.
