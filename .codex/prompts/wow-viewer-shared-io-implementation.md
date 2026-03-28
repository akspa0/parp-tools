---
description: "Implement or plan the next narrow wow-viewer shared I/O slice using the library-first format-ownership workflow. Use when extracting ADT root, _tex0.adt, _obj0.adt, _lod.adt, WDT, WMO, BLP, DBC, DB2, file detector, chunk reader, file-summary contract, map inspect, converter detect, or shared-format regression work without drifting into a broad migration debate."
name: "wow-viewer Shared I/O Implementation"
argument-hint: "Optional file family, detector seam, reader seam, tool consumer, or regression slice to prioritize"
agent: "codex"
---

Implement the next narrow `wow-viewer` shared-format or shared-I/O slice without losing the current ownership and validation rules.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/plans/wow_viewer_shared_io_library_plan_2026-03-26.md`
4. `wow-viewer/README.md`
5. `AGENTS.md`

## Goal

Move one more reusable non-PM4 format capability into `wow-viewer/src/core/WowViewer.Core` or `wow-viewer/src/core/WowViewer.Core.IO` with explicit validation and without reintroducing tool-local parser ownership. Typical slices include ADT root or split ADT family seams, WDT summaries, WMO top-level readers, or early BLP or DBC or DB2 detection or summary work.

## Current Working Rules

- `WowViewer.Core` and `WowViewer.Core.IO` are the intended shared home for non-PM4 file detection, chunk reading, and future format ownership.
- `WowViewer.Tool.Inspect` and `WowViewer.Tool.Converter` should stay thin consumers of those shared seams.
- The default direction is library-first `wow-viewer` work, not broader `MdxViewer` integration.
- Shared detection or summary behavior is real progress, but it is not the same thing as deep payload parsing or write support.

## Current WMO Continuation Note

- The current validated WMO root-light seam is standard and Alpha `MOLT` semantic summary in shared `Core.IO`, with real proof for Alpha `ironforge.wmo.MPQ` and standard `0.6.0` `world/wmo/khazmodan/cities/ironforge/ironforge.wmo`.
- Already-settled facts for the standard 48-byte layout: bytes `2..3` are a raw non-zero `headerFlagsWord` on real Ironforge (`0x0101`), offsets `24..39` are quaternion rotation, and offsets `40` and `44` are attenuation start and end.
- If the user says to continue this work without naming a narrower seam, default to one of these next WMO follow-ups instead of revisiting resolved offsets:
	1. prove whether `headerFlagsWord` varies across additional real standard roots
	2. add a per-light inspect dump for standard `MOLT` entries
- Do not claim per-bit meaning for `headerFlagsWord` until more than Ironforge has been checked.

## Active Shared Surface To Build On

- `FourCC`, `ChunkHeader`, `ChunkHeaderReader`
- `ChunkedFileReader`
- map contracts: `MapChunkIds`, `MapFileKind`, `MapChunkLocation`, `MapFileSummary`
- `MapFileSummaryReader`
- file detection contracts: `WowFileKind`, `WowFileDetection`
- `WowFileDetector`
- `WowViewer.Tool.Inspect` verb `map inspect`
- `WowViewer.Tool.Converter` verb `detect`
- regression floor in `wow-viewer/tests/WowViewer.Core.Tests`

## Non-Negotiable Constraints

- Do not turn a narrow shared-I/O slice into a broad migration-plan rewrite.
- Do not add new tool-local heuristics when the seam should live in `Core` or `Core.IO`.
- Do not claim full parsing or write support when the slice only proves detection or top-level summary.
- Prefer one concrete shared seam with proof over several partial abstractions.

## What The Work Must Produce

1. The exact seam to add or change.
2. The files that should own it in `wow-viewer`.
3. The validation required for that seam.
4. Any current-production touch point that remains reference-only.
5. The continuity files that must be updated after the slice lands.

## Deliverables

Return all items:

1. the next shared I/O slice to implement
2. why that slice is the right next step
3. exact files to change
4. exact validation to run
5. what should stay out of scope for this slice
6. which memory or prompt surfaces must be updated afterward

## First Output

Start with:

1. the current shared I/O boundary you are assuming
2. the single next seam you would extract or add
3. the narrowest proof that would show the slice is real
4. what you are explicitly not claiming yet