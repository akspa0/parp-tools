# GillijimProject Next — Integration Plan

Status: Planning
Owner: GillijimProject (Next refactor)
Scope: Non-destructive; new work lives entirely under `next/`

## 1. Context and Goals

- Preserve the existing C# port (under `src/gillijimproject-csharp/`) as a stable reference.
- Build a new refactor under `next/` with:
  - A reusable Core class library
  - A thin CLI
  - Tests and documented architecture
- Integrate:
  - Warcraft.NET as the ADT v18 (WotLK) writer backend
  - DBCD as the AreaTable loader to translate Alpha (0.5.3) AreaIDs → LK (3.3.5) AreaIDs
- Add UniqueID analysis tooling and Markdown reporting.

References that informed this plan:
- FourCC rule: forward in-memory, reversed on disk (Chunk-based I/O).
- MFBO/MTXF must be included for parity (ensure MHDR offsets are correct).
- 1:1 parity target was reached; this Next refactor focuses on architecture, safety, and integrations.

## 2. Repository Layout (Next)

- `next/gillijimproject-next.sln`
- `next/src/GillijimProject.Next.Core/`
  - `Domain/` — domain types (Alpha/LK), value objects
  - `IO/` — readers/parsers for Alpha WDT/ADT
  - `Transform/` — Alpha → LK conversion pipeline
  - `Services/` — orchestration services (AreaIdTranslator, UniqueIdAnalyzer, ReportWriter)
  - `Adapters/WarcraftNet/` — ADT writer adapter(s)
  - `Adapters/Dbcd/` — DBCD access and helpers
- `next/src/GillijimProject.Next.Cli/`
  - `Commands/` — subcommands: convert, analyze, fix-areaids
  - `Program.cs` — command routing
- `next/tests/GillijimProject.Next.Tests/`
  - Unit, integration, and golden-file tests
  - Fixture bindings (skippable when fixtures absent)
- `next/docs/`
  - `plan.md` (this document)
  - `README.md` — quickstart and overview
  - `architecture.md` — solution structure and data flow
  - `adapters-warcraftnet.md` — ADT v18 writer mapping details
  - `areaid-mapping.md` — DBCD usage and overrides semantics
  - `cli.md` — commands, options, examples
  - `testing.md` — fixtures, golden outputs, validation
  - `roadmap.md` — milestones and forward plan

## 3. External Dependencies (Source References)

- Warcraft.NET
  - Path: `next/libs/Warcraft.NET/Warcraft.NET/Warcraft.NET.csproj`
  - Purpose: Provide the ADT v18 model and writer; ensure `MVER=18`, correct chunk ordering, offsets, and padding.
- DBCD
  - Path: `next/libs/wow.tools.local/DBCD/DBCD/DBCD.csproj`
  - Purpose: Load `AreaTable.dbc` definitions and data for Alpha (0.5.3) and LK (3.3.5).

Note: If TFMs mismatch with net9.0, either multi-target `GillijimProject.Next.Core` or isolate adapters into compatible assemblies.

## 4. Core Library API (Proposed)

Namespaces:
- `GillijimProject.Next.Core.Domain`
- `GillijimProject.Next.Core.IO`
- `GillijimProject.Next.Core.Transform`
- `GillijimProject.Next.Core.Services`
- `GillijimProject.Next.Core.Adapters.WarcraftNet`
- `GillijimProject.Next.Core.Adapters.Dbcd`

Public surface (illustrative):
- `AlphaReader.ParseWdt(string alphaWdtPath): WdtAlpha`
- `AlphaReader.ParseAdt(string alphaAdtPath): AdtAlpha`
- `AlphaToLkConverter.Convert(WdtAlpha, IEnumerable<AdtAlpha>, AreaIdTranslator): IEnumerable<AdtLk>`
- `IAdtWriter.Write(AdtLk adt, string outputPath)`
- `AreaIdTranslator.TryTranslate(int alphaAreaId, out int lkAreaId): bool`
- `UniqueIdAnalyzer.Analyze(AdtLk adt): UniqueIdReport`
- `ReportWriter.WriteMarkdown(UniqueIdReport, string outputDir): string`

Key invariants:
- FourCC forward in memory; reversed on disk (via writer).
- Omit `MH2O` when empty.
- Write `MCLQ` last within `MCNK`.
- Include `MFBO` and `MTXF` when present; compute and set MHDR offsets consistently with LK expectations.

## 5. Warcraft.NET Adapter

`WarcraftNetAdtWriter` responsibilities:
- Map [AdtLk](cci:1://file:///i:/parp-tools/pm4next-branch/parp-tools/gillijimproject_refactor/src/gillijimproject-csharp/WowFiles/LichKing/AdtLk.cs:34:4-118:5) to Warcraft.NET ADT v18 model (ensure `MVER=18`).
- Enforce chunk ordering and compute offsets and sizes, including:
  - Proper handling of MMID/MWID (no invalid trailing offsets).
  - Correct serialization of `MCNK` sub-chunks and padding.
  - `MFBO/MTXF` presence and offset updates in `MHDR` (when present).
- Output should pass 010 Editor template validation.

Validation:
- Golden outputs compared against known-good samples.
- 010 template checks (manual or scripted in tests).

## 6. AreaID Translation (DBCD)

Inputs:
- `AreaTable.dbc` for Alpha (0.5.3) and LK (3.3.5).
- Optional overrides file (JSON).

Strategy:
- Primary: join on stable fields (e.g., English name), optionally parent or hierarchy when available.
- Fallback heuristics: normalized name matching, continent/zone linkage.
- Conflicts: record ambiguity, apply overrides for deterministic mapping.

Overrides JSON schema (example):
```json
{
  "schemaVersion": 1,
  "mappings": [
    { "alphaId": 123, "lkId": 456, "reason": "Manual disambiguation: name collision" }
  ],
  "denyList": [ 999 ],
  "notes": "Optional freeform notes."
}

Output:

Every MCNK.AreaId is a valid LK ID when possible.
Unresolved entries are reported with suggested next actions.

7. UniqueID Analysis
Scope:

Scan MDDF / MODF entries for unique IDs, referenced models (M2/WMOs).
Detect duplicates, collisions, suspicious gaps (optional).
Verify presence of referenced assets from one or more roots.
Markdown report (outline):

Summary counts (by ADT and overall)
Missing assets
Duplicate/colliding IDs
Recommendations and next steps

8. CLI Commands
convert
--wdt-alpha <path> (required)
--out <dir> (required)
--dbc-alpha <path> --dbc-wotlk <path> (required)
--areaid-overrides <path> (optional)
Behavior: parse Alpha → convert with AreaID translation → write LK ADTs via Warcraft.NET
analyze
--input <dir> (ADT directory)
--assets-root <dir> (repeatable)
--report-out <dir>
Behavior: UniqueID + asset presence analysis; generate Markdown
fix-areaids
DBCD inputs same as convert
--dry-run
Behavior: preview or re-emit ADTs with corrected AreaIDs

9. Testing Strategy
Unit tests for:
AreaIdTranslator (with synthetic DBCD fixtures)
UniqueIdAnalyzer (synthetic ADTs)
Integration tests for:
Alpha → LK pipeline producing ADT v18 outputs
Writer validations (offsets, chunk ordering, MFBO/MTXF)
Golden-file tests:
Compare byte-equal outputs for small fixtures
Validation with 010 templates (documented process in testing.md)

10. Risks and Mitigations
TFM mismatches (Warcraft.NET/DBCD): multi-target Core or isolate adapters.
DBC schema drift: leverage DBCD definitions and maintain overrides JSON.
Performance on large assets: profile after functional parity; optimize Span<T>, allocations, and I/O patterns.

11. Milestones
M0: Scaffold Next solution structure (Core/CLI/Tests; docs).
M1: Core skeleton (IO/Transform/Services/Adapters) builds.
M2: DBCD integration; AreaID translation produces stable mappings (with overrides).
M3: Warcraft.NET writing; ADT v18 outputs validated by 010.
M4: CLI commands functional (convert/analyze/fix-areaids).
M5: UniqueID analysis report with sample data.
M6: Perf validation on larger datasets; tune hot paths.

12. Acceptance Criteria
ADT v18 outputs pass 010 validation; MVER=18, correct chunk ordering and offsets, no invalid trailing indices in MMID/MWID.
AreaIDs are valid LK IDs, with unresolved cases clearly reported.
Analyzer generates Markdown reports that identify missing assets and ID issues.
CLI experience is documented and stable.

13. Implementation Steps (High-Level)
Scaffold next/ solution and projects (net9.0).
Add source references to:
- next/libs/Warcraft.NET/Warcraft.NET/Warcraft.NET.csproj
- next/libs/wow.tools.local/DBCD/DBCD/DBCD.csproj
Add Core interfaces and stubs:
IAdtWriter, WarcraftNetAdtWriter
DbcdAreaTableProvider, AreaIdTranslator
AlphaReader, AlphaToLkConverter
UniqueIdAnalyzer, ReportWriter
Implement CLI commands and wire DI where applicable.
Add tests and fixtures; establish golden-file outputs.
Author accompanying docs in next/docs/.
Validate with 010; iterate.

14. Documentation Deliverables
docs/README.md — What this “Next” refactor is and how to run it
docs/architecture.md — Components, namespaces, data flow, I/O boundaries
docs/adapters-warcraftnet.md — Mapping strategy, chunk invariants
docs/areaid-mapping.md — DBCD usage, overrides JSON, resolution workflow
docs/cli.md — Command reference, examples
docs/testing.md — How to run tests, fixtures, and 010 validation
docs/roadmap.md — Milestones and future directions
