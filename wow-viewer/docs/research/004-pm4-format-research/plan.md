# Research Plan: PM4 Format — Phase 1 Raw Analysis

**Branch**: `004-pm4-format-research` | **Date**: 2026-05-20 | **Spec**: `docs/research/004-pm4-format-research/spec.md`

**Input**: Spec with 16 research objectives, 9 user stories. User directive: run ALL analyzers first.

## Summary

Phase 1 executes every existing PM4 analyzer against the full 616-file development corpus, dumps raw outputs to disk, and captures baseline signals. No new analyzers, no code changes. This gives us raw data to guide deeper research in subsequent phases.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: xUnit (for integration tests), `WowViewer.Tool.Inspect` (CLI)

**Storage**: Analyzer outputs written to `wow-viewer/output/research/pm4-phase-1/`

**Testing**: `dotnet test` on `WowViewer.Core.PM4.Tests`

**Target Platform**: Windows, .NET 10

## Constitution Check

- Repo independence: ✅ No cross-repo references
- No H:\CLIENTS: ✅ Uses `wow-viewer/test_data/development/`
- Read-only reference: ✅ No writes to `gillijimproject_refactor`
- Real-data validation: ✅ 616-file staged corpus

## Project Structure

```text
wow-viewer/
├── src/core/WowViewer.Core.PM4/          # All analyzers live here
│   ├── Research/                          # Analyzer implementations
│   ├── Services/                          # Reader, placement math, correlation
│   └── Models/                            # Chunk models, report models
├── tests/WowViewer.Core.PM4.Tests/        # Integration tests
├── tools/inspect/WowViewer.Tool.Inspect/  # CLI entrypoints
├── test_data/development/                 # 616-file corpus
└── output/research/pm4-phase-1/           # Analyzer output dumps
```

## Implementation Phases

### Phase 1.1: Corpus Snapshot (existing tests)
Goal: Run all 27 existing integration tests, confirm they pass, capture baseline metrics.

Approach: `dotnet test` with verbosity, pipe output to file.

### Phase 1.2: CLI Analyzer Dumps
Goal: Run every `pm4` CLI command against the corpus, dump raw output to files.

Approach: Execute each `WowViewer.Tool.Inspect pm4 <command>` variant, capture stdout.

Commands:
1. `pm4 audit-directory -o <json>` — corpus-level decode audit
2. `pm4 linkage -o <json>` — cross-file linkage (RefIndex, MDOS, CK24 reuse)
3. `pm4 mshd -o <json>` — MSHD field profiling
4. `pm4 mscn -o <json>` — MSCN relationship analysis
5. `pm4 unknowns -o <json>` — unknowns exploration (families, modes, flags)
6. `pm4 hierarchy development_00_00 -o <json>` — object hierarchy for ref tile
7. `pm4 inspect development_00_00 -o <json>` — full analysis for ref tile
8. `pm4 export-json development_00_00 --ck24 0x43A9AA -o <json>` — CK24 forensics for top group

State: Run each command, save both stdout text and JSON output.
