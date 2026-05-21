# PM4 Phase 1: Raw Analysis — Task Breakdown

**Plan**: `plan.md` | **State**: `tasks.md`

## Prerequisites

- [x] `wow-viewer/test_data/original_development/World/Maps/development/` exists (616 PM4 files)
- [x] Analyzer code exists (6 directory-wide + 2 single-file + 1 forensics)
- [x] `WowViewer.Tool.Inspect` CLI exists with `pm4` commands

---

## Task 1.1 — Build and Test Baseline [P]

**Goal**: Verify the solution builds clean and all 27 integration tests pass.

**Command**:
```
dotnet test I:\parp\parp-tools\wow-viewer\WowViewer.slnx -c Debug --filter "FullyQualifiedName~Pm4ResearchIntegrationTests" --verbosity normal 2>&1 | Tee-Object -FilePath "I:\parp\parp-tools\wow-viewer\output\research\pm4-phase-1\test-baseline.log"
```

**Validation**: Exit code 0. All 27 tests pass. Log file captured.

---

## Task 1.2 — Corpus Audit (audit-directory) [P]

**Goal**: Get complete decode audit for all 616 files — chunk counts, stride violations, trailing bytes.

**Command**:
```
dotnet run --project I:\parp\parp-tools\wow-viewer\tools\inspect\WowViewer.Tool.Inspect -- pm4 audit-directory -o I:\parp\parp-tools\wow-viewer\output\research\pm4-phase-1\corpus-audit.json I:\parp\parp-tools\wow-viewer\test_data\original_development\World\Maps\development\
```

**Validation**: JSON output file created with corpus-level chunk counts and summary.

---

## Task 1.3 — Linkage Analysis (linkage) [P]

**Goal**: Cross-file linkage: RefIndex mismatches, MDOS integrity, CK24 reuse across tiles.

**Command**:
```
dotnet run --project I:\parp\parp-tools\wow-viewer\tools\inspect\WowViewer.Tool.Inspect -- pm4 linkage -o I:\parp\parp-tools\wow-viewer\output\research\pm4-phase-1\linkage.json I:\parp\parp-tools\wow-viewer\test_data\original_development\World\Maps\development\
```

**Validation**: JSON output with mismatch family clustering, reuse counts, CK24 cross-tile stats.

---

## Task 1.4 — MSHD Header Analysis (mshd) [P]

**Goal**: Profile all 8 MSHD fields across 616 files — find non-zero values, correlations, frequencies.

**Command**:
```
dotnet run --project I:\parp\parp-tools\wow-viewer\tools\inspect\WowViewer.Tool.Inspect -- pm4 mshd -o I:\parp\parp-tools\wow-viewer\output\research\pm4-phase-1\mshd.json I:\parp\parp-tools\wow-viewer\test_data\original_development\World\Maps\development\
```

**Validation**: JSON output with field distributions, relationship buckets, correlation table.

---

## Task 1.5 — MSCN Coordinate Analysis (mscn) [P]

**Goal**: MSCN coordinate space, swapped-vs-raw dominance, CK24-vs-MSCN overlap, tile-local alignment.

**Command**:
```
dotnet run --project I:\parp\parp-tools\wow-viewer\tools\inspect\WowViewer.Tool.Inspect -- pm4 mscn -o I:\parp\parp-tools\wow-viewer\output\research\pm4-phase-1\mscn.json I:\parp\parp-tools\wow-viewer\test_data\original_development\World\Maps\development\
```

**Validation**: JSON output with coordinate mode stats, overlap ratios, invalid-MDOS clusters.

---

## Task 1.6 — Unknowns Exploration (unknowns) [P]

**Goal**: MSLK/MSUR family distributions, MSPI mode classification, MSLK type/subtype/system, MSUR attribute/group/ck24 distributions.

**Command**:
```
dotnet run --project I:\parp\parp-tools\wow-viewer\tools\inspect\WowViewer.Tool.Inspect -- pm4 unknowns -o I:\parp\parp-tools\wow-viewer\output\research\pm4-phase-1\unknowns.json I:\parp\parp-tools\wow-viewer\test_data\original_development\World\Maps\development\
```

**Validation**: JSON output with all family distributions, MSPI mode counts, rare value signals.

---

## Task 1.7 — Hierarchy Analysis (hierarchy on ref tile) [P]

**Goal**: Object hierarchy hypothesis for development_00_00 — split families, shared placement, link groups.

**Command**:
```
dotnet run --project I:\parp\parp-tools\wow-viewer\tools\inspect\WowViewer.Tool.Inspect -- pm4 hierarchy -o I:\parp\parp-tools\wow-viewer\output\research\pm4-phase-1\hierarchy-ref-tile.json I:\parp\parp-tools\wow-viewer\test_data\original_development\World\Maps\development\development_00_00.pm4
```

**Validation**: JSON output with tile object hypothesis report, split family comparisons.

---

## Task 1.8 — CK24 Forensics (top CK24 group on ref tile) [P]

**Goal**: Deep-dive into CK24=0x43A9AA (896 surfaces, largest group in ref tile).

**Command**:
```
dotnet run --project I:\parp\parp-tools\wow-viewer\tools\inspect\WowViewer.Tool.Inspect -- pm4 export-json --ck24 0x43A9AA -o I:\parp\parp-tools\wow-viewer\output\research\pm4-phase-1\forensics-ck24-43A9AA.json I:\parp\parp-tools\wow-viewer\test_data\original_development\World\Maps\development\development_00_00.pm4
```

**Validation**: JSON output with link group reports, MPRL rows, heading evidence, placement comparisons.

---

## Task 2 — Compile Research Notes [S]

**Goal**: Consolidate all analyzer outputs into updated research-notes.md with new findings.

**Approach**: Read each JSON output file, extract key signals, add to research-notes.md sections.

**Validation**: research-notes.md updated with Phase 1 findings, new unknowns identified, corpus-level statistics confirmed or updated.
