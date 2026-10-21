# Implementation Plan: Offline WoWMuseum ADT Patcher

**Spec**: [spec.md](spec.md) · **Branch**: `281-museum-adt-patcher`

**Status**: Plan authored 2026-10-21. No implementation started.

## 1. Context

The viewer currently carries PM4 object matching as interactive tooling: the PM4 workbench
Correlation, Reconcile, and Alignment tabs, the `Pm4ObjectMatchStore`, and the manual match
curation in the Selection tab. The operator's direction is that matching is a **batch job that
patches ADTs**, not an interactive viewer feature. The data needed to match properly (Museum ADTs
plus the original PM4s) is expensive to assemble, so it belongs in a repeatable offline tool.

The building blocks already exist in the core libraries:

| Need | Existing service |
|---|---|
| Build PM4 object segments | `Pm4ObjectSegmentBuilder.Build(pm4Path)` |
| Read Museum placements | `AdtPlacementReader.Read(adtPath)` |
| Score segments against placements | `Pm4AssetMatchScorer.ScoreSegments` |
| Build guide observations and snapshots | `Pm4ReconciliationInputAdapter` |
| Build proposals | `Pm4ReconciliationEngine.BuildTileProposals` |
| Apply accepted proposals | `ReconciliationApplyService.ApplyAccepted` |
| Write 3.3.5 LK ADT | `LkAdtWriter.Write` / `LkAdtWriter.Build` |
| Tile coordinates | `Pm4CoordinateService.TryParseTileCoordinates` |

The offline tool is a thin CLI over this pipeline. It does not reimplement matching.

## 2. Architecture decisions

### AD-1 — New CLI project, not a viewer mode

The tool lives at `wow-viewer/tools/museum-patch/WowViewer.Tool.MuseumPatch/` and references
`WowViewer.Core.PM4`, `WowViewer.Core.Editor`, and `WowViewer.Core.IO`. It has no viewer, no
Silk.NET, no GPU, and no ImGui dependency.

**Rationale**: the operator wants an offline tool. A viewer mode would keep the GPU and UI
dependencies and would not be scriptable.

### AD-2 — Two-phase run: dry-run then apply

Phase 1 (`--dry-run`) writes a review file and no ADTs. Phase 2 (`--apply --review <file>`) reads
the review file and writes only accepted proposals. A run without `--review` treats every
proposal as rejected.

**Rationale**: the matching is heuristic. The operator must inspect and correct it before any ADT
is written.

### AD-3 — Output is a patch, never a distribution

The tool writes only to `--out`. It refuses to write outside that folder and refuses to overwrite
an input. The repository and release archive contain no `.adt`, `.pm4`, or `.pd4` data.

**Rationale**: the operator's explicit constraint. This is enforced in code, not in documentation.

### AD-4 — Deterministic output

Proposals are sorted by a stable key (tile, PM4 object id, proposal id). The review file is
canonical JSON. The output ADT is built from the same inputs in the same order. Two runs on the
same inputs produce identical hashes.

**Rationale**: reproducibility is what makes the patches reviewable.

### AD-5 — Provenance is a first-class output

Every patched tile gets a `.provenance.json` beside it. It records the input hashes, every
proposal (source, score, decision), and the output hash. The report is written even when zero
proposals are accepted.

**Rationale**: the operator must be able to audit every decision.

### AD-6 — Reuse the core pipeline, do not fork it

The tool calls the same services the viewer's reconciliation panel calls. If a service needs a
change to be usable offline, the change goes in the core library and the viewer keeps working.

**Rationale**: two matchers would drift. One pipeline, two front ends.

## 3. CLI surface

```text
WowViewer.Tool.MuseumPatch \
  --museum <folder> \
  --pm4 <folder> \
  --out <folder> \
  [--map <name>] \
  [--dry-run] \
  [--review <file>] \
  [--overwrite] \
  [--min-score <0..1>] \
  [--jobs <n>] \
  [--verbose]
```

| Flag | Meaning |
|---|---|
| `--museum` | Folder of 3.3.5 LK v18 Museum ADTs. Required. |
| `--pm4` | Folder of original PM4/PD4 files. Required. |
| `--out` | Output folder. Required. Must not be an input folder. |
| `--map` | Optional map name filter; default is every paired tile. |
| `--dry-run` | Write the review file and no ADTs. |
| `--review` | Apply only the decisions in this review file. |
| `--overwrite` | Allow replacing existing files in `--out`. |
| `--min-score` | Reject proposals below this score. Default 0.0. |
| `--jobs` | Parallel tile workers. Default 1. |
| `--verbose` | Per-proposal console output. |

Exit codes: `0` all paired tiles succeeded; `1` at least one tile failed; `2` invalid arguments or
unsafe paths.

## 4. Pipeline

```text
1. Discover
   - Enumerate Museum ADTs: <map>_<y>_<x>.adt and <map>_<y>_<x>_obj0.adt
   - Enumerate PM4s: <map>_<y>_<x>.pm4 / .pd4
   - Pair by (tileX, tileY); report paired and unpaired.

2. Per tile
   a. Parse PM4 -> Pm4BuiltObjectSegment[]
   b. Read Museum ADT -> AdtPlacementCatalog
   c. Build self-corpus references
   d. Score segments -> Pm4SegmentMatchResult[]
   e. Build guide observations, placement snapshots, candidates
   f. Build proposals -> ReconciliationProposal[]
   g. Sort proposals by stable key

3. Dry run
   - Write review.json: every proposal with id, source, score, status, default decision.
   - Write no ADTs.

4. Apply
   - Read review.json; map decisions by proposal id.
   - Reject proposals with no decision.
   - Apply accepted proposals to the Museum ADT bytes.
   - Write <map>_<y>_<x>.adt to --out.
   - Write <map>_<y>_<x>.provenance.json.

5. Report
   - Console summary: paired, unpaired, accepted, rejected, conflicts, failures.
   - Exit code.
```

## 5. Output layout

```text
<out>/
  <map>_<y>_<x>.adt
  <map>_<y>_<x>.provenance.json
  review.json                 (dry-run only)
  run-summary.json
```

`review.json`:

```json
{
  "schema": "museum-patch-review/v1",
  "inputs": { "museum": "...", "pm4": "..." },
  "tiles": [
    {
      "tileX": 32, "tileY": 48,
      "proposals": [
        {
          "proposalId": "…",
          "action": "Align|Substitute|Clone",
          "status": "ReviewRequired|Conflict|AlreadyAligned",
          "score": 0.87,
          "source": "…",
          "decision": "Accept|Reject|Undecided"
        }
      ]
    }
  ]
}
```

`provenance.json`:

```json
{
  "schema": "museum-patch-provenance/v1",
  "tileX": 32, "tileY": 48,
  "inputs": { "museumSha256": "…", "pm4Sha256": "…" },
  "outputSha256": "…",
  "proposals": [ { "proposalId": "…", "decision": "Accept", "score": 0.87 } ]
}
```

## 6. File-level change map

### New

| File | Purpose |
|---|---|
| `tools/museum-patch/WowViewer.Tool.MuseumPatch/WowViewer.Tool.MuseumPatch.csproj` | CLI project |
| `tools/museum-patch/WowViewer.Tool.MuseumPatch/Program.cs` | Argument parsing and exit codes |
| `tools/museum-patch/WowViewer.Tool.MuseumPatch/MuseumPatchRunner.cs` | Discovery, pairing, per-tile pipeline |
| `tools/museum-patch/WowViewer.Tool.MuseumPatch/ReviewFile.cs` | Review read/write, canonical JSON |
| `tools/museum-patch/WowViewer.Tool.MuseumPatch/ProvenanceReport.cs` | Provenance read/write |
| `tools/museum-patch/WowViewer.Tool.MuseumPatch/PathGuard.cs` | Output containment and input-overwrite refusal |
| `tests/WowViewer.Core.PM4.Tests/MuseumPatchRunnerTests.cs` | Pairing, dry-run, apply, determinism |

### Modified

| File | Change |
|---|---|
| `WowViewer.slnx` | Register the new tool project |
| `tools/README.md` | Document the tool and its no-data constraint |
| `docs/CLI-TOOLS.md` | Add the tool to the CLI index |

### Viewer removals (Spec 280 P6/P7)

| File | Change |
|---|---|
| `Pm4WorkbenchService.Alignment.cs` | Remove Correlation, Reconcile, Alignment tabs |
| `Pm4WorkbenchService.cs` | Remove the match-curation section from Selection |
| `Pm4ObjectMatchStore.cs` | Delete |
| `Pm4WmoGroupMatchService.cs` | Keep only if the viewer still needs shape search; otherwise delete |

## 7. Phase roadmap

### P0 — Skeleton and path guard

- Create the project, argument parsing, and `PathGuard`.
- Prove the tool builds and refuses unsafe paths.
- Receipt: `evidence/p0-skeleton.md`.

### P1 — Discovery and pairing

- Enumerate Museum ADTs and PM4s, pair by tile, report unpaired.
- Receipt: `evidence/p1-discovery.md`.

### P2 — Dry-run proposals

- Run the core pipeline per tile and write `review.json`.
- No ADT output.
- Receipt: `evidence/p2-dry-run.md`.

### P3 — Apply and LK output

- Read the review file, apply accepted proposals, write LK v18 ADTs.
- Receipt: `evidence/p3-apply.md`.

### P4 — Provenance and determinism

- Write provenance reports; prove two runs produce identical hashes.
- Receipt: `evidence/p4-provenance.md`.

### P5 — Viewer removal

- Remove the matching/alignment/reconciliation tabs and the match store from the viewer.
- Keep PM4 inspection.
- Receipt: `evidence/p5-viewer-removal.md`.

### P6 — Documentation and release guard

- Document the tool and the no-data constraint.
- Add a release check that the archive contains no `.adt`, `.pm4`, or `.pd4`.
- Receipt: `evidence/p6-docs-release.md`.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| The core pipeline assumes viewer state | P2 proves the pipeline runs with only file paths; any state dependency is moved into a parameter |
| Museum ADTs are not all LK v18 | Detect the format and skip with a report; do not guess |
| Matching quality is poor | Dry-run and review file make the operator the decision-maker; the tool reports scores |
| Output is not byte-deterministic | Sort proposals and use canonical JSON; P4 proves it |
| The tool accidentally ships data | P6 release check fails the build if any `.adt`/`.pm4`/`.pd4` is in the archive |
| Removing viewer tabs breaks a workflow | P5 lands after P3 proves the offline tool covers the same pipeline |

## 9. Verification strategy

- **Build**: `dotnet build wow-viewer/WowViewer.slnx -c Debug`.
- **Tests**: `dotnet test wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug --filter "FullyQualifiedName~MuseumPatch"`.
- **Dry-run**: a real Museum ADT folder and PM4 folder produce a review file and zero ADTs.
- **Apply**: the same inputs plus a review file produce one ADT per accepted tile.
- **Determinism**: two runs produce identical SHA-256 hashes.
- **Format**: the output ADT opens in the 3.3.5 adapter.
- **No data**: the release archive contains zero `.adt`, `.pm4`, or `.pd4` files.

Runtime and format proof are operator-owned; compilation and unit tests alone do not satisfy
SC-004 or SC-005.

## 10. Out of scope

- A new matcher. The tool reuses the existing scorer and reconciliation engine.
- Museum ADT creation. The operator supplies the Museum ADTs.
- Distributing patched ADTs. The tool writes them locally; distribution is out of scope.
- Viewer PM4 inspection changes beyond removing the matching tabs.
