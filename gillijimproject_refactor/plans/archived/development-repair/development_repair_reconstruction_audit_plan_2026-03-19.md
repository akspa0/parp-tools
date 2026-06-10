# Development Repair Reconstruction Audit Plan (Mar 19, 2026)

## 1. What We Have Right Now

### Canonical Input Dataset (Constituent Source)
Path: `test_data/development/World/Maps/development`

Observed filesystem inventory:
- Root ADTs: 466
- `_obj0.adt`: 426
- `_tex0.adt`: 333
- PM4 files: 616
- WL files: 64 `.wlw`, 4 `.wlm`, 0 `.wlq`, 2 `.wll`
- Zero-byte root ADTs: 114
- `development.wdt`: present
- `development.wdl` (or `.wdl.mpq`): present

### Current Repair Output Snapshot (Real Run)
Path: `output/development-repair-real-20260319`

Observed output inventory:
- Output ADTs: 703
- Tile manifests: 703
- Summary manifest: present
- Output `development.wdt`: present
- Output `development.wdl`: present

Manifest-derived processing reality:
- `RepairRoute`: 703 `wdl-rebuild`
- `SplitMergeRan`: 0
- `WdlGenerationRan`: 589
- `ChunkIndicesRepairRan`: 589
- `WlLiquidsConverted`: 65
- `NeedsManualReview`: 114
- Top warnings:
  - `WDL tile data is missing for this coordinate.`
  - `Copied root ADT without successful repair pipeline execution.`

## 2. Critical Audit Findings

### Finding A: Analyzer is currently dropping all tile files
`DevelopmentMapAnalyzer` captures extension without the leading dot, but switch cases expect dot-prefixed extensions.

Code evidence:
- Extension assignment: `extension = match.Groups[5].Value` in `DevelopmentMapAnalyzer.cs`
- Switch cases use `.adt`, `.pm4`, `.wlw`, etc.

Impact:
- Analyzer reports zero roots/obj0/tex0/pm4/wl tiles even when files exist.
- Tiles get classified as `wdl-rebuild` via the `missing` path.
- Repair orchestration receives misleading class/action metadata.

### Finding B: Current outputs are mostly WDL-generated, not constituent-reconstructed
Even with constituent-first attempt logic in `DevelopmentRepairService`, the effective route is dominated by WDL rebuild.

Impact:
- Produced ADTs do not reliably contain all reconstructable constituent data.
- We are not yet proving "4.x split inputs -> deterministic 3.3.5 monolithic ADT" end-to-end.

### Finding C: We need strict writer invariants for full reconstruction claims
To claim perfect 3.3.5 output from split inputs, we must enforce deterministic regeneration of:
- Top-level chunk table and offsets
- `MCIN` entries
- `MCNK` header index fields
- MCNK subchunk offsets/sizes (e.g., `OfsMcvt`, `OfsMcly`, `OfsMcal`, `SizeMcal`, etc.)
- Optional chunk alignment/padding policy at write boundaries

## 3. Target Processing Model (What "Correct" Looks Like)

Per-tile reconstruction precedence:
1. Constituent merge path: root + `_obj0` + `_tex0`
2. Constituent composition fallback: root + `_tex0` terrain payload graft where merge fails
3. Root-only salvage path: root ADT with full index/offset regeneration
4. WDL generation fallback only when no reconstructable terrain payload exists

Per-tile manifest must clearly encode:
- `RepairRoute` as one of:
  - `constituent-split-merge`
  - `constituent-tex0-compose`
  - `constituent-root-only`
  - `wdl-generated-fallback`
- Which source files were used
- Which structural repair phases ran
- Whether any fallback occurred

## 4. Implementation Plan (Staged)

### Phase 0: Truthful Audit Foundation
1. Fix extension parsing mismatch in `DevelopmentMapAnalyzer`.
2. Re-run `development-analyze` on canonical input.
3. Validate class distribution reflects actual files (not all missing).

Deliverable:
- Accurate class counts and tile-level source presence in analyzer output.

### Phase 1: Reconstruction-First Routing
1. Ensure `DevelopmentRepairService` routes by actual source availability, not only initial class.
2. Add explicit route transitions when falling back (`merge -> compose -> root -> wdl`).
3. Record route transitions and reasons in manifest warnings.

Deliverable:
- Manifest route distribution no longer 100% `wdl-rebuild`.

### Phase 2: Structural Writer Guarantees
1. Introduce/centralize a canonical ADT writer step after any reconstruction source.
2. Always regenerate:
   - `MCIN` from actual written MCNK offsets
   - MCNK index fields (`IndexX`, `IndexY`)
   - MCNK subchunk offsets/sizes in header
3. Validate top-level chunk offsets and referenced subchunk bounds.

Deliverable:
- Rebuilt ADTs pass structural validation independent of source path.

### Phase 3: Constituent Payload Completeness
1. Verify texture/layer/alpha/shadow propagation from constituent pieces.
2. Preserve/merge liquids (`WL* -> MH2O`) after terrain reconstruction.
3. Keep PM4/MPRL phase explicit and optional in this slice.

Deliverable:
- Reconstructed ADTs include expected terrain payload beyond WDL geometry.

### Phase 4: Real-Data Validation + Reporting
1. Run full repair on canonical input to a fresh output folder.
2. Produce machine-readable QA summary:
   - route distribution
   - fallback reasons
   - manual-review tile list grouped by reason
   - phase execution counts
3. Spot-check representative tiles in viewer.

Deliverable:
- Evidence-backed statement of how much output is fully constituent reconstructed vs fallback generated.

## 5. Real Data Test Commands

Use these exact commands:

```powershell
dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -- development-analyze --input-dir i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development
```

```powershell
dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -- development-repair --mode repair --input-dir i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development --output-dir i:/parp/parp-tools/gillijimproject_refactor/output/development-repair-real-next
```

## 6. Exit Criteria for "We Are Reconstructing"

We can claim successful constituent reconstruction only when all are true:
- Analyzer sees real source files and gives non-degenerate class distribution.
- Majority of tiles are processed via constituent routes, not WDL fallback.
- MCIN + all internal offset tables are regenerated and structurally valid.
- Manifest provenance clearly states the exact route per tile.
- Remaining fallback/manual tiles are explicit and bounded by reason.

## 7. Immediate Next Action

Implement Phase 0 first (analyzer extension parsing fix), then rerun full analyze + repair and compare route distribution before touching broader reconstruction logic.
