# MH2O Corpus Re-Audit And V7 Smoke Plan

## Date

- Apr 10, 2026

## Purpose

Execute the next bounded follow-up after the MH2O exporter repair:

1. re-export one or two real corpus roots that previously had dead liquid supervision
2. prove the repaired path on at least one real partial-coverage MH2O tile instead of only a full ocean tile
3. rerun the V7 smoke training on the rebuilt corpus only after the audit says the liquid channel is alive again

This is a fresh-chat execution plan, not another architecture discussion.

## Required Starting Context

Read these first in the next chat:

- `gillijimproject_refactor/memory-bank/activeContext.md`
- `gillijimproject_refactor/memory-bank/progress.md`
- `gillijimproject_refactor/plans/wow_viewer_ml_tool_suite_cutover_plan_2026-04-10.md`
- `gillijimproject_refactor/docs/VLM_DATASET_EXPORTER.md`

## Current Verified State

- The active LK exporter no longer hardcodes `terrain_data.liquids = null`.
- A real-data smoke on `3.3.5.12340` `Azeroth_35_20` logged `Parsed 256 MH2O liquid layers` and emitted non-null `liquids`, `liquid_mask`, and `liquid_height` outputs under `output/tmp/mh2o-smoke-335-azeroth/`.
- That smoke tile appears to be full-coverage water, so it proves dead-signal recovery, not partial-rect MH2O fidelity.
- The known bad corpora previously audited as missing effective liquid supervision were:
  - `output/ml-corpus/301_8303/Northrend`
  - `output/ml-corpus/400_11927/LostIsles`

## Hard Scope Boundaries

- Do not tune the model first.
- Do not broaden the exporter again unless the audit or partial-rect validation finds a concrete defect.
- Do not claim success based only on focused unit tests or build output.
- Do not rerun large full-corpus jobs until a narrow real-data re-export proves the repaired channel is alive.

## Step 1 - Rebuild Narrow Real Corpora And Re-Audit

### Goal

Replace at least one previously dead corpus root with fresh exports from the repaired exporter, then rerun `audit_v7_signals.py` and compare against the earlier 0% effective liquid coverage result.

### Recommended First Targets

- first target: `output/ml-corpus/301_8303/Northrend`
- second target if time allows: `output/ml-corpus/400_11927/LostIsles`

### Preferred Execution Shape

- avoid regenerating every configured map first
- regenerate one narrow dataset root per target map
- if the audit succeeds on the narrow root, then decide whether to refresh the checked-in broader corpus root

### Suggested Commands

Use the fixed local clients already recorded in continuity:

- `3.0.1.8303` client root: `H:\CLIENTS\3.X_Pre-Release_Windows_enUS_3.0.1.8303\World of Warcraft`
- `4.0.0.11927` client root: `H:\CLIENTS\World of Warcraft Cata beta 11927`

Suggested narrow export commands:

```powershell
dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj --configuration Debug -- ml-export --client "H:/CLIENTS/3.X_Pre-Release_Windows_enUS_3.0.1.8303/World of Warcraft" --map Northrend --out i:/parp/parp-tools/output/tmp/mh2o-reaudit-301-northrend --limit 8

dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj --configuration Debug -- ml-export --client "H:/CLIENTS/World of Warcraft Cata beta 11927" --map LostIsles --out i:/parp/parp-tools/output/tmp/mh2o-reaudit-400-lostisles --limit 8
```

Then rerun the audit script against those rebuilt roots:

```powershell
C:/Users/akspa/anaconda3/python.exe i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/scripts/audit_v7_signals.py --dataset-root i:/parp/parp-tools/output/tmp/mh2o-reaudit-301-northrend

C:/Users/akspa/anaconda3/python.exe i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/scripts/audit_v7_signals.py --dataset-root i:/parp/parp-tools/output/tmp/mh2o-reaudit-400-lostisles
```

### What To Record

- total tiles audited
- count of tiles with non-null `terrain_data.liquids`
- effective liquid-mask coverage, not just field presence
- whether `no_liquid_minimap` is emitted consistently when liquid masks exist
- whether object supervision remains dead or improves incidentally

### Exit Criteria

- at least one rebuilt corpus root shows clearly nonzero effective liquid supervision
- if both rebuilt roots still show effectively dead liquid coverage, stop and inspect the exported JSON and stitched liquid images before touching training

## Step 2 - Validate One Real Partial-Rect MH2O Tile

### Goal

Prove the new `x_offset` or `y_offset` or `width` or `height` or `exists_bitmap` metadata is doing real work on a live tile, instead of only validating full 8x8 water coverage.

### Why This Matters

The current smoke only proved that the exporter now emits water again.
It did not prove that partial MH2O rectangles survive export, stitching, and viewer-side loading correctly.

### Search Strategy

Use one of the rebuilt narrow roots from Step 1 and scan the emitted JSON for a liquid layer where at least one of the following is true:

- `x_offset != 0`
- `y_offset != 0`
- `width != 8`
- `height != 8`
- `exists_bitmap != null`

Suggested searches:

```powershell
rg '"x_offset":|"y_offset":|"width":|"height":|"exists_bitmap":' i:/parp/parp-tools/output/tmp/mh2o-reaudit-301-northrend/dataset

rg '"x_offset":|"y_offset":|"width":|"height":|"exists_bitmap":' i:/parp/parp-tools/output/tmp/mh2o-reaudit-400-lostisles/dataset
```

### Validation Actions

For one good candidate tile:

- inspect the tile JSON and note the exact liquid layer metadata
- confirm the stitched `liquid_mask` image is not just a full white 64x64 chunk block for that layer's chunk
- if useful, load the dataset root through the existing `MdxViewer` VLM project path and confirm the viewer-side liquid mesh respects the partial coverage via `TileFlags`

### Minimum Proof To Capture

- one tile path
- one chunk index with partial MH2O coverage
- the exported metadata values
- confirmation that the stitched mask shape matches the metadata rather than whole-chunk fill

### Exit Criteria

- one real partial-coverage tile is documented and behaves consistently across exported JSON and stitched mask output
- if no such tile appears in the narrow rebuilt roots, record that explicitly and do not overclaim partial-rect signoff

## Step 3 - Rerun V7 Smoke Training On Rebuilt Data

### Goal

Repeat the earlier V7 smoke only after Step 1 shows live liquid supervision and Step 2 does not expose a fresh exporter defect.

### Preferred Dataset Shape

- start with the rebuilt narrow root from Step 1, not the stale old corpora under `output/ml-corpus/...`
- if both rebuilt roots look healthy, use both
- keep the smoke bounded: low epoch count, batch size `1`, no augmentation, no architecture changes

### Suggested Command Shape

One-root smoke:

```powershell
C:/Users/akspa/anaconda3/python.exe i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py --profile manual --dataset-root i:/parp/parp-tools/output/tmp/mh2o-reaudit-301-northrend --output-dir i:/parp/parp-tools/output/tmp/v7-mh2o-reaudit-smoke-301 --epochs 3 --batch-size 1 --spatial-group-size 1 --no-augment
```

Two-root smoke if both audits are good:

```powershell
C:/Users/akspa/anaconda3/python.exe i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py --profile manual --dataset-root i:/parp/parp-tools/output/tmp/mh2o-reaudit-301-northrend --dataset-root i:/parp/parp-tools/output/tmp/mh2o-reaudit-400-lostisles --output-dir i:/parp/parp-tools/output/tmp/v7-mh2o-reaudit-smoke-mixed --epochs 3 --batch-size 1 --spatial-group-size 1 --no-augment
```

### What To Compare Against

Compare the new smoke against the earlier runs that used stale corpora with dead liquid or object channels:

- check dataset loading counts
- check whether any liquid-related feature stats or sample diagnostics change meaningfully
- check training or validation loss only as a secondary signal after verifying the input channels are alive

### Exit Criteria

- the smoke completes on rebuilt data without schema or loader breakage
- the run is explicitly described as a bounded smoke on refreshed data, not a final quality claim

## Failure Routing

### If Step 1 still shows dead liquid supervision

- stop before training
- inspect one emitted tile JSON plus `liquid_mask` or `liquid_height`
- check whether the exporter is writing only flat zero-height full-coverage layers on that map family

### If Step 2 finds a partial-rect mismatch

- stop before training
- fix the exporter or stitch or loader seam directly
- rerun the narrow validation on the same tile before broad corpus regeneration

### If Step 3 trains but still looks obviously bad

- do not jump straight into model surgery
- first compare rebuilt-corpus audit metrics against the stale-corpus audit metrics and verify what actually changed in the input channel population

## Fresh-Chat Prompt Seed

Use this to start the next chat cleanly:

```text
Continue from the MH2O exporter repair. Execute the plan in gillijimproject_refactor/plans/mh2o_corpus_reaudit_and_v7_smoke_plan_2026-04-10.md. The bounded goals are: (1) rebuild at least one previously dead corpus root and rerun audit_v7_signals.py, (2) prove one real partial-coverage MH2O tile instead of only full ocean coverage, and (3) rerun the V7 smoke only on the rebuilt corpus if the audit is healthy. Do the work, keep proof boundaries precise, and do not skip the real-data validation step.
```