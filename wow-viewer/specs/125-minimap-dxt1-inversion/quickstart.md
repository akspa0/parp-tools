# Quickstart: Minimap DXT1 Artifact Inversion

**Phase 1 output** | **Date**: 2026-08-02 | **Spec**: [spec.md](./spec.md)

## What this gives you

- **Fair comparison**: synthetic tiles now carry a DXT1 parity companion, so authored vs synthetic
  are scored on equal terms instead of clean-vs-lossy.
- **Lighting-baseline awareness**: per-map survey tells you whether authored tiles share a common
  lighting baseline, and accounts for it when scoring.
- **Restoration**: a residual model that pushes authored tiles back toward their pre-compression
  appearance, trained on locally generated pairs (no authored reference needed).

## Build

```powershell
dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```

## Run — parity companion + fair score

```powershell
dotnet run --project i:/parp/parp-tools/wow-viewer/tools/harvest/WowViewer.Tool.Harvest -- `
  synthetic-minimap --map <map> --client-root <root> --build <build> `
  --per-tile --dxt1-parity --score --authored-reference
```

- `--dxt1-parity` emits a `*_dxt1.png` parity companion per tile (FR-015).
- `--score --authored-reference` writes `authored-comparison.csv` with parity-adjusted agreement
  alongside unadjusted (FR-003).

## Run — lighting baseline survey

```powershell
dotnet run --project i:/parp/parp-tools/wow-viewer/tools/harvest/WowViewer.Tool.Harvest -- `
  synthetic-minimap --map <map> --client-root <root> --build <build> `
  --lighting-baseline --authored-reference
```

Reports whether a shared lighting baseline exists for the map and, if so, accounts for it (FR-016).

## Run — encoding survey

```powershell
dotnet run --project i:/parp/parp-tools/wow-viewer/tools/harvest/WowViewer.Tool.Harvest -- `
  synthetic-minimap --map <map> --client-root <root> --build <build> --encoding-survey
```

Reports the per-build/map distribution of encodings (FR-013).

## Run — restoration training (user-run, heavy)

```powershell
cd i:/parp/parp-tools/wow-viewer/data-harvester
uv run python scripts/train_v20_dxt1_restore.py --out <checkpoint-dir>
```

Trains the residual restoration model on locally generated pristine→encoded pairs (FR-007). This is
a training run — the user presses go.

## Test

```powershell
dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```

Covers the DXT1 round-trip check (FR-014), parity cycle (FR-002), and lighting-baseline detection
(FR-016).
