# Quickstart: Minimap Super-Resolution (Real-ESRGAN)

**Status**: planning-stage runbook — commands describe the intended sequence; none is implemented
or run yet. Filled in with real proof (counts, hashes, metrics, sample images) as each phase lands,
matching the Spec 109/112 discipline. All commands run from `wow-viewer/data-harvester/` unless
noted. C# changes build via `dotnet build ../WowViewer.slnx -c Debug` first.

**Prerequisite**: Spec 112 is complete — the rebuilt Kalimdor/Azeroth stores carry
`minimap_rgb_authored` and `minimap_rgb_1024` with honest coverage.

## Phase 1 — Detail render + alignment gate (US1, the make-or-break step)

### 1.1 Build and unit-test the detail render mode

```powershell
dotnet build ../WowViewer.slnx -c Debug
dotnet test ../tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~TerrainMinimapDetailRenderTests"
```

Expect a test proving a detail-mode pixel returns a real texel (varies with position within a
high-frequency synthetic texture) rather than the texture average, and that a 1024 detail render of
a tiling texture shows no moire beyond a stated bound.

### 1.2 USER RUNS: detail-render the two maps at 1024

```powershell
# per map, ~heavier than material-average synthesis; run with --detail for the 1024 pass
dotnet ../tools/harvest/WowViewer.Tool.Harvest/bin/Debug/net10.0/WowViewer.Tool.Harvest.dll `
  synthetic-minimap --client-root "H:\CLIENTS\0_5_3_3368\World of Warcraft" `
  --map Kalimdor --resolution 1024 --detail --per-tile `
  --output-dir ../output/reports/v50/v50.1/detail-1024-Kalimdor
```

(Or re-run the Spec 112 `build` with the detail 1024 pass wired in, so `minimap_rgb_1024` in the
store is the detail render — the exact wiring is a tasks.md decision. Either way this is a heavy,
user-run render step.)

### 1.3 Alignment analysis (the US1 gate)

```powershell
uv run python scripts/v50_analyze_minimap_alignment.py `
  --store ../output/datasets/v50/v50.1/0_5_3_3368-Kalimdor.zarr `
  --sample 60 `
  --output ../output/reports/v50/v50.1/alignment-0_5_3_3368-Kalimdor.json
```

Report conforms to the Alignment Report shape (data-model.md). **Gate**: `pass_identity` or
`pass_with_transform` (one transform consistent across all sampled tiles) → proceed. `fail_inconsistent`
→ STOP and surface the finding; do not build pairs. Also confirm SC-001 (detail render's
high-frequency content exceeds a bicubic upscale of the material-average render) here.

## Phase 2 — SR pair set (US2)

```powershell
uv run python scripts/v50_build_sr_pairset.py `
  --store ../output/datasets/v50/v50.1/0_5_3_3368-Kalimdor.zarr `
  --store ../output/datasets/v50/v50.1/0_5_3_3368-Azeroth.zarr `
  --alignment ../output/reports/v50/v50.1/alignment-0_5_3_3368-Kalimdor.json `
  --val-fraction 0.15 `
  --output ../output/datasets/v50/v50.1/sr-pairset-0_5_3_3368-v1.zarr
```

Only tiles with BOTH an authored minimap and a detail render become pairs; excluded tiles are
counted (schema `sr-pairset-and-run.schema.json`). Kalimdor+Azeroth only; leak-safe per-tile split.
This is CPU-side and assistant-runnable once the stores exist.

## Phase 3 — Train and evaluate (US3, user-run)

### 3.1 Stage 1 — PSNR/L1 generator (prove the pairing works before any GAN)

```powershell
uv run python scripts/v50_train_minimap_superres.py `
  --pairset ../output/datasets/v50/v50.1/sr-pairset-0_5_3_3368-v1.zarr `
  --stage psnr --arch rrdbnet_x4 --patch 256 `
  --output ../output/v50/v50.1/minimap_sr_psnr_v1 `
  --epochs 100 --batch 8 --patience 15
```

**Printed for the user to run, never launched by the assistant.** Review the summary
(`sr-run-v1`): held-out PSNR/SSIM/LPIPS vs detail HR and `beats_bicubic` on the SC-004 detail
metric. Then eyeball held-out outputs (SC-005).

### 3.2 Stage 2 — optional GAN fine-tune (only if stage 1 is too smooth)

```powershell
uv run python scripts/v50_train_minimap_superres.py `
  --pairset ../output/datasets/v50/v50.1/sr-pairset-0_5_3_3368-v1.zarr `
  --stage gan --init ../output/v50/v50.1/minimap_sr_psnr_v1/checkpoint_best.pt `
  --output ../output/v50/v50.1/minimap_sr_gan_v1 `
  --epochs 100 --batch 8 --patience 15
```

Entered only after reviewing stage 1 (contract: GAN is never trained first). Re-run the SC-004
metrics and the SC-005 visual gate; watch for GAN hallucination (fabricated structure) — that fails
SC-005 even if perceptual metrics improve.

## Out of scope for this pass

- 2048/4096 "giant" renders and training — future work; the pipeline avoids hardcoding 1024 (FR-011)
  but does not build those scales here.
- PVPZone02/Kalidar — excluded from this lane entirely.
- The synthetic-degradation fallback (degrade detail HR to make LR) — only relevant if US1 alignment
  fails; it changes the deployment story and needs a separate user decision.
