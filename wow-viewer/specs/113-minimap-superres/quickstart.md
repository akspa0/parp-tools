# Quickstart: ComfyUI-Native Minimap Super-Resolution (RealPLKSR)

**Status**: the detail renderer, v50 build wiring, provenance gate, alignment analyzer, datastore
visual-review surface, guarded pair-set builder, RealPLKSR model wrapper, and CPU contract tests are
implemented. The real cross-map analyzer ran on 2026-07-18, but raw full-frame
NCC is no longer the contract owner because authored objects/icons are intentionally absent from
the terrain-only target. A real synthetic lighting-space bug was found and fixed; existing synthetic
RGB must be refreshed after the bounded comparison-set proof below. All commands run from
`wow-viewer/data-harvester/` unless noted.

**Prerequisite**: Spec 112 is complete — the rebuilt Kalimdor/Azeroth stores carry
`minimap_rgb_authored` and `minimap_rgb_1024` with honest coverage.

## Phase 1 — Detail render + alignment gate (US1, the make-or-break step)

### 1.1 Build and unit-test the detail render mode

```powershell
dotnet build ../WowViewer.slnx -c Debug
dotnet test ../tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~TerrainMinimapDetailRenderTests"
```

Expect four focused tests: real position-dependent texels, honest undecodable fallback, unchanged
material-average default, and mip-stable 1024→512 behavior below the stated MAE bound. The UV and
manifest provenance use the production eight repeats per chunk.

**Fixture proof (2026-07-18)**: the combined C# filter reports 7/7 (including all 4 detail-render
tests); the focused Python alignment/pair/height contract set reports 28 passed, 1 optional
`jsonschema` check skipped; the current full v50 suite reports 175 passed, 4 skipped. Current
combined compositor/detail/DBC/lookup C# focus: 41 passed. Solution build: 0
errors (existing warnings only).

### 1.2 USER RUNS: rebuild two staged stores with detail HR

First run the bounded lighting proof. These commands use the already-built Harvest DLL and write
the native authored minimap, corrected 1024px terrain render, liquid render, and a left-authored /
right-synthetic comparison for every requested tile. They do not modify either dataset store.

```powershell
# Five 0.5.3 Kalimdor tiles already proven to carry authored and synthetic RGB in store lineage.
dotnet ../tools/harvest/WowViewer.Tool.Harvest/bin/Debug/net10.0/WowViewer.Tool.Harvest.dll synthetic-minimap `
  --client-root "H:\CLIENTS\0_5_3_3368" `
  --build 0.5.3.3368 --map Kalimdor `
  --tile-list "41,35;41,32;40,21;27,14;37,37" `
  --resolution 1024 --detail --per-tile --authored-reference `
  --output-dir ../output/lighting-proof/0.5.3-Kalimdor-comparison-set-noon-white

# Six 2.4.3 tiles proven WDT-occupied, authored-nonblack, and backed by 6-10 declared terrain
# materials with 5-10 nonblack decoded BLPs. Expansion01 32,32 is explicitly retired: it was black.
dotnet ../tools/harvest/WowViewer.Tool.Harvest/bin/Debug/net10.0/WowViewer.Tool.Harvest.dll synthetic-minimap `
  --client-root "H:\CLIENTS\2.X_Retail_Windows_enUS_2.4.3.8606\World of Warcraft" `
  --build 2.4.3.8606 `
  --map Expansion01 --tile-list "24,24;21,28;28,30;26,26;27,27;23,30" `
  --resolution 1024 --detail --per-tile --authored-reference `
  --output-dir ../output/lighting-proof/2.4.3-Expansion01-comparison-set-noon-white
```

Inspect `tiles/*_authored_vs_synthesized.png`: authored is the left half, corrected synthetic is the
right half. Native 256px authored and 1024px synthetic files are also preserved separately. The
command rejects an all-black authored reference or synthetic result instead of reporting success.
Both manifests must report `NoonWhiteGlobal` with
`synthetic_minimap_fixed_noon_global_white`; neither may contain per-tile LIT/Light DBC colors.
The interactive viewer's exact-build DBC status is a separate runtime proof.

After that visual sign-off, rerun the two canonical store builds below. The numeric terrain arrays
are already sane; this refresh is required because the old `minimap_rgb` and `minimap_rgb_1024`
pixels were rendered before the MCNR-to-renderer normal correction.

The canonical v50 builder now keeps `minimap_rgb` material-average and automatically passes
`--detail` only for `minimap_rgb_1024`. Write staging stores so the proven Spec 112 stores remain
intact until this gate passes.

```powershell
uv run python scripts/v50_build_dataset.py build `
  --harvest-project ../tools/harvest/WowViewer.Tool.Harvest `
  --clients-root H:\CLIENTS --map Kalimdor --stream-profile v22 `
  --signals-config ./v50_configs/v50-signals-0_5_3_3368.json `
  --manifest-template ./v50_configs/v50-manifest-template-0_5_3_3368.json `
  --report ../output/reports/v50/v50.1/build-detail-0_5_3_3368-Kalimdor.json `
  --write-store ../output/datasets/v50/v50.1/0_5_3_3368-Kalimdor-detail-staging.zarr `
  --write-manifest ../output/reports/v50/v50.1/build-manifest-detail-0_5_3_3368-Kalimdor.json `
  --confirm-run

uv run python scripts/v50_build_dataset.py build `
  --harvest-project ../tools/harvest/WowViewer.Tool.Harvest `
  --clients-root H:\CLIENTS --map Azeroth --stream-profile v22 `
  --signals-config ./v50_configs/v50-signals-0_5_3_3368.json `
  --manifest-template ./v50_configs/v50-manifest-template-0_5_3_3368.json `
  --report ../output/reports/v50/v50.1/build-detail-0_5_3_3368-Azeroth.json `
  --write-store ../output/datasets/v50/v50.1/0_5_3_3368-Azeroth-detail-staging.zarr `
  --write-manifest ../output/reports/v50/v50.1/build-manifest-detail-0_5_3_3368-Azeroth.json `
  --confirm-run
```

Then finalize each staging store with its real build manifest and lineage report:

```powershell
uv run python scripts/v50_build_dataset.py finalize `
  --store ../output/datasets/v50/v50.1/0_5_3_3368-Kalimdor-detail-staging.zarr `
  --manifest ../output/reports/v50/v50.1/build-manifest-detail-0_5_3_3368-Kalimdor.json `
  --row-lineages ../output/reports/v50/v50.1/build-detail-0_5_3_3368-Kalimdor.json `
  --policy-template ./v50_configs/v50-manifest-template-0_5_3_3368.json `
  --output ../output/datasets/v50/v50.1/0_5_3_3368-Kalimdor-detail-staging.manifest.json

uv run python scripts/v50_build_dataset.py finalize `
  --store ../output/datasets/v50/v50.1/0_5_3_3368-Azeroth-detail-staging.zarr `
  --manifest ../output/reports/v50/v50.1/build-manifest-detail-0_5_3_3368-Azeroth.json `
  --row-lineages ../output/reports/v50/v50.1/build-detail-0_5_3_3368-Azeroth.json `
  --policy-template ./v50_configs/v50-manifest-template-0_5_3_3368.json `
  --output ../output/datasets/v50/v50.1/0_5_3_3368-Azeroth-detail-staging.manifest.json
```

`--policy-template` keeps the frozen required/optional policy and derives coverage from the
per-row lineage. It repairs a manifest written before the synthesized-minimap partial-coverage
correction without rebuilding or changing any tile bytes.

Estimated runtime: roughly 20–45 minutes for Kalimdor and 15–30 minutes for Azeroth, depending on
CPU and storage throughput. This is a heavy client-backed run and is user-launched only.

Render the actual same-row datastore evidence before building pairs. This writes one contact sheet
per store plus the machine-readable `visual-review.json` required by cross-domain pairing:

```powershell
uv run python scripts/v50_visualize_store.py `
  --store ../output/datasets/v50/v50.1/0_5_3_3368-Kalimdor-detail-staging.zarr `
  --store ../output/datasets/v50/v50.1/0_5_3_3368-Azeroth-detail-staging.zarr `
  --output-dir ../output/reports/v50/v50.1/visual-review-0_5_3_3368 `
  --samples-per-store 6
```

Each row shows authored 256, synthetic 256, detail 1024 overview, a native 1024 center crop,
relative height, and terrain normals. The report explicitly records that authored images may carry
objects while synthetic targets are terrain-only; it does not claim pixel equality.

### 1.3 Alignment analysis (the US1 gate)

```powershell
uv run python scripts/v50_analyze_minimap_alignment.py `
  --store ../output/datasets/v50/v50.1/0_5_3_3368-Kalimdor-detail-staging.zarr `
  --store ../output/datasets/v50/v50.1/0_5_3_3368-Azeroth-detail-staging.zarr `
  --sample 60 `
  --output ../output/reports/v50/v50.1/alignment-detail-0_5_3_3368-big-maps.json
```

Report conforms to the Alignment Report shape (data-model.md). The raw-pixel result remains a
diagnostic for the intentionally cross-domain pair. Proceed only with same-row identity lineage,
the explicit `--terrain-only-cross-domain` flag, and the persisted visual-review evidence. Also
confirm SC-001 (detail render's high-frequency content exceeds a bicubic upscale of the
material-average render) here.

**Real result (2026-07-18): raw full-frame NCC is inapplicable as the promotion owner.**
`alignment-detail-0_5_3_3368-big-maps.json` sampled 120 tiles (60/map):
`gate=fail_inconsistent`, no global transform or offset, NCC p50 `0.2113`, NCC p05 `0.0000`,
residual p50 `0.7887`, residual p95 `1.0000`. SC-001 passed with detail gain `16.1047`.
The authored client minimap and synthesized terrain target intentionally differ in objects/icons,
water, and material treatment. The lighting difference also included the now-fixed MCNR coordinate
bug. Do not apply a per-tile transform. Refresh the synthetic RGB and use the explicit terrain-only
cross-domain contract below.

## Phase 2 — SR pair set (US2)

The guarded builder and its contract tests are implemented. Do not build the real pair set until
the corrected-light comparison and refreshed-store contact sheets above are accepted.

```powershell
uv run python scripts/v50_build_sr_pairset.py `
  --store ../output/datasets/v50/v50.1/0_5_3_3368-Kalimdor-detail-staging.zarr `
  --store ../output/datasets/v50/v50.1/0_5_3_3368-Azeroth-detail-staging.zarr `
  --alignment ../output/reports/v50/v50.1/alignment-detail-0_5_3_3368-big-maps.json `
  --terrain-only-cross-domain `
  --visual-review ../output/reports/v50/v50.1/visual-review-0_5_3_3368/visual-review.json `
  --val-fraction 0.15 `
  --output ../output/datasets/v50/v50.1/sr-pairset-0_5_3_3368-v1.zarr
```

Only tiles with BOTH an authored minimap and a detail render become pairs; excluded tiles are
counted (schema `sr-pairset-and-run.schema.json`). Kalimdor+Azeroth only; leak-safe per-tile split.
This is CPU-side and assistant-runnable once the stores exist.

## Phase 3 — Train and evaluate (US3, user-run)

> **Not runnable yet.** T014's RealPLKSR/ComfyUI model wrapper exists, but T015-T017 and
> `scripts/v50_train_minimap_superres.py` remain unimplemented until T010b's corrected-light proof
> is accepted and T013 builds the real paired corpus. The commands below are the frozen post-gate
> handoff, not commands to execute now.

### 3.1 Stage 1 — PSNR/L1 generator (prove the pairing works before any GAN)

```powershell
uv run python scripts/v50_train_minimap_superres.py `
  --pairset ../output/datasets/v50/v50.1/sr-pairset-0_5_3_3368-v1.zarr `
  --stage psnr --arch realplksr_x4 --patch 256 `
  --output ../output/v50/v50.1/minimap_realplksr_psnr_v1 `
  --epochs 100 --batch 8 --patience 15
```

**Printed for the user to run, never launched by the assistant.** Review the summary
(`sr-run-v1`): held-out PSNR/SSIM/LPIPS vs detail HR and `beats_bicubic` on the SC-004 detail
metric. Then eyeball held-out outputs (SC-005).

### 3.2 Stage 2 — optional GAN fine-tune (only if stage 1 is too smooth)

```powershell
uv run python scripts/v50_train_minimap_superres.py `
  --pairset ../output/datasets/v50/v50.1/sr-pairset-0_5_3_3368-v1.zarr `
  --stage gan --init ../output/v50/v50.1/minimap_realplksr_psnr_v1/checkpoint_best.pth `
  --output ../output/v50/v50.1/minimap_realplksr_gan_v1 `
  --epochs 100 --batch 8 --patience 15
```

Entered only after reviewing stage 1 (contract: GAN is never trained first). Re-run the SC-004
metrics and the SC-005 visual gate; watch for GAN hallucination (fabricated structure) — that fails
SC-005 even if perceptual metrics improve.

## Out of scope for this pass

- 2048/4096 "giant" renders and training — future work; the pipeline avoids hardcoding 1024 (FR-011)
  but does not build those scales here.
- PVPZone02/Kalidar — excluded from this lane entirely.
- Synthetic-degradation pairing (degrade detail HR to make LR) — not selected; the accepted contract
  uses real authored LR plus same-row terrain-only HR with explicit object-policy evidence.
