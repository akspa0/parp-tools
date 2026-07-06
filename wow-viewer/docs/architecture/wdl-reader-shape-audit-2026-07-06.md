# WDL Reader Shape Audit — 2026-07-06 (Spec 094, Phase 0)

Audited the existing C# WDL surfaces that Spec 094 (V24) wraps. The C# reader is the
de-facto ground truth for the WDL grid shape; this document records what it actually
returns, verified against real staged-client data for both target builds.

## C# surfaces (unmodified by V24)

| Surface | Location | Role |
|---|---|---|
| `WdlSummaryReader.Read(path\|stream)` | `src/core/WowViewer.Core.IO/Maps/WdlSummaryReader.cs` | Real WDL reader (MVER → MAOF → per-tile MARE) |
| `WdlSummary` / `WdlTileSummary` | `src/core/WowViewer.Core/Maps/WdlSummary.cs` | Reader output model |
| `WdlWriter.ExtractTileHeightsFromAlpha(float[,],int,int)` | `src/core/WowViewer.Core.IO/Maps/WdlWriter.cs` | Terrain→WDL lattice extraction (the "click on map to spawn" path) |
| `NativeMpqService` | `src/core/WowViewer.Core.IO/Files/NativeMpqService.cs` | Archive resolution for staged clients |

## Per-MARE output shape (both target builds)

- `OuterHeights`: **17×17 = 289** `short` (int16), row-major. Sampled at `height_257[16r, 16c]`.
- `InnerHeights`: **16×16 = 256** `short` (int16), row-major. Sampled at `height_257[16r+8, 16c+8]`.
- 64×64 tile grid per WDL, MAOF offset table, `MVER` optional (reader tolerates its absence and both FourCC byte orders, so era differences are absorbed).
- **MAHO is not parsed** by `WdlSummaryReader`. V24 therefore stores no `wdl_prior_holes`; Stage B hole gating uses V18 `holes_16` (real ADT MCNK holes) instead.

## Verified real-data reads (via `WowViewer.Tool.WdlRead read`)

| Build | Map | Resolution path | MARE tiles | MVER |
|---|---|---|---|---|
| `3_3_5_12340` | Azeroth | `World\Maps\Azeroth\Azeroth.wdl` (inside big MPQs) | 687 | 18 |
| `0_5_3_3368` | Azeroth | `World\Maps\Azeroth\Azeroth.wdl` (loose `.wdl.mpq` mini-MPQ) | 685 | 18 |

Both eras return the identical 17×17 + 16×16 int16 layout. The Alpha-era layout
difference feared in Spec 094 Risk 9 did not materialize: the 0.5.3 WDL parses with
the same reader and the same shape.

## Synthetic-vs-real convergence check

Built synthetic WDL grids from V18 `height_257` (8 Azeroth `3_3_5_12340` tiles) via
`WowViewer.Tool.WdlRead synth` (which wraps `WdlWriter.ExtractTileHeightsFromAlpha`
with nearest-non-liquid resampling) and compared them against the real client WDL:

- Per-cell |synthetic − real| ≤ 1.0 world unit at **100 %** of lattice points on all 8 tiles.
- Mean per-tile L1 was 0.70–1.00, i.e. entirely int16 quantization (the client stores
  WDL heights as int16; several tiles show a systematic 1.0 offset consistent with
  floor-vs-round quantization).

This validates the Spec 094 premise ("terrain and wdl data match for 99% of the valid
tiles") and fixes the merge rule: `disagree_threshold` defaults to 1.0 and the
comparison is **inclusive** (`|real − synth| ≤ threshold` → cell counts as agreeing).

## Shim contract (canonical Python entry point)

```text
WowViewer.Tool.WdlRead read  --client-root <staged-root> --map <name> [--tile-x N --tile-y N] --output <npz>
WowViewer.Tool.WdlRead read  --wdl <loose .wdl path>     [--tile-x N --tile-y N] --output <npz>
WowViewer.Tool.WdlRead synth --height <npz> [--liquid <npz>] --output <npz>
```

- `read` NPZ: `tile_xy` (N,2) int32, `outer` (N,17,17) float32, `inner` (N,16,16) float32, `version` (1,) int32 (−1 if no MVER). Exit 0 ok / 2 map has no WDL / 3 requested tile absent.
- `synth` NPZ in: `height_257` (257,257) or (N,257,257) float32; optional `liquid_mask` (256,256) or (N,256,256), >0.5 = liquid. NPZ out: `outer`/`inner` stacked as above. Batch-first: one process call covers a whole map or tile stack.
- int16 → float32 conversion happens at the shim boundary (Spec 094 Risk 8).
