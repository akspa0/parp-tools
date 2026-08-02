# Implementation Plan: Minimap DXT1 Artifact Inversion

**Branch**: `125-minimap-dxt1-inversion` | **Date**: 2026-08-02 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/125-minimap-dxt1-inversion/spec.md`

## Summary

Authored 0.5.3 minimap tiles are DXT1-compressed (BLP2/DXTC/DXT1, 256×256, one mip, no alpha). Our
synthesizer produces pristine 24-bit output, so every comparison has been scoring a clean image
against a lossy one — a codec confound that biases every metric and would teach any mixed corpus to
separate authored from synthetic by compression damage alone. This feature:

1. **Detects** each authored tile's actual encoding per file (FR-001, FR-013).
2. **Reproduces** the same lossy encode/decode cycle on synthetic tiles (FR-002), and — per the
   2026-08-02 update — has the synthesizer **emit a DXT1-compressed parity companion** alongside the
   pristine render (FR-015).
3. **Reports** parity-adjusted agreement alongside unadjusted (FR-003), excluding degenerate tiles
   (FR-004), recording parity status on every row (FR-005), era-gated (FR-006).
4. **Tests the global lighting normalisation hypothesis** — whether authored tiles of a map share a
   common lighting baseline — and accounts for it if found (FR-016).
5. **Restores** authored tiles toward their pre-compression appearance via a learned inverse, trained
   on locally generated pristine→encoded pairs, gated on hallucination (FR-007..FR-012).
6. **Decodes the terrain shadow** from any authored minimap using the synthesizer's lighting model and
   **reconstructs terrain directly** — minimap RGB → heightmap → 3D mesh with a single model
   (FR-017..FR-020). This is the strategic payoff the user stated on 2026-08-02: because we now know
   how the minimap terrain shadow is created, the shadow in an authored tile is a readable
   terrain-shape signal, and the best residual to train against is the decoded shadow itself.
7. **Super-resolves** terrain and texturing data from real low-res/high-res pairs produced by the
   synthesizer (same terrain, matching lighting, no objects), reported separately from artifact
   removal and reconstruction (FR-021). Another door opened by the parity and lighting work.

The DXT1 encoder already exists in-tree: `BCnEncoder.Net` is a dependency of `WowViewer.Core.IO`, and
`AlphaBlpCompatibilityService.EncodeBlp2` already drives `BcEncoder` with `CompressionFormat.Bc1`
(DXT1). The decode half is `SereniaBLPLib` (wrapped by `BlpRgbReader`). No new codec dependency is
required.

## Technical Context

**Language/Version**: C# / .NET 10 (Core.IO, harvest tool); Python 3.11+ / uv (restoration training)

**Primary Dependencies**: `BCnEncoder.Net` (DXT1 encode — already referenced by `WowViewer.Core.IO`),
`SereniaBLPLib` (DXT1/BLP decode — already referenced), `SixLabors.ImageSharp` (pixel I/O), PyTorch
(restoration model, data-harvester)

**Storage**: Files on disk (PNG tiles, `authored-comparison.csv`, manifest JSON); Zarr for any
restoration corpus (per-build store, data-harvester)

**Testing**: `dotnet test WowViewer.slnx` (Core.IO + harvest tool tests); `uv run pytest` for the
restoration model

**Target Platform**: Windows 11 / PowerShell 7; CLI tool `WowViewer.Tool.Harvest synthetic-minimap`

**Project Type**: Shared library (`WowViewer.Core.IO`) + CLI tool (`WowViewer.Tool.Harvest`) + Python
training (data-harvester)

**Performance Goals**: DXT1 encode of a 256×256 tile is sub-millisecond; per-tile parity companion
adds negligible cost to the existing parallel tile pass. Restoration inference is a small residual
network at native 256×256.

**Constraints**: `gillijimproject_refactor` read-only; `AlphaWdtWriter.cs` frozen; no BLP *export*
(encode/decode cycle in memory only); no super-resolution; era-gating with unrecognised-build flag;
user runs training and heavy client-backed proof.

**Scale/Scope**: Per-tile encode/decode cycle; per-map lighting-baseline survey; restoration model
trained on locally generated pairs (no authored reference required).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Repo Independence | PASS | All new code in `wow-viewer/src/core/WowViewer.Core.IO` + harvest tool + data-harvester. No path outside `wow-viewer/`. |
| II. Library-First | PASS | DXT1 cycle + lighting-baseline live in `WowViewer.Core.IO`; the harvest tool is a thin wrapper. |
| III. Real-Data Validation | PASS | Survey + parity validated against `H:\CLIENTS` authored tiles; build + fingerprint recorded. |
| IV. Residual Model Chain | PASS | Restoration is a single residual model (predicts pre-encode minus encoded), one output, own checkpoint. |
| V. Streaming-First Dataset | PASS | Restoration pairs generated in-memory; no intermediate NPZ on disk. |
| VI. No Client Path Assumptions | PASS | Client root is runtime config; no hardcoded path. |
| Read-Only Reference | PASS | No writes to `gillijimproject_refactor`. |
| Format Reader/Writer Ownership | PASS | Reuses existing `BlpRgbReader` + `BCnEncoder`; no parser rewrite. |
| One Phase at a Time | PASS | Phases below are independently validatable. |
| Bite-Sized Plans | PASS | ≤10 steps per phase, one concern each. |
| Data Policy | PASS | No distribution of authored assets; restoration trained on locally generated pairs. |

## Project Structure

### Documentation (this feature)

```text
specs/125-minimap-dxt1-inversion/
├── spec.md              # Feature spec (updated 2026-08-02)
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (speckit-tasks)
```

### Source Code (repository root)

```text
wow-viewer/src/core/WowViewer.Core.IO/
├── Blp/
│   ├── BlpRgbReader.cs                 # EXISTING decode wrapper (unchanged)
│   └── Dxt1TileCodec.cs                # NEW: encode/decode cycle + round-trip check (FR-002, FR-014)
├── Maps/
│   ├── MinimapComparisonMetrics.cs     # EXISTING (unchanged)
│   ├── MinimapEraProfile.cs            # EXISTING (unchanged)
│   ├── MinimapLightingBaseline.cs      # NEW: per-map lighting-baseline survey (FR-016)
│   └── MinimapEncodingSurvey.cs        # NEW: per-build/map encoding distribution (FR-013)

wow-viewer/tools/harvest/WowViewer.Tool.Harvest/
└── Program.cs                          # MODIFIED: --dxt1-parity, --lighting-baseline, parity-aware score

wow-viewer/data-harvester/
├── scripts/
│   └── train_v20_dxt1_restore.py       # NEW: restoration model training (FR-007..FR-012)
└── src/harvester/
    └── dxt1_restore.py                 # NEW: model def + inference + hallucination gate

wow-viewer/tests/WowViewer.Core.Tests/
├── Dxt1TileCodecTests.cs               # NEW: round-trip, parity, idempotency
└── MinimapLightingBaselineTests.cs     # NEW: baseline detection + accounting
```

**Structure Decision**: Library-first. The DXT1 cycle and lighting-baseline survey are shared
`WowViewer.Core.IO` capabilities; the harvest tool wires them into `synthetic-minimap`; the
restoration model is a data-harvester residual network. This matches the constitution's
library-first and residual-model-chain principles.

## Complexity Tracking

> No constitution violations. No complexity justification required.

## Phase 0 — Research

See [research.md](./research.md). Key decisions:

- **DXT1 encoder**: reuse `BCnEncoder.Net` (`CompressionFormat.Bc1`) already referenced by
  `WowViewer.Core.IO`. No new dependency. Round-trip check (FR-014) validates our encoder against
  authored bytes.
- **Lighting baseline**: measure per-map mean/std luma across authored tiles; a shared baseline is
  present when cross-tile variance is small relative to within-tile variance. Account for it by
  normalising synthetic tiles to the authored baseline before scoring.
- **Restoration**: single residual network (predicts pristine − encoded), trained on locally
  generated pairs, gated on hallucination (re-encode agreement + unsupported-detail fraction).

## Phase 1 — Design & Contracts

See [data-model.md](./data-model.md) and [contracts/](./contracts/). Key contracts:

- `Dxt1TileCodec.EncodeDecode(Image<Rgba32>) -> Image<Rgba32>` — the parity cycle.
- `Dxt1TileCodec.RoundTripAgreement(byte[] authoredBlp, Image<Rgba32> decoded) -> float` — FR-014.
- `MinimapLightingBaseline.Survey(IEnumerable<byte[,,]> authoredTiles) -> LightingBaselineResult` — FR-016.
- `synthetic-minimap --dxt1-parity` — emit `*_dxt1.png` parity companion per tile (FR-015).
- `synthetic-minimap --lighting-baseline --authored-reference` — report per-map baseline (FR-016).

## Phase 2 — Tasks

See [tasks.md](./tasks.md) (generated by speckit-tasks).
