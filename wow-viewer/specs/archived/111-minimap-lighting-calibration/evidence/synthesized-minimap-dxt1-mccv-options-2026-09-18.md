# Spec 111 Evidence — Synthesized-Minimap DXT1 + MCCV Options

Date: 2026-09-18

## Scope

Operator request: the synthesized-minimap export must expose (a) an option to **skip DXT1
compression** (later-era minimaps differ too much from the DXT1 codec floor) and (b) an option to
**include MCCV** vertex colors.

## What changed

### DXT1 option (UI exposure of an existing harvest flag)

The harvest tool already supported `--no-dxt1`; it was not reachable from the viewer dialog.

- [`ViewerApp_SynthesizedMinimapExport.cs`](../../../../src/viewer/WoWViewer/ViewerApp_SynthesizedMinimapExport.cs):
  added `_synthesizedMinimapDxt1` (default `true`), an "Apply DXT1 compression" checkbox, and
  forwarding of `--no-dxt1` when unchecked.

### MCCV option (new)

- [`TerrainMinimapCompositor.cs`](../../../../src/core/WowViewer.Core.IO/Maps/TerrainMinimapCompositor.cs):
  added `ApplyMccv` to `TerrainMinimapLighting` (default `false`) and a `ResolveMccvTint` helper that
  multiplies the composed albedo by `clamp(mccv * 2, 0, 2)` — matching the terrain shader's
  `tintColor = clamp(vertexColor.rgb * 2, 0, 2)` (neutral 127/255 → 1.0). Applied only in the main
  composition loop, not the textureless-residual path.
- [`WowViewer.Tool.Harvest/Program.cs`](../../../../tools/harvest/WowViewer.Tool.Harvest/Program.cs):
  added `--mccv`, threaded through `ResolveSyntheticMinimapLighting(..., applyMccv)`.
- [`ViewerApp_SynthesizedMinimapExport.cs`](../../../../src/viewer/WoWViewer/ViewerApp_SynthesizedMinimapExport.cs):
  added `_synthesizedMinimapMccv` (default `false`), an "Apply MCCV vertex colors" checkbox, and
  forwarding of `--mccv`.

Both default to the prior behavior (DXT1 on, MCCV off), so existing exports are unchanged.

## Verification

| Command | Exit | Output |
|---|---:|---|
| `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug` | 0 | `Build succeeded. 0 Error(s)` |
| `dotnet build wow-viewer/tools/harvest/WowViewer.Tool.Harvest/WowViewer.Tool.Harvest.csproj -c Debug` | 0 | `Build succeeded. 0 Error(s)` |

## Proof boundary

Source + build proof only. **No runtime export or visual proof is claimed** — the operator should
run an export with each option toggled and confirm the output.

## Still open (same operator report)

Liquids (ocean/other layers) render with **grid lines/omissions** in synthesized minimaps. The
unified liquid mask ([`AdtTensorPackBuilder.BuildUnifiedLiquid`](../../../src/core/WowViewer.Core.IO/Maps/AdtTensorPackBuilder.cs:4402))
is a clean per-vertex 257×257 field and the compositor bilinearly samples it, so the artifact is not
obviously in either. Root cause is **not yet established** — needs a zoomed capture and the exact
map/era to reproduce before any fix is attempted (guessing here would risk a wrong change).