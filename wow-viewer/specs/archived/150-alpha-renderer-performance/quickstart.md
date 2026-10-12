# Quickstart: Alpha 0.5.3 Renderer Performance Evidence

This lane is evidence-first. Do not change renderer behavior until the baseline and native evidence
rows identify one owner.

## Local source proof

From `I:\parp\parp-tools\wow-viewer`:

```powershell
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --no-restore
git diff --check
```

## Production profile

`profile-render` is the existing production OpenGL path. Run it against a configured approved
0.5.3.3368 client root and a fixed map/tile. Replace the placeholder values; do not hardcode them in
source or commit proprietary reports.

```powershell
$ClientRoot = 'H:\CLIENTS\Vanilla\0.x\0_5_3_3368\World of Warcraft'
$Wdt = Join-Path $ClientRoot 'World\Maps\Azeroth\Azeroth.wdt'
$Output = 'output\diagnostics\alpha053-render-baseline.json'

dotnet run --project tools/validation-capture/WowViewer.Tool.ValidationCapture/WowViewer.Tool.ValidationCapture.csproj -- `
  profile-render `
  --client-root $ClientRoot `
  --map-input $Wdt `
  --output $Output `
  --build '0.5.3.3368' `
  --tile-x 31 `
  --tile-y 31 `
  --warmup-frames 30 `
  --frames 120
```

Repeat the command without source changes and record variance. For an explicit residency stress
comparison, use `--load-all-tiles` only as a separate stress case; it is not the normal navigation
baseline.

## Native Ghidra evidence

In the already-open read-only 0.5.3 `WoWClient.exe` program, inspect one bounded question at a time:

1. Locate the world render entry and identify whether it dispatches terrain, objects, liquids, and
   overlays as separate passes.
2. Locate terrain/chunk admission and submission; record the inputs used for distance/frustum or
   visibility rejection.
3. Locate WMO/M2/MDX admission and any draw-distance/LOD/state bucket decisions.
4. Locate resource creation/reuse and material/texture state setup.
5. Locate far-horizon or reduced-detail terrain behavior, if present.

Add only behavior/anchor/confidence rows to
`memory-bank/workstream-alpha053-renderer-performance.md`. Do not copy original code or treat an
address-only hypothesis as proven.

## A/B handoff

For a candidate experiment, run the same profile with the old path and the candidate path, then record:

- selected owner and before/after stage timing;
- visible/culled/submitted/batch/fallback counts;
- GPU timing state;
- visual gate status;
- fallback reason and decision.

The user owns the interactive native-client versus viewer FPS/visual comparison. The agent must report
that proof separately from tests, build, and headless profile output.
