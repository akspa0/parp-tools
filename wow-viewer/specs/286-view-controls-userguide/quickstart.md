# Verification quickstart — Spec 286

Prepared checks, not executed implementation proof. Run from wow-viewer using PowerShell 7.

```powershell
dotnet build ./WowViewer.slnx -c Debug
dotnet test ./tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter 'FullyQualifiedName~ViewerUiLayout'
pwsh -NoProfile -File ./eng/ui/Verify-UiReference.ps1
```

Add new phase policy/state tests to the filter when named; layout tests alone do not prove
scale persistence, activation, semantic categories or runtime appearance.

## Operator-owned cards
1. **U0 / SC-008/010:** empty/loaded scene, narrow and 1080p/1440p/4K, default/custom scales.
   No footer scrollbars; FPS/audio actions fit; FPS hover exposes pending/debug fields inside
   client bounds; tooltip/popup does not move camera or select through UI.
2. **U1 / SC-001–003:** change scale from View, use each reset, open tools, restart. Verify exact
   reset effects, persistent consistent text, loaded scene retained and high-DPI sizing preserved.
3. **U2 / SC-004/005:** first/replacement load, Maps open/closed, refresh after close, cancel,
   failure, successful-empty and stale discovery. One current Maps window per success only.
4. **U3/U4 / SC-009/010:** named/unnamed, known-tree, WMO, NPC and unknown selections. Readable
   top-middle identity, matching header/bounds category, no fake HP/stale identity/input leakage.
5. **U5 / SC-006/007:** genuine running-build shell/View/reset/Maps/spawn/FPS/selection/workbench
   images; reviewed captions and end-to-end guide walkthrough.

Receipts live below this spec's evidence directory: changed files, exact commands/exit codes,
revision/version, configured client/build/fingerprint when applicable, display/scale, reproduction,
criterion→real-output table and relative reviewed images. No bundled game data/private paths in
published guide; missing visual proof remains open.

Next: generate unchecked tasks from [plan](plan.md), implement U0 only, stop at its gate.
Renderer execution remains separately owned by [249](../249-epic-renderer-performance-and-correctness/quickstart.md).
