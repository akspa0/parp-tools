# Quickstart: Feature 151

## Structural validation

From PowerShell 7:

```powershell
dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter "FullyQualifiedName~Wmo|FullyQualifiedName~GameMode|FullyQualifiedName~ViewerSurface|FullyQualifiedName~Diagnostic"
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```

## Phase 1 checkpoint evidence

The bounded portal decision is implemented in the shared runtime and used by the WMO renderer.
The focused structural proof is:

```powershell
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj `
  -c Debug --no-build `
  --filter "FullyQualifiedName~WmoPortalVisibilityDecisionTests|FullyQualifiedName~WorldScenePortalVisibilityEvaluatorTests"
```

Result on 2026-08-14: 16 passed, 0 failed. The full solution Debug build also passes with 0 errors.
Existing repository warnings, including the Snappier advisory, remain. This is source-level proof;
the user still owns real-client visual, submission-time, and FPS comparison.

## Controlled renderer comparison

1. Use a persisted approved client root such as `H:\CLIENTS` and record the exact build/fingerprint.
2. Load one WMO fixture with multiple groups and valid portal geometry.
3. Run the same camera positions with portal visibility enabled and with the existing conservative
   route selected for comparison.
4. Capture WMO visibility time, WMO submission time, visible group count, portal tests, fallback
   reason, and draw counts. Do not infer FPS from unit tests.

## Game mode smoke path

1. Load a map with a selected/spawned character model.
2. Start in `SimpleInteractive`; verify game mode is off.
3. Enable game mode, verify the head-anchor source, walk, run, jump, and disable game mode.
4. Verify the editor camera returns to its pre-game-mode pose.

## Proof boundary

Compilation and focused tests establish code-level behavior only. Real-client visual, GPU/FPS,
audio, and long-duration interaction proof remains a user-owned handoff and must record the client
root, build, map/asset fixture, and selected surface/diagnostic profile.
