# Spec 271 Phase A receipt (2026-10-11)

## Files changed
`eng/Version.props`, `docs/releases/v0.6.0.md`, `CHANGELOG.md`, `README.md` (root + wow-viewer), `docs/WoWViewer/README.md`,
`docs/WoWViewer/USERGUIDE.md`, `specs/STATUS.md`, `specs/271-*`, `memory-bank/activeContext.md`, `memory-bank/progress.md`.

## Verification
| Command | Exit |
|---|---|
| `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` | 0 (0 errors, 736 warnings) |
| `dotnet test ...WowViewer.Core.PM4.Tests.csproj -c Debug --filter "FullyQualifiedName!~Pm4RegionObjectGrouperTests"` | 0 (108/108); grouper test alone 1/1 after `IsEmptyStubRegion` fix |

## Criterion → evidence
| Criterion | Evidence |
|---|---|
| A1 version `0.6.0` | `eng/Version.props` InformationalVersion `0.6.0`; title-bar display is operator-visual (not yet witnessed) |
| A2 docs | files above |
| A3 / A4 | recorded in the git/release step of this session |
