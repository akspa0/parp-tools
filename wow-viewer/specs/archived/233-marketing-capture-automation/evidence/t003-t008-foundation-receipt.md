# T003–T008 Foundation Receipt — 2026-09-08

## Files changed

- `src/core/WowViewer.Core.Runtime/Marketing/FeatureTourRecipe.cs`
- `src/core/WowViewer.Core.Runtime/Marketing/MarketingCaptureOutputPolicy.cs`
- `src/core/WowViewer.Core.Runtime/Marketing/AuthoringHandoff.cs`
- `src/core/WowViewer.Core.Runtime/Marketing/BuiltinFeatureTourRecipes.cs`
- `tests/WowViewer.Core.Tests/MarketingCapture/FeatureTourRecipeTests.cs`
- `tests/WowViewer.Core.Tests/MarketingCapture/MarketingCaptureOutputPolicyTests.cs`

## Test-first and verification record

| Command | Exit | Real output |
|---|---:|---|
| `dotnet test tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~MarketingCapture"` before source | 1 | Expected compile failure: `WowViewer.Core.Runtime.Marketing` and recipe/output types did not exist. |
| `dotnet test tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~MarketingCapture"` after source | 0 | `Passed: 12, Failed: 0, Skipped: 0`. |

The surrounding project emits pre-existing `NU1903` dependency warnings; the focused result itself is green.

## Criterion-to-evidence mapping

| Criterion | Source/test evidence | Scope of proof |
|---|---|---|
| FR-001 named/versioned recipes and ordered valid beats | `FeatureTourRecipe`, `FeatureTourRecipeTests` | Deterministic source contract. |
| FR-002 recipe/output validation before viewer state changes | `FeatureTourRecipeValidator`, `MarketingCaptureOutputPolicyTests` | Pure validation only; encoder/client checks remain in existing viewer path and need runtime witness. |
| FR-008 safe versioned external handoff | `AuthoringHandoffFactory`, containment tests | Descriptor-only; no transport call. |
| FR-009 no undocumented Comfy call/path leakage | output policy converts to relative paths; JSON test asserts no managed root in handoff | Source contract only. |
| SC-003 / SC-004 deterministic validation and root refusal | 9 foundation tests within the final 12 focused tests | Does not establish external automation runtime. |

No capture, performance, or ComfyUI behavior was run for this phase.

