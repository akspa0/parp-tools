# Data Model: M2 and WMO Shader Permutation System

**Date**: 2026-09-01

All types live in `WowViewer.Core.Renderer` unless noted. No type here reads a file format;
the selector's provenance is resolved in T002 and enters through `PermutationRequest`.

## `M2PixelShader` (enum, 35 values)

Ordinal-valued, matching the client's pixel table exactly (research.md R2). Value order is
the contract — a member's numeric value IS the client index, so members must never be
reordered or alphabetised.

`CombinersOpaque = 0` … `Illum = 34`.

## `M2VertexShader` (enum, 16 values)

Same contract, for the vertex table at `0x00eb4ba0`. Populated in T003.

## `WmoShader` (enum, 6 values)

`Diffuse`, `Specular`, `TwoLayerDiffuse`, `DiffuseEmissive`, `TwoLayerDiffuseOpaque`,
`TwoLayerDiffuseEmissive`. Ordering to be decoded alongside T003; until then members carry
explicit values only where measured.

## `ShaderPermutation` (readonly record struct)

| Field | Type | Notes |
|---|---|---|
| `Vertex` | `M2VertexShader` | |
| `Pixel` | `M2PixelShader` | |
| `NativeName` | `string` | e.g. `"Diffuse_T1 / Combiners_Opaque_Mod2xNA_Alpha"` — FR-009 |
| `RequiredTextureUnits` | `int` | derived from the vertex name's `T`/`Env` count |

## `PermutationRequest` (readonly record struct)

What one batch asks for.

| Field | Type | Notes |
|---|---|---|
| `Permutation` | `ShaderPermutation?` | `null` when the selector could not be resolved |
| `UnresolvedReason` | `string?` | non-null exactly when `Permutation` is null — FR-002 |
| `ModelPath` | `string` | for the report's offender list |
| `BatchIndex` | `int` | |

**Validation**: exactly one of `Permutation` / `UnresolvedReason` is non-null. A request may
never carry a guessed permutation — that is the failure mode FR-002 exists to prevent.

## `PermutationRegistry`

Owns compiled programs. Keyed by `ShaderPermutation`.

| Member | Behaviour |
|---|---|
| `TryGetProgram(permutation, out program)` | returns the compiled program, compiling on first request (FR-005) |
| `FallbackProgram` | the existing single program; never null (FR-003) |
| `IsImplemented(permutation)` | false for pairs with no authored implementation |
| `Enabled` | when false, every request resolves to `FallbackProgram` (FR-007) |

**State transitions** per permutation: `Unknown → Requested → (Compiled | CompileFailed | NotImplemented)`.
`CompileFailed` and `NotImplemented` both route to the fallback and are counted separately —
a driver failure and an unwritten shader are different problems.

## `PermutationSelectionReport`

Per scene load. Satisfies FR-004 and SC-006.

| Field | Type |
|---|---|
| `Requested` | `IReadOnlyDictionary<ShaderPermutation, int>` |
| `Implemented` | `int` |
| `FellBackNotImplemented` | `int` |
| `FellBackCompileFailed` | `int` |
| `Unresolved` | `int` |
| `UnresolvedModels` | `IReadOnlyList<string>` (capped, deduplicated) |
| `NotImplementedPermutations` | `IReadOnlyList<ShaderPermutation>` |

Queryable from the viewer without a debugger (FR-004) — surfaced in the diagnostics panel.

## Relationships

```
Batch ──produces──▶ PermutationRequest ──looked up in──▶ PermutationRegistry ──▶ Program
                            │                                    │
                            └──────────counted in───────▶ PermutationSelectionReport
```

`M2MaterialPassProfile` remains the owner of blend/depth pipeline state and is **not** a
member of any type here. Combiner selection and pass classification are orthogonal (R6).
