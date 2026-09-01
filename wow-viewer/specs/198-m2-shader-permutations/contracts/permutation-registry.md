# Contract: Permutation Registry and Selection

**Date**: 2026-09-01

This feature exposes no network or CLI surface. Its contract is the in-process boundary
between the renderers and `WowViewer.Core.Renderer`. Stated as behaviour, so the tests in
`PermutationRegistryTests` can pin it directly.

## C1 — A request never carries a guess

```
PermutationRequest is valid  ⟺  (Permutation is null) XOR (UnresolvedReason is null)
```

Constructing a request with both, or neither, is a programming error and throws. This is
FR-002 made structural: there is no representable state in which a batch carries a
permutation that was not resolved from data.

## C2 — Every request yields a program

```
Resolve(request) returns a non-null program, always.
```

| Condition | Result | Counted as |
|---|---|---|
| `Enabled == false` | `FallbackProgram` | — (switch is off, not a shortfall) |
| `request.Permutation == null` | `FallbackProgram` | `Unresolved` |
| permutation not implemented | `FallbackProgram` | `FellBackNotImplemented` |
| compile previously failed | `FallbackProgram` | `FellBackCompileFailed` |
| otherwise | compiled program | `Implemented` |

No path throws on an unknown or out-of-range permutation (FR-003, and the first edge case in
the spec).

## C3 — Compile at most once per permutation

```
For any permutation p, across the renderer's lifetime:
    count(compile calls for p) <= 1
```

Holds regardless of how many batches, frames, or models request `p`, and regardless of
whether the first compile succeeded. A failed compile is cached as failed and is not retried
(FR-005, and the "shader compilation fails on a driver" edge case).

## C4 — The off switch is total

```
Enabled == false  ⟹  rendering output is byte-identical to the pre-feature renderer
```

Not merely "similar" — SC-005 verifies this by pixel comparison. The switch must therefore
bypass selection entirely rather than select-then-discard.

## C5 — Enum ordinals are the client's indices

```
(int)M2PixelShader.CombinersOpaque              == 0
(int)M2PixelShader.Illum                        == 34
(int)M2PixelShader.CombinersOpaqueMod2xNAAlpha  == 12
```

Pinned by test against research.md R2. These are not arbitrary identifiers — reordering or
alphabetising the members silently changes what every model renders as. The test exists to
make that a build failure rather than a visual mystery.

## C6 — The report is complete and bounded

```
Requested.Values.Sum() == Implemented + FellBackNotImplemented
                        + FellBackCompileFailed + Unresolved
```

Every request lands in exactly one bucket. `UnresolvedModels` is deduplicated and capped;
when the cap truncates, the report says so rather than silently omitting — a truncated
offender list that looks complete is the failure mode this clause prevents.
