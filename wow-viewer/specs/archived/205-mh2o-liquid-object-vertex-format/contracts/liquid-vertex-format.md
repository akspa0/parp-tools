# Contract: Liquid Vertex-Format Resolution

**Date**: 2026-09-01

Stated as behaviour so the decode tests can pin it without a client.

## C1 — The field has two meanings and the boundary is 42

```
liquid_object_or_lvf <  42  ⟹  it IS the liquid vertex format (0-3)
liquid_object_or_lvf >= 42  ⟹  it is a LiquidObject.dbc id; the format must be resolved
```

Pre-Cata data only ever produces the first case, which is why this went unnoticed until 5.x
(FR-003). The measured MoP corpus produces **only** the second.

## C2 — Resolution is a lookup chain, never a guess

```
LiquidObject.id -> LiquidTypeID -> LiquidType.MaterialID -> LiquidMaterial.LVF
```

The float-plausibility probe that diagnosed this misclassified 18 of 6,194 ocean layers. It is an
instrument, not a decoder (FR-008). It may report disagreement with the chain; it may not stand in
for it.

## C3 — An unresolved format is a reported outcome

```
resolution fails ⟹ the layer is counted, its id logged, and the fallback stated
```

The absence of this is the whole defect: a `switch` with no `default` turned a 100%-unhandled
encoding into plausible-looking flat water, with no error, no counter and no log. A silent fallback
is not permitted here again, whatever the fallback is.

## C4 — Depth-only stays flat

```
format is depth-only ⟹ surface is flat at the declared level
```

Ocean is 17,317 of the 17,461 measured layers and is **correct today**. A change that makes ocean
non-flat is a regression, not a fix.

## C5 — Height-bearing layers use their vertices, not their header

```
format carries heights ⟹ per-vertex heights are used
```

And specifically not `minHeight`: in every measured layer whose vertices vary, the lowest vertex
disagrees with the header by more than 0.5 units. The header is a bound, not the surface.

## C6 — Both decode paths agree

```
Mh2oChunk.Parse(adt)  ≡  AdtLiquidReader.Parse(adt)     for per-vertex heights
```

Two independent decoders carried this defect independently (research R5). One implementation is
preferred; if two entry points survive, this equivalence is enforced by test, not by intent
(FR-007).

## C7 — A missing DBC degrades, it does not crash

```
DBC absent ⟹ current behaviour, reported as degraded
```

Clients without `LiquidObject.dbc` are legitimate inputs (FR-009, Constitution VI). Degraded output
must be visible as degraded, per C3.
