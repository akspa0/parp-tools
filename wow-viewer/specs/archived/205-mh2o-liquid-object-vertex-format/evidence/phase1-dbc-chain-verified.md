# Phase 1 gate: the DBC chain, verified against the client

**Date**: 2026-09-01
**Client**: `C:\WoW4-data\MoPBeta`, build **5.0.1.15464**, map `HawaiiMainLand`, 80 root ADTs, 17,461 layers
**Command**:

```
inspect adt liquid-formats --client "C:\WoW4-data\MoPBeta" --map HawaiiMainLand --limit 80 --build 5.0.1.15464
```

Research R7 said the chain was *"documented on wowdev and is **not** verified against this client"*
and made verification task one. It is now verified, and **two things in the research were wrong**.

## Correction 1 — the "≥ 42" threshold does not hold for this client

`LiquidObject` in 5.0.1.15464 has **1244 rows with real ids 57..2390** (lowest twelve: 57, 58, 260,
263, 264, 265, 266, 281, 282, 283, 284, 285).

**42 is genuinely absent from the table.**

So the wiki rule "values at or above 42 are a LiquidObject id" is a heuristic whose magic number
comes from some other build. Applying it here routes all **17,317** ocean layers into the chain,
where they would resolve through whatever row happened to sit nearby and decode their **depth bytes
as floats**. Ocean's 42 must stay *unresolved*, which keeps the flat plane — and R2 already
established that flat is **correct** for ocean.

The threshold is kept in the code as the lower bound of the *candidate* range, but membership in the
table is what actually decides. A value at or above the threshold that is not a row is unresolved,
never guessed.

## Correction 2 — the first run reported the exact inverse, and it was our bug

The first gate run said 42 resolved (to LiquidType 5 / Material 1) and that 2325/2333/2372 were
absent. That is backwards, and the cause was in the reader, not the client:

**`DBCDRow.ID` is a positional key for these WDB2 tables, not the row id.**

| table | positional keys | real ids (`ID` column) |
|---|---|---|
| `LiquidObject` | 1..1244 | 57..2390, sparse |
| `LiquidMaterial` | 1..7 | {1, 2, 3, 4, 5, 8, 10} |
| `LiquidType` | 1..59 | 1..809, sparse |

`storage[42]` returned a row whose `ID` column read **316**. Keying `map[row.ID]` therefore built a
table indexed by row *order*: it silently resolved the wrong row for every sparse id and reported
every id above the row count as absent. It did not throw and it did not look empty.

Fixed by `DbcTableLoader.ResolveRowId`, which keys on the detected `ID` column.

**The dump is why this was caught.** A chain that returns a confident wrong answer is
indistinguishable from a correct one until you print the table you resolved against.

## What the verified chain resolves

```
value  layers  ->liquidType  ->material  ->LVF                 source             probe agreement
   42   17317        -            -       -                    UNRESOLVED         n/a
 2325     104        5            1       0 HeightDepth        LiquidObjectChain  agree
 2333      17        5            1       0 HeightDepth        LiquidObjectChain  agree
 2372      23        5            1       0 HeightDepth        LiquidObjectChain  agree
```

The `probe agreement` column is R6's cross-check: the float-plausibility probe is an instrument, not
the decoder, and its only job is disagreement detection. It **agrees** on all 144 river layers —
the DBC says "heights" and the bytes read as heights, from two independent routes.

## Correction 3 — this client's LiquidMaterial cannot produce LVF 2 or 3

All 7 rows: ids {1,2,3,4,5,8,10} with LVF {0,1,0,1,0,0,0}. **Only LVF 0 and 1 exist.** There is no
depth-only material, which is consistent with ocean not going through the chain at all.

## Bonus finding — `DbcLiquidTypeTable`'s "Type field at 0x38" is `MaterialID`

`DbcLiquidTypeTable` reads field index 14 (byte offset `0x38`) and calls it `Type`, mapping
1=Water/2=Magma/3=Slime. Per WoWDBDefs, field 14 in the 3.1.0–5.4.8 `LiquidType` layout is
**`MaterialID`**; the actual liquid family field is index 3 (`SoundBank`, 0=water/1=ocean/2=magma/
3=slime). The existing mapping produces plausible answers because the material ids happen to line up,
which is exactly the failure mode in `feedback_a_name_stops_the_looking`.

Not changed here — it is out of this spec's scope and its current output is not known to be wrong in
practice — but it is recorded, and it is why `LiquidVertexFormatChain` reads `MaterialID` through
DBCD by column name rather than reusing that offset.

## Gate verdict

**PASS.** The chain is real, the offsets are right, the resolution is measured rather than assumed,
and the two corrections above are recorded before any code depended on them.
