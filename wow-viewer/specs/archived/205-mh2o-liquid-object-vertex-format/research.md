# Phase 0 Research: MH2O LiquidObject Vertex-Format Resolution

**Date**: 2026-09-01

**Method**: a purpose-built corpus probe, `inspect adt liquid-formats`, run against the operator's
own MoP Beta client. Everything below is measured, not inferred from the symptom.

## R1 — The measurement

```
adt liquid-formats --client "C:\WoW4-data\MoPBeta" --map HawaiiMainLand --limit 80
```

```
root ADTs scanned=80  withMh2o=80  liquidLayers=17461

value  layers  withVertexData  sloped  maxSpread  size
   42   17317            6194       0       0.00  8x8
        vertexBlocks: plausibleHeights=18 implausible=6008 varying=9  maxSpread=8192.00 disagreeWithHeader=1
 2325     104             104       0       0.00  8x8
        vertexBlocks: plausibleHeights=104 implausible=0  varying=6  maxSpread=11.90  disagreeWithHeader=6
 2333      17              17       0       0.00  8x8
        vertexBlocks: plausibleHeights=17  implausible=0  varying=10 maxSpread=70.15  disagreeWithHeader=10
 2372      23              23       0       0.00  8x8
        vertexBlocks: plausibleHeights=23  implausible=0  varying=11 maxSpread=163.25 disagreeWithHeader=11
```

**100% of layers carry a value ≥ 42.** Not one is a vertex format in 0–3.

The tool lives at `tools/inspect/WowViewer.Tool.Inspect/AdtLiquidFormatSupport.cs` and should be
re-run after the fix as the SC-001 check.

## R2 — Ocean is not broken; rivers are

Reading each vertex block as `(w+1)·(h+1)` floats separates the two populations cleanly:

- **id 42 (liquidType 2, ocean)**: 6,008 of 6,194 blocks contain values that are not finite,
  in-range floats. The block is **depth bytes**, so this is a depth-only format and flat rendering
  is **correct**. 11,123 further layers carry no vertex block at all.
- **ids 2325 / 2333 / 2372 (liquidType 5)**: **144 of 144** blocks read as plausible heights, and
  they *vary* — spreads of 11.90, 70.15 and 163.25 world units. These are real sloped surfaces.

This is why the defect reads as "waterways are flat": it is specifically the waterways.

## R3 — The flat plane is also at the wrong height

`disagreeWithHeader` counts layers whose lowest decoded vertex differs from the header `minHeight`
by more than 0.5 units. For the river ids that is **6 of 6, 10 of 10, and 11 of 11** of the varying
layers.

So the substituted flat plane is not even the surface's true low point. Each chunk flattens to its
own wrong value, and adjacent chunks therefore **step against each other** — which is exactly the
seam the operator reported, and it explains why the two symptoms always appear together.

## R4 — The defect, in one line of each decoder

```csharp
AdtLiquidVertexFormat vertexFormat = (AdtLiquidVertexFormat)BinaryPrimitives.ReadUInt16LittleEndian(...);
...
switch (vertexFormat)
{
    case AdtLiquidVertexFormat.HeightDepth:    ...
    case AdtLiquidVertexFormat.HeightUv:       ...
    case AdtLiquidVertexFormat.DepthOnly:      ...
    case AdtLiquidVertexFormat.HeightUvDepth:  ...
}   // no default
```

A value of 42 or 2325 matches nothing, `heights` stays `null`, and the consumer falls back to the
header level. **No error, no counter, no log.** An unhandled encoding became plausible-looking
output, which is why it survived; that is what US3 exists to prevent recurring.

## R5 — There are two decoders and only one is in the render path

| Decoder | Location | Used by |
|---|---|---|
| `Mh2oChunk.Parse` | `Core.IO/Liquids/Mh2oChunk.cs` | **`StandardTerrainAdapter` — the viewer**, plus `Vlm/LiquidService`, `VlmDatasetExporter` |
| `AdtLiquidReader.ParseLayer` | `Core.IO/Maps/AdtLiquidReader.cs` | `LkAdtReader`, `AdtTensorPackBuilder`, `LiquidBasicTypePackBuilder`, `WorldLiquidTileBuilder`, the converter |

**Both carry the defect independently.** Fixing `AdtLiquidReader` alone changes nothing the operator
can see; fixing `Mh2oChunk` alone leaves every harvested dataset wrong.

This is the same shape as the `MdxRenderer` / `M2Renderer` trap hit earlier the same day — a fix
landed in the class that was easiest to find, and the render path used the other one. Check which
decoder the path under test actually calls **before** concluding a fix did nothing.

## R6 — Why the diagnostic probe must not become the decoder

The float-plausibility test that made R2 possible is a good *instrument* and a bad *decoder*. It
misclassified **18 of 6,194** ocean layers (0.3%) as height-bearing. As a decode rule that is bogus
geometry on real water, intermittently, in a way that would be blamed on something else later.

The correct resolution is the DBC chain (FR-002). The probe may stay as a cross-check that reports
disagreement with the DBC answer — which is genuinely useful, because a disagreement means one of
the two is wrong and the corpus can say which.

## R7 — What is not yet verified

The chain `LiquidObject.dbc → LiquidTypeID → LiquidType.dbc → MaterialID → LiquidMaterial.dbc → LVF`
is documented on wowdev and is **not** verified against this client. Neither `LiquidObject` nor
`LiquidMaterial` has a reader in this repo; `DbcLiquidTypeTable` reads `LiquidType.dbc` only, and
only its `Type` field at `0x38`.

**Task one is to confirm the field offsets against the MoP DBCs**, not to trust the wiki — the
project has been burned by inherited names before. The measurement in R1–R3 is independent of that
and stands either way: the field is an id, and the river heights exist and are being discarded.
