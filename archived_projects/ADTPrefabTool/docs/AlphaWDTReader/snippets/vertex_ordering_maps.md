# Vertex Ordering Maps (Alpha ↔ 3.x)

- Purpose: Re-map Alpha `MCVT` 145 heights (outer81 + inner64) to 3.x order.
- References: `lib/gillijimproject/wowfiles/alpha/McnkAlpha.cpp`, `.../lichking/McnkLk.cpp`.

## Indices
- 145 vertices = 9×9 outer grid (81) + 8×8 inner grid (64).
- Alpha encodes [outer81, inner64]; 3.x expects interleaved diamond pattern.

## Strategy
- Precompute an `int[145] alphaTo3x` map.
- Apply map to heights and normals consistently.
- Keep absolute heights; do not renormalize.

```csharp
// pseudo
for (int i = 0; i < 145; i++) outHeights[i] = inHeights[alphaTo3x[i]];
```
