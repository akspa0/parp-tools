# Alpha MCLQ Layout

Purpose: Capture Alpha-era MCLQ fields used to build water occupancy + height grid per MCNK.

References:
- docs/wowdev.wiki/Alpha.md
- lib/noggit3/src/noggit/liquid_chunk.cpp
- lib/gillijimproject/wowfiles/alpha/McnkAlpha.cpp

Key points:
- Resolution: per-chunk grid (implementation-dependent; confirm against sample files). Typically small boolean mask + per-cell (or uniform) height.
- Occupancy: bit/byte flags indicating water present per cell.
- Height: per-cell height or uniform level for the region.
- Types/flags: optional type hints (ocean/river/magma). Map conservatively to MH2O flags.

Reader output model (AlphaWater):
- int Width, Height
- bool[,] Occupied
- float[,] Level (or single float Level if uniform)
- Optional: Type/Flags

Converter usage:
- Derive connected components from Occupied.
- For each component, compute bbox and mask, carry heights into MH2O.
