namespace AlphaWdtAnalyzer.Core.Terrain;

/// <summary>
/// MCSH shadow map data for a single chunk (64×64 bit shadow bitmap)
/// </summary>
public record McnkShadowEntry(
    string Map,
    int TileRow,
    int TileCol,
    int ChunkRow,
    int ChunkCol,
    bool HasShadow,
    int ShadowSize,
    string ShadowBitmapBase64
);
