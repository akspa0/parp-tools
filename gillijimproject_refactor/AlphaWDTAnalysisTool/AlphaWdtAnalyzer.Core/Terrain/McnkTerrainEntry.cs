namespace AlphaWdtAnalyzer.Core.Terrain;

/// <summary>
/// Complete MCNK terrain data for a single chunk (32×32 game units within a tile)
/// </summary>
public record McnkTerrainEntry(
    string Map,
    int TileRow,
    int TileCol,
    int ChunkRow,
    int ChunkCol,
    uint FlagsRaw,
    bool HasMcsh,
    bool Impassible,
    bool LqRiver,
    bool LqOcean,
    bool LqMagma,
    bool LqSlime,
    bool HasMccv,
    bool HighResHoles,
    int AreaId,
    int NumLayers,
    bool HasHoles,
    string HoleType,
    string HoleBitmapHex,
    int HoleCount,
    float PositionX,
    float PositionY,
    float PositionZ
);
