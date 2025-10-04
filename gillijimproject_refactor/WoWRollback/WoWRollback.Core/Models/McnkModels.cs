namespace WoWRollback.Core.Models;

/// <summary>
/// Complete MCNK terrain data for a single chunk
/// Matches CSV output from AlphaWDTAnalysisTool
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

/// <summary>
/// MCSH shadow map data for a single chunk
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

/// <summary>
/// AreaID boundary between two areas
/// </summary>
public record AreaBoundary(
    int FromArea,
    string FromName,
    int ToArea,
    string ToName,
    int ChunkRow,
    int ChunkCol,
    string Edge  // "north", "east", "south", "west"
);

/// <summary>
/// Area table lookup data
/// </summary>
public class AreaTableLookup
{
    private readonly Dictionary<int, string> alphaAreas;
    private readonly Dictionary<int, string> lkAreas;

    public AreaTableLookup(Dictionary<int, string> alphaAreas, Dictionary<int, string> lkAreas)
    {
        this.alphaAreas = alphaAreas;
        this.lkAreas = lkAreas;
    }

    public string GetName(int areaId, bool preferAlpha = true)
    {
        if (!preferAlpha && lkAreas.TryGetValue(areaId, out var lkPreferred))
        {
            return lkPreferred;
        }

        if (preferAlpha && alphaAreas.TryGetValue(areaId, out var alphaName))
        {
            return alphaName;
        }

        if (lkAreas.TryGetValue(areaId, out var lkName))
        {
            return lkName;
        }

        return $"Unknown Area {areaId}";
    }

    public (string? alphaName, string? lkName) GetBothNames(int areaId)
    {
        var alphaName = alphaAreas.TryGetValue(areaId, out var a) ? a : null;
        var lkName = lkAreas.TryGetValue(areaId, out var l) ? l : null;

        return (alphaName, lkName);
    }

    public string? GetAlphaName(int areaId)
    {
        return alphaAreas.TryGetValue(areaId, out var name) ? name : null;
    }

    public string? GetLichKingName(int areaId)
    {
        return lkAreas.TryGetValue(areaId, out var name) ? name : null;
    }

    public IEnumerable<int> GetAllAreaIds()
    {
        return alphaAreas.Keys.Union(lkAreas.Keys).Distinct();
    }
}
