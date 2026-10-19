using System.Numerics;

namespace WowViewer.Core.Population;

/// <summary>
/// Transposes Blizzard internal development test fixtures that originally resided in the
/// upper-left and upper-right corners of Azeroth (Map 0) in 0.5.3-0.5.5 Alpha builds into
/// Map 451 ('development'), matching the historical migration Blizzard executed around version 1.0.0.
/// </summary>
public static class AlphaCoreCornerTransposer
{
    public const float MapOrigin = 17066.666f;
    public const float ChunkSize = 533.33333f;
    public const int TargetMapId = 451;

    /// <summary>
    /// Computes ADT tile index (0..63) for the given WoW X/Y coordinates.
    /// </summary>
    public static (int TileX, int TileY) WorldToTile(float x, float y)
    {
        int tx = (int)MathF.Floor((MapOrigin - x) / ChunkSize);
        int ty = (int)MathF.Floor((MapOrigin - y) / ChunkSize);
        return (tx, ty);
    }

    /// <summary>
    /// Evaluates whether a given spawn position on Map 0 is within the historical prototype
    /// corner test zones (upper-left: tiles (0..2, 0..2); upper-right: tiles (0..2, 60..63)).
    /// </summary>
    public static bool IsCornerTestFixture(int mapId, float x, float y)
    {
        if (mapId != 0)
            return false;

        var (tx, ty) = WorldToTile(x, y);

        // Upper-left corner test area (e.g. tiles (0,0) and (1,1))
        bool isUpperLeft = tx >= 0 && tx <= 2 && ty >= 0 && ty <= 2;

        // Upper-right corner test area (e.g. tile (1,62))
        bool isUpperRight = tx >= 0 && tx <= 2 && ty >= 60 && ty <= 63;

        return isUpperLeft || isUpperRight;
    }

    /// <summary>
    /// Transposes a prototype Map 0 corner spawn into a Map 451 population record.
    /// Preserves authentic world coordinates, orientation, scale, and entry ID while
    /// tagging the record with IsTransposed = true and source lineage.
    /// </summary>
    public static PopulationRecord TransposeToMap451(PopulationRecord spawn)
    {
        ArgumentNullException.ThrowIfNull(spawn);

        var (tx, ty) = WorldToTile(spawn.Position.X, spawn.Position.Y);
        string cornerName = ty >= 60 ? "upper-right" : "upper-left";

        return new PopulationRecord
        {
            SpawnId = spawn.SpawnId,
            EntryId = spawn.EntryId,
            MapId = TargetMapId,
            SpawnType = spawn.SpawnType,
            Name = spawn.Name,
            Subname = spawn.Subname,
            ModelPath = spawn.ModelPath,
            Scale = spawn.Scale,
            DisplayScale = spawn.DisplayScale,
            EffectiveScale = spawn.EffectiveScale,
            Position = spawn.Position,
            Orientation = spawn.Orientation,
            Rotation = spawn.Rotation,
            Faction = spawn.Faction,
            Flags = spawn.Flags,
            GameObjectType = spawn.GameObjectType,
            IsTransposed = true,
            SourceNote = $"Transposed from Alpha-Core Map 0 {cornerName} corner tile ({tx}, {ty}) (original GUID {spawn.SpawnId})"
        };
    }

    /// <summary>
    /// Scans a collection of population records, selects all prototype corner fixtures,
    /// and returns their transposed Map 451 representations.
    /// </summary>
    public static List<PopulationRecord> TransposeAllCornerSpawns(IEnumerable<PopulationRecord> spawns)
    {
        ArgumentNullException.ThrowIfNull(spawns);

        var result = new List<PopulationRecord>();
        foreach (var spawn in spawns)
        {
            if (IsCornerTestFixture(spawn.MapId, spawn.Position.X, spawn.Position.Y))
            {
                result.Add(TransposeToMap451(spawn));
            }
        }

        return result;
    }
}
