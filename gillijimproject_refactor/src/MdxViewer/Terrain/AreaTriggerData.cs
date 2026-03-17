using System.Numerics;
using DBCD.Providers;
using MdxViewer.Logging;
using MdxViewer.Rendering;

namespace MdxViewer.Terrain;

/// <summary>
/// Represents an AreaTrigger from AreaTrigger.dbc.
/// Used for visualizing instance portals, event markers, and script triggers.
/// </summary>
public class AreaTriggerEntry
{
    public int Id;
    public int MapId;
    public Vector3 WoWPosition; // X, Y, Z in WoW coordinates
    public Vector3 Position;    // Transformed renderer coordinates
    public float Radius;        // Sphere radius (when > 0)
    public float BoxLength;     // Box dimensions (when radius = 0)
    public float BoxWidth;
    public float BoxHeight;
    public float BoxOrientation; // Rotation in radians

    /// <summary>True if this is a sphere trigger (radius > 0), false if box.</summary>
    public bool IsSphere => Radius > 0f;
}

public static class AreaTriggerRenderMath
{
    public static Vector3 ToScenePosition(Vector3 wowPosition, bool useRawWorldCoordinates)
    {
        return useRawWorldCoordinates
            ? wowPosition
            : new Vector3(
                WoWConstants.MapOrigin - wowPosition.Y,
                WoWConstants.MapOrigin - wowPosition.X,
                wowPosition.Z);
    }

    public static Vector3[] BuildBoxCorners(Vector3 center, float length, float width, float height, float yawRadians)
    {
        float halfLength = length * 0.5f;
        float halfWidth = width * 0.5f;
        float halfHeight = height * 0.5f;

        var localCorners = new[]
        {
            new Vector3(-halfLength, -halfWidth, -halfHeight),
            new Vector3( halfLength, -halfWidth, -halfHeight),
            new Vector3( halfLength,  halfWidth, -halfHeight),
            new Vector3(-halfLength,  halfWidth, -halfHeight),
            new Vector3(-halfLength, -halfWidth,  halfHeight),
            new Vector3( halfLength, -halfWidth,  halfHeight),
            new Vector3( halfLength,  halfWidth,  halfHeight),
            new Vector3(-halfLength,  halfWidth,  halfHeight),
        };

        if (MathF.Abs(yawRadians) < 0.0001f)
        {
            for (int i = 0; i < localCorners.Length; i++)
                localCorners[i] += center;
            return localCorners;
        }

        float cosYaw = MathF.Cos(yawRadians);
        float sinYaw = MathF.Sin(yawRadians);
        var worldCorners = new Vector3[localCorners.Length];
        for (int i = 0; i < localCorners.Length; i++)
        {
            var local = localCorners[i];
            float rotatedX = (local.X * cosYaw) - (local.Y * sinYaw);
            float rotatedY = (local.X * sinYaw) + (local.Y * cosYaw);
            worldCorners[i] = center + new Vector3(rotatedX, rotatedY, local.Z);
        }

        return worldCorners;
    }
}

/// <summary>
/// Loads and manages AreaTrigger data from AreaTrigger.dbc via DBCD.
/// </summary>
public class AreaTriggerLoader
{
    private readonly List<AreaTriggerEntry> _triggers = new();

    public IReadOnlyList<AreaTriggerEntry> Triggers => _triggers;
    public int Count => _triggers.Count;

    /// <summary>
    /// Load AreaTrigger.dbc from the given DBC provider.
    /// </summary>
    public void Load(IDBCProvider dbcProvider, string dbdDir, string build, int mapId, bool useRawWorldCoordinates = false)
    {
        _triggers.Clear();

        var dbdProvider = new FilesystemDBDProvider(dbdDir);
        var dbcd = new DBCD.DBCD(dbcProvider, dbdProvider);

        DBCD.IDBCDStorage storage;
        try
        {
            storage = dbcd.Load("AreaTrigger", build);
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[AreaTrigger] Failed to load AreaTrigger.dbc: {ex.Message}");
            return;
        }

        int loaded = 0;
        int errors = 0;
        foreach (var row in storage.Values)
        {
            try
            {
                int? id = TryGetInt(row, "ID");
                if (!id.HasValue)
                    continue;

                int? rowMapId = TryGetInt(row, "ContinentID", "MapID", "Map");
                
                // Only load triggers for this map
                if (rowMapId != mapId) continue;

                if (!TryGetPosition(row, out var wowPosition))
                {
                    if (errors++ < 3)
                        ViewerLog.Trace($"[AreaTrigger] Skipping row {id.Value}: missing position fields for build {build}");
                    continue;
                }

                var entry = new AreaTriggerEntry
                {
                    Id = id.Value,
                    MapId = rowMapId.Value,
                    WoWPosition = wowPosition,
                    Position = AreaTriggerRenderMath.ToScenePosition(wowPosition, useRawWorldCoordinates),
                    Radius = TryGetFloat(row, "Radius") ?? 0f,
                    BoxLength = TryGetFloat(row, "Box_length", "Box_Length", "BoxLength") ?? 0f,
                    BoxWidth = TryGetFloat(row, "Box_width", "Box_Width", "BoxWidth") ?? 0f,
                    BoxHeight = TryGetFloat(row, "Box_height", "Box_Height", "BoxHeight") ?? 0f,
                    BoxOrientation = TryGetFloat(row, "Box_yaw", "Box_Yaw", "BoxOrientation") ?? 0f
                };

                _triggers.Add(entry);
                loaded++;
            }
            catch (Exception ex)
            {
                if (errors++ < 3)
                    ViewerLog.Trace($"[AreaTrigger] Failed to parse row: {ex.Message}");
            }
        }

        if (errors > 0)
            ViewerLog.Info(ViewerLog.Category.Terrain, $"[AreaTrigger] {errors} rows failed to parse");
        ViewerLog.Info(ViewerLog.Category.Terrain, $"[AreaTrigger] Loaded {loaded} triggers for map {mapId}");
    }

    /// <summary>Try multiple field names, returning the first that resolves.</summary>
    private static int? TryGetInt(dynamic row, params string[] names)
    {
        foreach (var name in names)
        {
            try
            {
                var value = row[name];
                if (value == null)
                    continue;

                if (value is int intValue)
                    return intValue;
                if (value is uint uintValue)
                    return (int)uintValue;
                if (value is short shortValue)
                    return shortValue;
                if (value is ushort ushortValue)
                    return ushortValue;
                if (int.TryParse(value.ToString(), out int parsed))
                    return parsed;
            }
            catch (KeyNotFoundException) { }
        }
        return null;
    }

    /// <summary>Try multiple field names, returning the first that resolves.</summary>
    private static float? TryGetFloat(dynamic row, params string[] names)
    {
        foreach (var name in names)
        {
            try
            {
                var value = row[name];
                if (TryConvertToSingle(value, out float parsed))
                    return parsed;
            }
            catch (KeyNotFoundException) { }
        }
        return null;
    }

    private static bool TryGetPosition(dynamic row, out Vector3 position)
    {
        float wowX = 0f;
        float wowY = 0f;
        float wowZ = 0f;
        if (TryGetIndexedFloat(row, "Pos", 0, out wowX) &&
            TryGetIndexedFloat(row, "Pos", 1, out wowY) &&
            TryGetIndexedFloat(row, "Pos", 2, out wowZ))
        {
            position = new Vector3(wowX, wowY, wowZ);
            return true;
        }

        float? fallbackX = TryGetFloat(row, "Pos[0]", "Pos_X", "X", "Location[0]", "Location_X");
        float? fallbackY = TryGetFloat(row, "Pos[1]", "Pos_Y", "Y", "Location[1]", "Location_Y");
        float? fallbackZ = TryGetFloat(row, "Pos[2]", "Pos_Z", "Z", "Location[2]", "Location_Z");
        if (fallbackX.HasValue && fallbackY.HasValue && fallbackZ.HasValue)
        {
            position = new Vector3(fallbackX.Value, fallbackY.Value, fallbackZ.Value);
            return true;
        }

        position = default;
        return false;
    }

    private static bool TryGetIndexedFloat(dynamic row, string fieldName, int index, out float value)
    {
        try
        {
            var container = row[fieldName];
            if (container is Array array && index < array.Length)
            {
                if (TryConvertToSingle(array.GetValue(index), out value))
                    return true;
            }
        }
        catch (KeyNotFoundException) { }

        value = 0f;
        return false;
    }

    private static bool TryConvertToSingle(object? value, out float parsed)
    {
        if (value is null)
        {
            parsed = 0f;
            return false;
        }

        if (value is float floatValue)
        {
            parsed = floatValue;
            return true;
        }

        if (value is double doubleValue)
        {
            parsed = (float)doubleValue;
            return true;
        }

        if (value is decimal decimalValue)
        {
            parsed = (float)decimalValue;
            return true;
        }

        return float.TryParse(value.ToString(), out parsed);
    }

    /// <summary>
    /// Get all triggers for a specific map.
    /// </summary>
    public List<AreaTriggerEntry> GetTriggersForMap(int mapId)
    {
        return _triggers.Where(t => t.MapId == mapId).ToList();
    }
}
