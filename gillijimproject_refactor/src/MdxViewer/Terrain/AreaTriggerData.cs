using System.Numerics;
using DBCD.Providers;
using MdxViewer.Logging;

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
    public void Load(IDBCProvider dbcProvider, string dbdDir, string build, int mapId)
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
                int id = Convert.ToInt32(row["ID"]);
                int rowMapId = GetInt(row, "ContinentID", "MapID", "Map");
                
                // Only load triggers for this map
                if (rowMapId != mapId) continue;

                // DBCD array fields: Pos[3] in the DBD → accessed as "Pos[0]", "Pos[1]", "Pos[2]"
                float wowX = GetFloat(row, "Pos[0]", "Pos_X", "X");
                float wowY = GetFloat(row, "Pos[1]", "Pos_Y", "Y");
                float wowZ = GetFloat(row, "Pos[2]", "Pos_Z", "Z");

                // Transform WoW coords to renderer coords (same as terrain)
                const float MapOrigin = 17066.666f;
                float rendererX = MapOrigin - wowY;
                float rendererY = MapOrigin - wowX;
                float rendererZ = wowZ;

                var entry = new AreaTriggerEntry
                {
                    Id = id,
                    MapId = rowMapId,
                    WoWPosition = new Vector3(wowX, wowY, wowZ),
                    Position = new Vector3(rendererX, rendererY, rendererZ),
                    Radius = GetFloat(row, "Radius"),
                    BoxLength = GetFloat(row, "Box_Length", "BoxLength"),
                    BoxWidth = GetFloat(row, "Box_Width", "BoxWidth"),
                    BoxHeight = GetFloat(row, "Box_Height", "BoxHeight"),
                    BoxOrientation = GetFloat(row, "Box_Yaw", "BoxOrientation")
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
    private static int GetInt(dynamic row, params string[] names)
    {
        foreach (var name in names)
        {
            try { return Convert.ToInt32(row[name]); }
            catch (KeyNotFoundException) { }
        }
        return 0;
    }

    /// <summary>Try multiple field names, returning the first that resolves.</summary>
    private static float GetFloat(dynamic row, params string[] names)
    {
        foreach (var name in names)
        {
            try { return Convert.ToSingle(row[name]); }
            catch (KeyNotFoundException) { }
        }
        return 0f;
    }

    /// <summary>
    /// Get all triggers for a specific map.
    /// </summary>
    public List<AreaTriggerEntry> GetTriggersForMap(int mapId)
    {
        return _triggers.Where(t => t.MapId == mapId).ToList();
    }
}
