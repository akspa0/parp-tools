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
        foreach (var row in storage.Values)
        {
            try
            {
                int id = Convert.ToInt32(row["ID"]);
                int rowMapId = Convert.ToInt32(row["ContinentID"] ?? row["MapID"] ?? row["Map"]);
                
                // Only load triggers for this map
                if (rowMapId != mapId) continue;

                float wowX = Convert.ToSingle(row["Pos_X"] ?? row["X"]);
                float wowY = Convert.ToSingle(row["Pos_Y"] ?? row["Y"]);
                float wowZ = Convert.ToSingle(row["Pos_Z"] ?? row["Z"]);

                // Transform WoW coords to renderer coords (same as terrain)
                const float MapOrigin = 17066.666f;
                const float ChunkSize = 533.33333f;
                float rendererX = MapOrigin - wowY;
                float rendererY = MapOrigin - wowX;
                float rendererZ = wowZ;

                var entry = new AreaTriggerEntry
                {
                    Id = id,
                    MapId = rowMapId,
                    WoWPosition = new Vector3(wowX, wowY, wowZ),
                    Position = new Vector3(rendererX, rendererY, rendererZ),
                    Radius = Convert.ToSingle(row["Radius"] ?? 0f),
                    BoxLength = Convert.ToSingle(row["Box_Length"] ?? row["BoxLength"] ?? 0f),
                    BoxWidth = Convert.ToSingle(row["Box_Width"] ?? row["BoxWidth"] ?? 0f),
                    BoxHeight = Convert.ToSingle(row["Box_Height"] ?? row["BoxHeight"] ?? 0f),
                    BoxOrientation = Convert.ToSingle(row["Box_Yaw"] ?? row["BoxOrientation"] ?? 0f)
                };

                _triggers.Add(entry);
                loaded++;
            }
            catch (Exception ex)
            {
                ViewerLog.Trace($"[AreaTrigger] Failed to parse row: {ex.Message}");
            }
        }

        ViewerLog.Info(ViewerLog.Category.Terrain, $"[AreaTrigger] Loaded {loaded} triggers for map {mapId}");
    }

    /// <summary>
    /// Get all triggers for a specific map.
    /// </summary>
    public List<AreaTriggerEntry> GetTriggersForMap(int mapId)
    {
        return _triggers.Where(t => t.MapId == mapId).ToList();
    }
}
