using System.Numerics;
using DBCD;
using DBCD.Providers;
using MdxViewer.Rendering;

namespace MdxViewer.Terrain;

/// <summary>
/// Loads AreaPOI entries from DBC and provides them for rendering in the world scene and minimap.
/// AreaPOI positions are in WoW world coordinates (X=north, Y=west, Z=height).
/// </summary>
public class AreaPoiLoader
{
    public record AreaPoiEntry(
        int Id,
        string Name,
        Vector3 Position,      // Renderer coords (swapped from WoW)
        Vector3 WoWPosition,   // Original WoW coords for diagnostics
        int Icon,
        int Importance,
        int Flags,
        int ContinentId);

    public List<AreaPoiEntry> Entries { get; } = new();

    /// <summary>
    /// Load AreaPOI entries for a specific map from DBC via DBCD.
    /// </summary>
    /// <param name="dbcProvider">DBC provider (MPQ or filesystem)</param>
    /// <param name="dbdDir">Path to WoWDBDefs/definitions</param>
    /// <param name="build">Build string (e.g. "0.5.3.3368")</param>
    /// <param name="mapName">Map directory name (e.g. "Kalidar") to filter by</param>
    public void Load(IDBCProvider dbcProvider, string dbdDir, string build, string mapName)
    {
        var dbdProvider = new FilesystemDBDProvider(dbdDir);
        var dbcd = new DBCD.DBCD(dbcProvider, dbdProvider);

        // First, find the Map ID for this map name
        int mapId = FindMapId(dbcd, build, mapName);
        if (mapId < 0)
        {
            Console.WriteLine($"[AreaPOI] Map '{mapName}' not found in Map.dbc");
            return;
        }
        Console.WriteLine($"[AreaPOI] Map '{mapName}' = ID {mapId}");

        // Load AreaPOI entries
        IDBCDStorage storage;
        try
        {
            try { storage = dbcd.Load("AreaPOI", build, Locale.EnUS); }
            catch { storage = dbcd.Load("AreaPOI", build, Locale.None); }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[AreaPOI] Failed to load AreaPOI.dbc: {ex.Message}");
            return;
        }

        int total = 0, matched = 0;
        foreach (var key in storage.Keys)
        {
            total++;
            var row = storage[key];

            int continentId = TryGetInt(row, "ContinentID") ?? 0;
            if (continentId != mapId) continue;

            string name = TryGetString(row, "Name_lang") ?? $"POI #{key}";
            int icon = TryGetInt(row, "Icon") ?? 0;
            int importance = TryGetInt(row, "Importance") ?? 0;
            int flags = TryGetInt(row, "Flags") ?? 0;

            // Pos is a float[3] array: WoW X (north), WoW Y (west), WoW Z (height)
            float wowX = 0, wowY = 0, wowZ = 0;
            try
            {
                var posArr = row["Pos"];
                if (posArr is float[] fa && fa.Length >= 3)
                {
                    wowX = fa[0]; wowY = fa[1]; wowZ = fa[2];
                }
                else if (posArr is object[] oa && oa.Length >= 3)
                {
                    wowX = Convert.ToSingle(oa[0]);
                    wowY = Convert.ToSingle(oa[1]);
                    wowZ = Convert.ToSingle(oa[2]);
                }
            }
            catch { continue; }

            // Convert to renderer coords: rendererX = MapOrigin - wowY, rendererY = MapOrigin - wowX
            var rendererPos = new Vector3(
                WoWConstants.MapOrigin - wowY,
                WoWConstants.MapOrigin - wowX,
                wowZ);

            Entries.Add(new AreaPoiEntry(
                key, name, rendererPos, new Vector3(wowX, wowY, wowZ),
                icon, importance, flags, continentId));
            matched++;
        }

        Console.WriteLine($"[AreaPOI] Loaded {matched}/{total} entries for map '{mapName}' (ID={mapId})");

        // Diagnostic: print first few
        foreach (var e in Entries.Take(5))
            Console.WriteLine($"[AreaPOI]   [{e.Id}] \"{e.Name}\" pos=({e.Position.X:F0},{e.Position.Y:F0},{e.Position.Z:F0}) icon={e.Icon} imp={e.Importance}");
    }

    private static int FindMapId(DBCD.DBCD dbcd, string build, string mapName)
    {
        IDBCDStorage storage;
        try
        {
            try { storage = dbcd.Load("Map", build, Locale.EnUS); }
            catch { storage = dbcd.Load("Map", build, Locale.None); }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[AreaPOI] Failed to load Map.dbc: {ex.Message}");
            return -1;
        }

        foreach (var key in storage.Keys)
        {
            var row = storage[key];
            string? dir = TryGetString(row, "Directory");
            if (dir != null && dir.Equals(mapName, StringComparison.OrdinalIgnoreCase))
                return key;
        }
        return -1;
    }

    private static string? TryGetString(dynamic row, string fieldName)
    {
        try
        {
            var val = row[fieldName];
            if (val is string s) return s;
            return val?.ToString();
        }
        catch { return null; }
    }

    private static int? TryGetInt(dynamic row, string fieldName)
    {
        try
        {
            var val = row[fieldName];
            if (val is int i) return i;
            if (val is uint u) return (int)u;
            if (val is short s) return s;
            if (val is ushort us) return us;
            if (int.TryParse(val?.ToString(), out int parsed)) return parsed;
            return null;
        }
        catch { return null; }
    }
}
