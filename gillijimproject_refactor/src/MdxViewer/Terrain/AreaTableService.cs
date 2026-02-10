using DBCD;
using DBCD.Providers;

namespace MdxViewer.Terrain;

/// <summary>
/// Loads AreaTable.dbc via DBCD and provides AreaID → area name lookups.
/// Used to display the current area name in the status bar based on the camera's chunk position.
/// </summary>
public class AreaTableService
{
    private readonly Dictionary<int, AreaEntry> _areas = new();

    public record AreaEntry(int Id, string Name, int ParentAreaId, int MapId, int Flags);

    public int Count => _areas.Count;

    /// <summary>
    /// Load AreaTable.dbc from the given DBC provider.
    /// </summary>
    public void Load(IDBCProvider dbcProvider, string dbdDir, string build)
    {
        var dbdProvider = new FilesystemDBDProvider(dbdDir);
        var dbcd = new DBCD.DBCD(dbcProvider, dbdProvider);

        IDBCDStorage storage;
        try
        {
            try { storage = dbcd.Load("AreaTable", build, Locale.EnUS); }
            catch { storage = dbcd.Load("AreaTable", build, Locale.None); }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[AreaTable] Failed to load AreaTable.dbc: {ex.Message}");
            return;
        }

        // Detect column names (varies by build)
        string nameCol = DetectColumn(storage, "AreaName_lang", "AreaName", "Name");
        string idCol = DetectColumn(storage, "AreaNumber", "ID", "AreaID");
        string parentCol = DetectColumn(storage, "ParentAreaNum", "ParentAreaID", "ParentAreaNum");
        string mapCol = DetectColumn(storage, "ContinentID", "MapID", "Continent");
        string flagsCol = DetectColumn(storage, "Flags", "AreaFlags");

        // MCNK AreaId uses the DBC row key (implicit ID), NOT the AreaNumber field.
        // AreaNumber is a packed value (e.g. 1048576) unrelated to MCNK placement.
        // Index primarily by row key. Also index by AreaNumber as fallback.
        foreach (var key in storage.Keys)
        {
            var row = storage[key];
            int areaNumber = SafeField<int>(row, idCol, key);
            string name = Sanitize(SafeField<string>(row, nameCol, "") ?? "");
            int parentId = SafeField<int>(row, parentCol, 0);
            int mapId = SafeField<int>(row, mapCol, 0);
            int flags = SafeField<int>(row, flagsCol, 0);

            var entry = new AreaEntry(key, name, parentId, mapId, flags);
            _areas[key] = entry;           // Primary: DBC row key (matches MCNK AreaId)
            _areas.TryAdd(areaNumber, entry); // Fallback: AreaNumber field
        }

        Console.WriteLine($"[AreaTable] Loaded {_areas.Count} area entries (idCol={idCol}, nameCol={nameCol}, parentCol={parentCol}, mapCol={mapCol})");
        // Dump first 10 entries for diagnostics
        int dumped = 0;
        foreach (var kv in _areas)
        {
            if (dumped++ >= 10) break;
            var e = kv.Value;
            Console.WriteLine($"  [AreaTable] key={kv.Key} id={e.Id} name=\"{e.Name}\" parent={e.ParentAreaId} map={e.MapId}");
        }
    }

    /// <summary>
    /// Look up an area name by AreaID. Returns null if not found.
    /// </summary>
    public string? GetAreaName(int areaId)
    {
        return _areas.TryGetValue(areaId, out var entry) ? entry.Name : null;
    }

    /// <summary>
    /// Get full area entry by ID.
    /// </summary>
    public AreaEntry? GetArea(int areaId)
    {
        return _areas.TryGetValue(areaId, out var entry) ? entry : null;
    }

    /// <summary>
    /// Get area name with parent context (e.g. "Durotar > Razor Hill").
    /// </summary>
    public string GetAreaDisplayName(int areaId)
    {
        if (!_areas.TryGetValue(areaId, out var entry))
            return $"Unknown ({areaId})";

        if (entry.ParentAreaId != 0 && _areas.TryGetValue(entry.ParentAreaId, out var parent))
            return $"{parent.Name} > {entry.Name}";

        return entry.Name;
    }

    /// <summary>
    /// Get area name with parent context, but only if the area belongs to the given MapID.
    /// Returns null if the area belongs to a different map or AreaID is not found.
    /// MCNK AreaID maps directly to AreaTable.dbc ID — no byte packing.
    /// </summary>
    public string? GetAreaDisplayNameForMap(int areaId, int mapId)
    {
        if (!_areas.TryGetValue(areaId, out var entry) || entry.MapId != mapId)
            return null;

        if (entry.ParentAreaId != 0 && _areas.TryGetValue(entry.ParentAreaId, out var parent))
            return $"{parent.Name} > {entry.Name}";
        return entry.Name;
    }

    private static string DetectColumn(IDBCDStorage storage, params string[] candidates)
    {
        if (storage.Values.Count == 0) return candidates[0];
        var row = storage.Values.First();
        foreach (var col in candidates)
        {
            try { _ = row[col]; return col; }
            catch { }
        }
        return candidates[0];
    }

    private static T SafeField<T>(dynamic row, string col, T fallback)
    {
        try { return (T)row[col]; }
        catch { return fallback; }
    }

    private static string Sanitize(string s)
    {
        if (string.IsNullOrEmpty(s)) return s;
        int nullIdx = s.IndexOf('\0');
        if (nullIdx >= 0) s = s[..nullIdx];
        return new string(s.Where(c => !char.IsControl(c) || c == '\n').ToArray());
    }
}
