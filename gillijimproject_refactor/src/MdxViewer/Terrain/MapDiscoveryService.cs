using System.Numerics;
using DBCD;
using DBCD.Providers;
using MdxViewer.DataSources;
using MdxViewer.Logging;

namespace MdxViewer.Terrain;

public record MapDefinition(int Id, string Directory, string Name, bool HasWdt, bool HasWdl);

public class MapDiscoveryService
{
    private readonly IDBCProvider _dbcProvider;
    private readonly string _dbdDir;
    private readonly string _build;
    private readonly IDataSource _dataSource;

    public MapDiscoveryService(IDBCProvider dbcProvider, string dbdDir, string build, IDataSource dataSource)
    {
        _dbcProvider = dbcProvider;
        _dbdDir = dbdDir;
        _build = build;
        _dataSource = dataSource;
    }

    public List<MapDefinition> DiscoverMaps()
    {
        var maps = new List<MapDefinition>();
        var dbdProvider = new FilesystemDBDProvider(_dbdDir);
        var dbcd = new DBCD.DBCD(_dbcProvider, dbdProvider);

        IDBCDStorage storage;
        try
        {
            try { storage = dbcd.Load("Map", _build, Locale.EnUS); }
            catch { storage = dbcd.Load("Map", _build, Locale.None); }
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[MapDiscovery] Failed to load Map.dbc: {ex.Message}");
            return maps;
        }

        // Detect the actual MapID column (varies by build: "ID", "MapID", etc.)
        string idCol = DetectIdColumn(storage);

        // Log available columns once for diagnostics
        try
        {
            var cols = storage.AvailableColumns ?? Array.Empty<string>();
            ViewerLog.Trace($"[MapDiscovery] Map.dbc columns ({cols.Length}): {string.Join(", ", cols)}");
        }
        catch { }

        // Detect the name column — varies across builds
        string nameCol = DetectNameColumn(storage);
        ViewerLog.Trace($"[MapDiscovery] Using idCol='{idCol}', nameCol='{nameCol}'");

        foreach (var key in storage.Keys)
        {
            var row = storage[key];
            // Use the actual MapID field, falling back to DBCD key
            int id = !string.IsNullOrEmpty(idCol) ? TryGetInt(row, idCol) ?? key : key;
            string? dir = Sanitize(TryGetString(row, "Directory"));
            string? name = Sanitize(!string.IsNullOrEmpty(nameCol) ? TryGetString(row, nameCol) : null);
            if (string.IsNullOrEmpty(name)) name = dir ?? $"Map {id}";

            if (string.IsNullOrEmpty(dir)) continue;

            // Check if WDT exists in data source
            string wdtPath = $"World\\Maps\\{dir}\\{dir}.wdt";
            bool hasWdt = _dataSource.FileExists(wdtPath);

            // Check if WDL exists (Alpha 0.5.3 stores as .wdl.mpq)
            string wdlPath = $"World\\Maps\\{dir}\\{dir}.wdl";
            bool hasWdl = _dataSource.FileExists(wdlPath);

            maps.Add(new MapDefinition(id, dir, name, hasWdt, hasWdl));
        }

        return maps.OrderBy(m => m.Name).ToList();
    }

    private static string DetectIdColumn(IDBCDStorage storage)
    {
        try
        {
            var cols = storage.AvailableColumns ?? Array.Empty<string>();
            string[] prefers = new[] { "ID", "Id", "MapID", "MapId", "m_ID" };
            foreach (var p in prefers)
            {
                var match = cols.FirstOrDefault(x => string.Equals(x, p, StringComparison.OrdinalIgnoreCase));
                if (!string.IsNullOrEmpty(match)) return match;
            }
            return "";
        }
        catch { return ""; }
    }

    private static string DetectNameColumn(IDBCDStorage storage)
    {
        try
        {
            var cols = storage.AvailableColumns ?? Array.Empty<string>();
            // Try all known name column variants across WoW builds
            string[] prefers = new[] {
                "MapName_lang", "MapName", "MapName_Lang",
                "m_MapName_lang", "m_MapName",
                "Name_lang", "Name", "name_lang", "name",
                "MapDescription0_lang", "MapDescription0"
            };
            foreach (var p in prefers)
            {
                var match = cols.FirstOrDefault(x => string.Equals(x, p, StringComparison.OrdinalIgnoreCase));
                if (!string.IsNullOrEmpty(match)) return match;
            }
            // Last resort: find any column containing "name" (case-insensitive)
            var fuzzy = cols.FirstOrDefault(x => x.IndexOf("name", StringComparison.OrdinalIgnoreCase) >= 0);
            if (!string.IsNullOrEmpty(fuzzy))
            {
                ViewerLog.Trace($"[MapDiscovery] Fuzzy name column match: '{fuzzy}'");
                return fuzzy;
            }
            return "";
        }
        catch { return ""; }
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

    private static string? Sanitize(string? s)
    {
        if (string.IsNullOrEmpty(s)) return s;
        int nullIdx = s.IndexOf('\0');
        if (nullIdx >= 0) s = s[..nullIdx];
        return new string(s.Where(c => !char.IsControl(c) || c == '\n').ToArray());
    }
}
