using System.Numerics;
using DBCD;
using DBCD.Providers;
using MdxViewer.DataSources;
using MdxViewer.Logging;

namespace MdxViewer.Terrain;

public record MapDefinition(int Id, string Directory, string Name, bool HasWdt, bool HasWdl, bool HasDbcEntry = true);

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
        var mapsByDirectory = new Dictionary<string, MapDefinition>(StringComparer.OrdinalIgnoreCase);
        var dbdProvider = new FilesystemDBDProvider(_dbdDir);
        var dbcd = new DBCD.DBCD(_dbcProvider, dbdProvider);

        IDBCDStorage storage;
        Locale localeUsed;
        try
        {
            try
            {
                storage = dbcd.Load("Map", _build, Locale.EnUS);
                localeUsed = Locale.EnUS;
            }
            catch (Exception enUsEx)
            {
                ViewerLog.Important(ViewerLog.Category.General,
                    $"[MapDiscovery] Locale.EnUS load failed for build {_build}: {enUsEx.Message}. Retrying Locale.None.");
                storage = dbcd.Load("Map", _build, Locale.None);
                localeUsed = Locale.None;
            }
        }
        catch (Exception ex)
        {
            ViewerLog.Error(ViewerLog.Category.Dbc,
                $"[MapDiscovery] Failed to load Map.dbc for build {_build}: {ex.Message}");
            return DiscoverLooseMapsOnly(_dataSource);
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
        ViewerLog.Important(ViewerLog.Category.General,
            $"[MapDiscovery] Using build={_build} locale={localeUsed} idCol='{idCol}' nameCol='{nameCol}' rowCount={storage.Keys.Count}");

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

            mapsByDirectory[dir] = new MapDefinition(id, dir, name, hasWdt, hasWdl, HasDbcEntry: true);
        }

        MergeLooseMaps(mapsByDirectory, _dataSource);

        var maps = mapsByDirectory.Values.OrderBy(m => m.Name).ToList();
        ViewerLog.Important(ViewerLog.Category.General,
            $"[MapDiscovery] Produced {maps.Count} map definitions ({maps.Count(m => m.HasWdt)} with WDTs, {maps.Count(m => m.HasWdl)} with WDLs, {maps.Count(m => !m.HasDbcEntry)} custom loose maps)");

        return maps;
    }

    public static List<MapDefinition> DiscoverLooseMapsOnly(IDataSource dataSource)
    {
        var mapsByDirectory = new Dictionary<string, MapDefinition>(StringComparer.OrdinalIgnoreCase);
        MergeLooseMaps(mapsByDirectory, dataSource);

        var maps = mapsByDirectory.Values.OrderBy(m => m.Name).ToList();
        ViewerLog.Important(ViewerLog.Category.General,
            $"[MapDiscovery] Produced {maps.Count} loose map definitions without Map.dbc metadata.");
        return maps;
    }

    private static void MergeLooseMaps(IDictionary<string, MapDefinition> mapsByDirectory, IDataSource dataSource)
    {
        int nextSyntheticId = mapsByDirectory.Values
            .Where(map => !map.HasDbcEntry)
            .Select(map => map.Id)
            .DefaultIfEmpty(0)
            .Min() - 1;
        if (nextSyntheticId >= 0)
            nextSyntheticId = -1;

        foreach (MapDefinition looseMap in EnumerateLooseMaps(dataSource, nextSyntheticId))
        {
            if (mapsByDirectory.TryGetValue(looseMap.Directory, out var existing))
            {
                mapsByDirectory[looseMap.Directory] = existing with
                {
                    HasWdt = existing.HasWdt || looseMap.HasWdt,
                    HasWdl = existing.HasWdl || looseMap.HasWdl
                };
                continue;
            }

            mapsByDirectory[looseMap.Directory] = looseMap;
            nextSyntheticId = looseMap.Id - 1;
        }
    }

    private static IEnumerable<MapDefinition> EnumerateLooseMaps(IDataSource dataSource, int startingSyntheticId)
    {
        int nextSyntheticId = startingSyntheticId;
        foreach (string path in dataSource.GetFileList(".wdt"))
        {
            if (!TryExtractMapDirectoryFromWdtPath(path, out string? mapDirectory) || string.IsNullOrWhiteSpace(mapDirectory))
                continue;

            string discoveredDirectory = mapDirectory;
            string normalizedWdtPath = $"World\\Maps\\{discoveredDirectory}\\{discoveredDirectory}.wdt";
            bool hasWdt = dataSource.FileExists(normalizedWdtPath);
            bool hasWdl = dataSource.FileExists($"World\\Maps\\{discoveredDirectory}\\{discoveredDirectory}.wdl");
            yield return new MapDefinition(nextSyntheticId--, discoveredDirectory, discoveredDirectory, hasWdt, hasWdl, HasDbcEntry: false);
        }
    }

    private static bool TryExtractMapDirectoryFromWdtPath(string path, out string? mapDirectory)
    {
        string normalized = path.Replace('/', '\\').TrimStart('\\');
        string[] segments = normalized.Split('\\', StringSplitOptions.RemoveEmptyEntries);
        if (segments.Length < 4)
        {
            mapDirectory = null;
            return false;
        }

        if (!segments[0].Equals("World", StringComparison.OrdinalIgnoreCase) ||
            !segments[1].Equals("Maps", StringComparison.OrdinalIgnoreCase))
        {
            mapDirectory = null;
            return false;
        }

        string directory = segments[2];
        string fileName = segments[^1];
        string expectedFileName = directory + ".wdt";
        if (!fileName.Equals(expectedFileName, StringComparison.OrdinalIgnoreCase))
        {
            mapDirectory = null;
            return false;
        }

        mapDirectory = directory;
        return true;
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
