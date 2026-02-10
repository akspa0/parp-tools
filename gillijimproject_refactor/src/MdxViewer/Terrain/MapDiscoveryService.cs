using System.Numerics;
using DBCD;
using DBCD.Providers;
using MdxViewer.DataSources;

namespace MdxViewer.Terrain;

public record MapDefinition(int Id, string Directory, string Name, bool HasWdt);

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
            Console.WriteLine($"[MapDiscovery] Failed to load Map.dbc: {ex.Message}");
            return maps;
        }

        foreach (var key in storage.Keys)
        {
            var row = storage[key];
            int id = key;
            string? dir = TryGetString(row, "Directory");
            string? name = TryGetString(row, "MapName_lang") ?? dir ?? $"Map {id}";

            if (string.IsNullOrEmpty(dir)) continue;

            // Check if WDT exists in data source
            string wdtPath = $"World\\Maps\\{dir}\\{dir}.wdt";
            bool hasWdt = _dataSource.FileExists(wdtPath);

            maps.Add(new MapDefinition(id, dir, name, hasWdt));
        }

        return maps.OrderBy(m => m.Name).ToList();
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
}
