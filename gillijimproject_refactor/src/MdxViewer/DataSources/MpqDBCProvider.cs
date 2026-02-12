using System.IO;
using DBCD.Providers;
using WoWMapConverter.Core.Services;

namespace MdxViewer.DataSources;

/// <summary>
/// IDBCProvider that reads DBC files from MPQ archives via NativeMpqService.
/// DBC files in WoW Alpha live at "DBFilesClient\TableName.dbc" inside the MPQ.
/// </summary>
public class MpqDBCProvider : IDBCProvider
{
    private readonly NativeMpqService _mpq;
    private readonly Dictionary<string, byte[]> _cache = new(StringComparer.OrdinalIgnoreCase);

    public MpqDBCProvider(NativeMpqService mpq)
    {
        _mpq = mpq;
    }

    public Stream StreamForTableName(string tableName, string build)
    {
        if (_cache.TryGetValue(tableName, out var cached))
            return new MemoryStream(cached);

        // Try standard DBC path inside MPQ
        string[] paths =
        {
            $"DBFilesClient\\{tableName}.dbc",
            $"DBFilesClient\\{tableName}.db2",
            $"DBFilesClient/{tableName}.dbc",
            $"DBFilesClient/{tableName}.db2",
        };

        foreach (var path in paths)
        {
            var data = _mpq.ReadFile(path);
            if (data != null && data.Length > 0)
            {
                _cache[tableName] = data;
                return new MemoryStream(data);
            }
        }

        throw new FileNotFoundException($"DBC/DB2 not found in MPQ: {tableName}");
    }
}
