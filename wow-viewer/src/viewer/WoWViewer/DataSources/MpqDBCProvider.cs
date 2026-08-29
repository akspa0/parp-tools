using System.IO;
using DBCD.Providers;
using WowViewer.Core.IO.Files;

namespace WoWViewer.DataSources;

/// <summary>
/// IDBCProvider that reads DBC files through the data source (loose files first, then MPQ archive).
/// DBC files in WoW live at "DBFilesClient\TableName.dbc" or "DBC\TableName.dbc".
/// </summary>
public class MpqDBCProvider : IDBCProvider
{
    private readonly IArchiveReader? _archiveReader;
    private readonly IDataSource? _dataSource;
    private readonly Dictionary<string, byte[]> _cache = new(StringComparer.OrdinalIgnoreCase);

    public MpqDBCProvider(IArchiveReader archiveReader)
    {
        _archiveReader = archiveReader;
    }

    public MpqDBCProvider(IArchiveReader? archiveReader, IDataSource? dataSource)
    {
        _archiveReader = archiveReader;
        _dataSource = dataSource;
    }

    public MpqDBCProvider(IDataSource dataSource)
    {
        _dataSource = dataSource;
    }

    public Stream StreamForTableName(string tableName, string build)
    {
        if (_cache.TryGetValue(tableName, out var cached))
            return new MemoryStream(cached);

        // 1. Check IDataSource first (prioritizes loose files on disk / attached overlays)
        if (_dataSource != null)
        {
            foreach (string path in DbClientFileReader.EnumerateTablePaths(tableName))
            {
                byte[]? data = _dataSource.ReadFile(path);
                if (data is { Length: > 0 })
                {
                    _cache[tableName] = data;
                    return new MemoryStream(data);
                }
            }
        }

        // 2. Fall back to archive reader (MPQ archives)
        if (_archiveReader != null)
        {
            byte[]? data = DbClientFileReader.TryReadTable(_archiveReader, tableName);
            if (data is { Length: > 0 })
            {
                _cache[tableName] = data;
                return new MemoryStream(data);
            }
        }

        throw new FileNotFoundException($"DBC/DB2 not found in MPQ or loose files: {tableName}");
    }
}

