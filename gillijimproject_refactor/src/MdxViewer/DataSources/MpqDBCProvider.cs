using System.IO;
using DBCD.Providers;
using WowViewer.Core.IO.Files;

namespace MdxViewer.DataSources;

/// <summary>
/// IDBCProvider that reads DBC files through the shared archive-reader boundary.
/// DBC files in WoW Alpha live at "DBFilesClient\TableName.dbc" inside the MPQ.
/// </summary>
public class MpqDBCProvider : IDBCProvider
{
    private readonly IArchiveReader _archiveReader;
    private readonly Dictionary<string, byte[]> _cache = new(StringComparer.OrdinalIgnoreCase);

    public MpqDBCProvider(IArchiveReader archiveReader)
    {
        _archiveReader = archiveReader;
    }

    public Stream StreamForTableName(string tableName, string build)
    {
        if (_cache.TryGetValue(tableName, out var cached))
            return new MemoryStream(cached);

        byte[]? data = DbClientFileReader.TryReadTable(_archiveReader, tableName);
        if (data is { Length: > 0 })
        {
            _cache[tableName] = data;
            return new MemoryStream(data);
        }

        throw new FileNotFoundException($"DBC/DB2 not found in MPQ: {tableName}");
    }
}
