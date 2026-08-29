using DBCD.Providers;
using WowViewer.Core.IO.Files;

namespace WowViewer.Core.IO.Dbc;

/// <summary>
/// Shared DBCD provider over the repository's archive-reader boundary and optional loose file overrides.
/// Viewer and offline tools therefore consume the same exact DBC bytes and never grow a second table parser.
/// </summary>
public sealed class ArchiveReaderDbcProvider(IArchiveReader archiveReader, Func<string, byte[]?>? looseFileReader = null) : IDBCProvider
{
    private readonly Dictionary<string, byte[]> _cache = new(StringComparer.OrdinalIgnoreCase);

    public Stream StreamForTableName(string tableName, string build)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tableName);
        if (!_cache.TryGetValue(tableName, out byte[]? bytes))
        {
            if (looseFileReader != null)
            {
                foreach (string path in DbClientFileReader.EnumerateTablePaths(tableName))
                {
                    bytes = looseFileReader(path);
                    if (bytes is { Length: > 0 })
                        break;
                }
            }

            if (bytes is not { Length: > 0 })
            {
                bytes = DbClientFileReader.TryReadTable(archiveReader, tableName);
            }

            if (bytes is not { Length: > 0 })
                throw new FileNotFoundException($"DBC/DB2 not found in archive or loose files: {tableName} (build {build})");
            _cache.Add(tableName, bytes);
        }

        return new MemoryStream(bytes, writable: false);
    }
}

