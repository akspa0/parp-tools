using WowViewer.Core.IO.Files;

namespace WowViewer.Core.IO.Dbc;

public sealed class MapDirectoryLookup
{
    private readonly Dictionary<string, string> _mapDirectoryLookup = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<int, string> _mapIdToDirectory = [];

    public bool IsLoaded => _mapDirectoryLookup.Count > 0;

    public void Load(IEnumerable<string> searchPaths, IArchiveReader? archiveReader = null)
    {
        ArgumentNullException.ThrowIfNull(searchPaths);

        if (IsLoaded)
            return;

        byte[]? data = TryReadFromDisk(searchPaths) ?? TryReadFromArchive(archiveReader);
        if (data is null)
        {
            Console.WriteLine("[WARN] Map.dbc not found. Minimap export may fail if casing is incorrect.");
            return;
        }

        LoadFromBytes(data);
    }

    public string? ResolveDirectory(string mapNameOrId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(mapNameOrId);

        if (int.TryParse(mapNameOrId, out int id) && _mapIdToDirectory.TryGetValue(id, out string? directory))
            return directory;

        return _mapDirectoryLookup.TryGetValue(mapNameOrId, out string? match) ? match : null;
    }

    private static byte[]? TryReadFromDisk(IEnumerable<string> searchPaths)
    {
        foreach (string basePath in searchPaths.Where(static path => !string.IsNullOrWhiteSpace(path)))
        {
            foreach (string candidate in EnumerateDiskCandidates(basePath, "Map"))
            {
                if (File.Exists(candidate))
                    return File.ReadAllBytes(candidate);
            }
        }

        return null;
    }

    private static byte[]? TryReadFromArchive(IArchiveReader? archiveReader)
    {
        return archiveReader is null ? null : DbClientFileReader.TryReadTable(archiveReader, "Map");
    }

    private static IEnumerable<string> EnumerateDiskCandidates(string basePath, string tableName)
    {
        yield return Path.Combine(basePath, "DBFilesClient", $"{tableName}.dbc");
        yield return Path.Combine(basePath, "DBFilesClient", $"{tableName}.db2");
        yield return Path.Combine(basePath, "DBC", $"{tableName}.dbc");
        yield return Path.Combine(basePath, "DBC", $"{tableName}.db2");
        yield return Path.Combine(basePath, $"{tableName}.dbc");
        yield return Path.Combine(basePath, $"{tableName}.db2");
    }

    private void LoadFromBytes(byte[] data)
    {
        DbcReader dbc = DbcReader.Load(data);

        for (int rowIndex = 0; rowIndex < dbc.Rows.Count; rowIndex++)
        {
            try
            {
                int id = dbc.GetInt(rowIndex, 0);
                string directory = dbc.GetString(rowIndex, 1);
                if (string.IsNullOrEmpty(directory))
                    continue;

                _mapIdToDirectory[id] = directory;
                _mapDirectoryLookup.TryAdd(directory, directory);

                if (dbc.Header.FieldCount > 4)
                {
                    string mapName = dbc.GetString(rowIndex, 4);
                    if (!string.IsNullOrEmpty(mapName))
                        _mapDirectoryLookup.TryAdd(mapName, directory);
                }
            }
            catch
            {
            }
        }
    }
}