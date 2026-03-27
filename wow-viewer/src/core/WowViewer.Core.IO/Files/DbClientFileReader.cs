namespace WowViewer.Core.IO.Files;

public static class DbClientFileReader
{
    public static IReadOnlyList<string> EnumerateTablePaths(string tableName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tableName);

        return
        [
            $"DBFilesClient\\{tableName}.dbc",
            $"DBFilesClient\\{tableName}.db2",
            $"DBFilesClient/{tableName}.dbc",
            $"DBFilesClient/{tableName}.db2",
            $"DBC\\{tableName}.dbc",
            $"DBC\\{tableName}.db2",
            $"DBC/{tableName}.dbc",
            $"DBC/{tableName}.db2",
            $"{tableName}.dbc",
            $"{tableName}.db2",
        ];
    }

    public static byte[]? TryReadTable(IArchiveReader archiveReader, string tableName)
    {
        ArgumentNullException.ThrowIfNull(archiveReader);

        foreach (string path in EnumerateTablePaths(tableName))
        {
            byte[]? data = archiveReader.ReadFile(path);
            if (data is { Length: > 0 })
                return data;
        }

        return null;
    }
}