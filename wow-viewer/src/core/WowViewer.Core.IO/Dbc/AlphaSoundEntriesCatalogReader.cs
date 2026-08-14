using System.Collections;
using System.Globalization;
using DBCD;
using DBCD.Providers;
using WowViewer.Core.Audio;

namespace WowViewer.Core.IO.Dbc;

public sealed class AlphaSoundEntriesCatalogReader
{
    public AlphaSoundEntriesCatalog Load(IDBCProvider dbcProvider, string definitionsDirectory, string buildVersion)
    {
        ArgumentNullException.ThrowIfNull(dbcProvider);
        if (string.IsNullOrWhiteSpace(definitionsDirectory))
            throw new ArgumentException("A WoWDBDefs definitions directory is required.", nameof(definitionsDirectory));
        if (string.IsNullOrWhiteSpace(buildVersion))
            throw new ArgumentException("An exact client build is required.", nameof(buildVersion));

        IDBCDStorage storage = LoadTable(dbcProvider, definitionsDirectory, buildVersion);
        Dictionary<int, AlphaSoundEntry> entries = [];
        foreach (DBCDRow row in storage.Values)
        {
            int id = GetInt(row, "ID") ?? 0;
            if (id <= 0)
                continue;

            entries[id] = new AlphaSoundEntry(
                id,
                GetStrings(row, "File"),
                GetString(row, "DirectoryBase") ?? string.Empty,
                GetFloat(row, "VolumeFloat") ?? 1f,
                GetFloat(row, "MinDistance") ?? 0f,
                GetFloat(row, "MaxDistance") ?? 0f,
                GetFloat(row, "DistanceCutoff") ?? 0f);
        }

        return new AlphaSoundEntriesCatalog(entries);
    }

    private static IDBCDStorage LoadTable(IDBCProvider dbcProvider, string definitionsDirectory, string buildVersion)
    {
        DBCD.DBCD dbcd = new(dbcProvider, new FilesystemDBDProvider(definitionsDirectory));
        try
        {
            return dbcd.Load("SoundEntries", buildVersion, Locale.EnUS);
        }
        catch
        {
            return dbcd.Load("SoundEntries", buildVersion, Locale.None);
        }
    }

    private static int? GetInt(DBCDRow row, string field)
    {
        try { return Convert.ToInt32(row[field], CultureInfo.InvariantCulture); }
        catch { return null; }
    }

    private static float? GetFloat(DBCDRow row, string field)
    {
        try { return Convert.ToSingle(row[field], CultureInfo.InvariantCulture); }
        catch { return null; }
    }

    private static string? GetString(DBCDRow row, string field)
    {
        try
        {
            object value = row[field];
            return value as string ?? value.ToString();
        }
        catch { return null; }
    }

    private static string[] GetStrings(DBCDRow row, string field)
    {
        try
        {
            object value = row[field];
            if (value is string text)
                return string.IsNullOrWhiteSpace(text) ? [] : [text];

            if (value is IEnumerable enumerable)
            {
                List<string> result = [];
                foreach (object? item in enumerable)
                {
                    string? textItem = item?.ToString();
                    if (!string.IsNullOrWhiteSpace(textItem))
                        result.Add(textItem);
                }

                return result.ToArray();
            }
        }
        catch { }

        return [];
    }
}
