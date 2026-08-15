using System.Globalization;
using DBCD;
using DBCD.Providers;
using WowViewer.Core.Audio;

namespace WowViewer.Core.IO.Dbc;

/// <summary>
/// Reads the active client's SoundWaterType table without embedding any
/// SoundEntries IDs in the renderer.
/// </summary>
public sealed class AlphaSoundWaterTypeCatalogReader
{
    public SoundWaterTypeCatalog Load(
        IDBCProvider dbcProvider,
        string definitionsDirectory,
        string buildVersion)
    {
        ArgumentNullException.ThrowIfNull(dbcProvider);
        if (string.IsNullOrWhiteSpace(definitionsDirectory))
            throw new ArgumentException("A WoWDBDefs definitions directory is required.", nameof(definitionsDirectory));
        if (string.IsNullOrWhiteSpace(buildVersion))
            throw new ArgumentException("An exact client build is required.", nameof(buildVersion));

        DBCD.DBCD dbcd = new(dbcProvider, new FilesystemDBDProvider(definitionsDirectory));
        IDBCDStorage storage;
        try
        {
            storage = dbcd.Load("SoundWaterType", buildVersion, Locale.EnUS);
        }
        catch
        {
            storage = dbcd.Load("SoundWaterType", buildVersion, Locale.None);
        }

        List<SoundWaterTypeEntry> entries = [];
        foreach (DBCDRow row in storage.Values)
        {
            int id = GetInt(row, "ID") ?? 0;
            int soundType = GetInt(row, "SoundType") ?? 0;
            int soundSubtype = GetInt(row, "SoundSubtype") ?? 0;
            int soundId = GetInt(row, "SoundID") ?? 0;
            if (id <= 0 || soundId <= 0)
                continue;

            entries.Add(new SoundWaterTypeEntry(id, soundType, soundSubtype, soundId));
        }

        return new SoundWaterTypeCatalog(entries);
    }

    private static int? GetInt(DBCDRow row, string field)
    {
        try { return Convert.ToInt32(row[field], CultureInfo.InvariantCulture); }
        catch { return null; }
    }
}
