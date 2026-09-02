using DBCD;
using DBCD.Providers;

namespace WowViewer.Core.IO.Dbc;

/// <summary>
/// Shared DBCD loading helpers for the small lookup tables in this namespace.
/// </summary>
/// <remarks>
/// Client tables from Cataclysm onward are WDB2 and later, which <see cref="DbcReader"/> (WDBC only)
/// cannot parse, and whose field order is build-scoped. Everything here therefore goes through DBCD
/// and WoWDBDefs so a build change is a definition change rather than a new hardcoded byte offset.
/// Column names are <em>detected</em>, never assumed: a definition rename must surface as a named
/// failure, not as a table that silently resolves everything to zero.
/// </remarks>
internal static class DbcTableLoader
{
    public static IDBCDStorage Load(
        IDBCProvider dbcProvider,
        string definitionsDirectory,
        string buildVersion,
        string tableName)
    {
        ArgumentNullException.ThrowIfNull(dbcProvider);
        ArgumentException.ThrowIfNullOrWhiteSpace(definitionsDirectory);
        ArgumentException.ThrowIfNullOrWhiteSpace(buildVersion);
        ArgumentException.ThrowIfNullOrWhiteSpace(tableName);

        FilesystemDBDProvider dbdProvider = new(definitionsDirectory);
        DBCD.DBCD dbcd = new(dbcProvider, dbdProvider);

        try
        {
            return dbcd.Load(tableName, buildVersion, Locale.EnUS);
        }
        catch
        {
            return dbcd.Load(tableName, buildVersion, Locale.None);
        }
    }

    /// <summary>
    /// Return the first of <paramref name="candidates"/> that the storage actually exposes, or null.
    /// </summary>
    public static string? DetectColumn(IDBCDStorage storage, IReadOnlyList<string> candidates)
    {
        ArgumentNullException.ThrowIfNull(storage);
        ArgumentNullException.ThrowIfNull(candidates);

        HashSet<string> available = new(storage.AvailableColumns, StringComparer.OrdinalIgnoreCase);
        foreach (string candidate in candidates)
        {
            if (!available.Contains(candidate))
                continue;

            // Return the storage's own spelling so downstream indexing is exact.
            foreach (string column in storage.AvailableColumns)
            {
                if (string.Equals(column, candidate, StringComparison.OrdinalIgnoreCase))
                    return column;
            }
        }

        return null;
    }

    /// <summary>Column names that carry a table's own row id.</summary>
    public static readonly string[] IdColumns = ["ID", "Id"];

    /// <summary>
    /// The row's authoritative id.
    /// </summary>
    /// <remarks>
    /// MEASURED on MoP 5.0.1.15464: <c>DBCDRow.ID</c> is a <b>positional</b> key for these WDB2
    /// tables, not the row id. <c>LiquidObject</c> loads as keys 1..1244 while the row keyed 42
    /// carries <c>ID=316</c>, and <c>LiquidMaterial</c> loads as keys 1..7 while its real ids are
    /// {1,2,3,4,5,8,10}. Keying a lookup on <c>row.ID</c> therefore builds a table indexed by row
    /// order, which silently resolves the wrong row for every sparse id and reports every id above
    /// the row count as absent. Always prefer the ID column when the table exposes one.
    /// </remarks>
    public static int ResolveRowId(DBCDRow row, string? idColumn)
        => TryGetInt(row, idColumn) ?? row.ID;

    /// <summary>Read an integer field, returning null rather than throwing on a type mismatch.</summary>
    public static int? TryGetInt(DBCDRow row, string? column)
    {
        if (string.IsNullOrWhiteSpace(column))
            return null;

        try
        {
            return Convert.ToInt32(row[column]);
        }
        catch (Exception ex) when (ex is InvalidCastException or FormatException or OverflowException or KeyNotFoundException)
        {
            return null;
        }
    }
}
