using System.Globalization;
using DBCD;
using DBCD.Providers;

namespace WowViewer.Core.IO.Dbc;

/// <summary>
/// Lookup table for <c>LiquidObject.dbc</c>: a liquid-object id to the
/// <c>LiquidType.dbc</c> row it selects.
/// </summary>
/// <remarks>
/// <para>
/// In Cataclysm and later the second <c>uint16</c> of <c>SMLiquidInstance</c> is
/// <c>liquid_object_or_lvf</c>. Values at or above
/// <see cref="LiquidVertexFormatChain.LiquidObjectIdThreshold"/> are a row id in this table, not a
/// liquid vertex format, and the real vertex format has to be resolved through
/// <c>LiquidType</c> and <c>LiquidMaterial</c>. See
/// <c>specs/205-mh2o-liquid-object-vertex-format/</c>.
/// </para>
/// <para>
/// The table is read through DBCD and WoWDBDefs rather than a hardcoded byte offset, because the
/// MoP-era client files are WDB2 and the field order is build-scoped. Column names are detected
/// rather than assumed so a definition rename does not silently resolve to zero.
/// </para>
/// </remarks>
public sealed class DbcLiquidObjectTable
{
    private static readonly string[] LiquidTypeIdColumns = ["LiquidTypeID", "LiquidTypeId", "LiquidType"];

    private readonly Dictionary<int, int> _liquidTypeIdByObjectId;

    private DbcLiquidObjectTable(Dictionary<int, int> liquidTypeIdByObjectId, string? liquidTypeIdColumn)
    {
        _liquidTypeIdByObjectId = liquidTypeIdByObjectId;
        LiquidTypeIdColumn = liquidTypeIdColumn;
    }

    /// <summary>Number of rows loaded.</summary>
    public int RowCount => _liquidTypeIdByObjectId.Count;

    /// <summary>The column the LiquidTypeID was actually read from, for reporting.</summary>
    public string? LiquidTypeIdColumn { get; }

    /// <summary>An empty table. Every lookup misses; callers degrade and report (FR-009).</summary>
    public static DbcLiquidObjectTable Empty { get; } = new([], null);

    /// <summary>Build a table from explicit rows. Used by tests and by non-DBC sources.</summary>
    public static DbcLiquidObjectTable FromRows(IEnumerable<(int LiquidObjectId, int LiquidTypeId)> rows)
    {
        ArgumentNullException.ThrowIfNull(rows);

        Dictionary<int, int> map = [];
        foreach ((int liquidObjectId, int liquidTypeId) in rows)
            map[liquidObjectId] = liquidTypeId;

        return new DbcLiquidObjectTable(map, "(explicit rows)");
    }

    /// <summary>
    /// Load <c>LiquidObject</c> for <paramref name="buildVersion"/> through DBCD.
    /// </summary>
    public static DbcLiquidObjectTable Load(IDBCProvider dbcProvider, string definitionsDirectory, string buildVersion)
    {
        IDBCDStorage storage = DbcTableLoader.Load(dbcProvider, definitionsDirectory, buildVersion, "LiquidObject");
        return FromStorage(storage);
    }

    /// <summary>Build a table from an already-loaded DBCD storage.</summary>
    public static DbcLiquidObjectTable FromStorage(IDBCDStorage storage)
    {
        ArgumentNullException.ThrowIfNull(storage);

        string? column = DbcTableLoader.DetectColumn(storage, LiquidTypeIdColumns);
        string? idColumn = DbcTableLoader.DetectColumn(storage, DbcTableLoader.IdColumns);
        Dictionary<int, int> map = [];
        if (column is null)
            return new DbcLiquidObjectTable(map, null);

        foreach (DBCDRow row in storage.Values)
        {
            int liquidTypeId = DbcTableLoader.TryGetInt(row, column) ?? 0;
            if (liquidTypeId <= 0)
                continue;

            map[DbcTableLoader.ResolveRowId(row, idColumn)] = liquidTypeId;
        }

        return new DbcLiquidObjectTable(map, column);
    }

    /// <summary>Resolve a liquid-object id to its LiquidType row id.</summary>
    public bool TryGetLiquidTypeId(int liquidObjectId, out int liquidTypeId)
        => _liquidTypeIdByObjectId.TryGetValue(liquidObjectId, out liquidTypeId);

    /// <summary>All loaded rows, ordered by id. Reporting only.</summary>
    public IEnumerable<KeyValuePair<int, int>> Rows
        => _liquidTypeIdByObjectId.OrderBy(static pair => pair.Key);

    internal static string Describe(int value) => value.ToString(CultureInfo.InvariantCulture);
}
