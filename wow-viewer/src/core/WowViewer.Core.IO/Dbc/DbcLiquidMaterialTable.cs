using DBCD;
using DBCD.Providers;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Dbc;

/// <summary>
/// Lookup table for <c>LiquidMaterial.dbc</c>: a material id to the liquid vertex format (LVF) that
/// says how a MH2O vertex block is laid out.
/// </summary>
/// <remarks>
/// This is the last link of the chain <c>LiquidObject &#8594; LiquidType &#8594; LiquidMaterial &#8594; LVF</c>,
/// and the only one that produces a value the MH2O reader can act on. See
/// <c>specs/205-mh2o-liquid-object-vertex-format/</c>.
/// </remarks>
public sealed class DbcLiquidMaterialTable
{
    private static readonly string[] LvfColumns = ["LVF", "VertexFormat", "LiquidVertexFormat"];

    private readonly Dictionary<int, int> _lvfByMaterialId;

    private DbcLiquidMaterialTable(Dictionary<int, int> lvfByMaterialId, string? lvfColumn)
    {
        _lvfByMaterialId = lvfByMaterialId;
        LvfColumn = lvfColumn;
    }

    /// <summary>Number of rows loaded.</summary>
    public int RowCount => _lvfByMaterialId.Count;

    /// <summary>The column the LVF was actually read from, for reporting.</summary>
    public string? LvfColumn { get; }

    /// <summary>An empty table. Every lookup misses; callers degrade and report (FR-009).</summary>
    public static DbcLiquidMaterialTable Empty { get; } = new([], null);

    /// <summary>Build a table from explicit rows. Used by tests and by non-DBC sources.</summary>
    public static DbcLiquidMaterialTable FromRows(IEnumerable<(int MaterialId, int Lvf)> rows)
    {
        ArgumentNullException.ThrowIfNull(rows);

        Dictionary<int, int> map = [];
        foreach ((int materialId, int lvf) in rows)
            map[materialId] = lvf;

        return new DbcLiquidMaterialTable(map, "(explicit rows)");
    }

    /// <summary>Load <c>LiquidMaterial</c> for <paramref name="buildVersion"/> through DBCD.</summary>
    public static DbcLiquidMaterialTable Load(IDBCProvider dbcProvider, string definitionsDirectory, string buildVersion)
    {
        IDBCDStorage storage = DbcTableLoader.Load(dbcProvider, definitionsDirectory, buildVersion, "LiquidMaterial");
        return FromStorage(storage);
    }

    /// <summary>Build a table from an already-loaded DBCD storage.</summary>
    public static DbcLiquidMaterialTable FromStorage(IDBCDStorage storage)
    {
        ArgumentNullException.ThrowIfNull(storage);

        string? column = DbcTableLoader.DetectColumn(storage, LvfColumns);
        string? idColumn = DbcTableLoader.DetectColumn(storage, DbcTableLoader.IdColumns);
        Dictionary<int, int> map = [];
        if (column is null)
            return new DbcLiquidMaterialTable(map, null);

        foreach (DBCDRow row in storage.Values)
        {
            int? lvf = DbcTableLoader.TryGetInt(row, column);
            if (lvf is null)
                continue;

            map[DbcTableLoader.ResolveRowId(row, idColumn)] = lvf.Value;
        }

        return new DbcLiquidMaterialTable(map, column);
    }

    /// <summary>Resolve a material id to its liquid vertex format.</summary>
    public bool TryGetVertexFormat(int materialId, out AdtLiquidVertexFormat vertexFormat)
    {
        vertexFormat = default;
        if (!_lvfByMaterialId.TryGetValue(materialId, out int lvf))
            return false;

        // Only 0-3 are defined. An out-of-range LVF is a decode failure, not a value to clamp:
        // clamping would put a heightmap read on a depth-only block.
        if (lvf is < 0 or > 3)
            return false;

        vertexFormat = (AdtLiquidVertexFormat)lvf;
        return true;
    }

    /// <summary>Raw LVF for a material id, including out-of-range values. Reporting only.</summary>
    public bool TryGetRawLvf(int materialId, out int lvf) => _lvfByMaterialId.TryGetValue(materialId, out lvf);

    /// <summary>All loaded rows, ordered by id. Reporting only.</summary>
    public IEnumerable<KeyValuePair<int, int>> Rows => _lvfByMaterialId.OrderBy(static pair => pair.Key);
}
