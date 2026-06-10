using System.Text.Json;
using System.Text.Json.Serialization;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Dbc;

/// <summary>
/// Lookup table for 3.3.5 <c>LiquidType.dbc</c> records. Maps a MH2O
/// <c>LiquidTypeId</c> (the DBC row ID) to the canonical
/// <see cref="AdtLiquidBasicType"/>.
///
/// The 3.3.5 Ghidra evidence in <c>specs/040-mh2o-mclq-liquid-type-determination/research.md</c>
/// (<c>FUN_00439760</c>, Material Bank liquid type lookup) shows that the
/// authoritative source is the DBC record's <c>Type</c> field at byte offset
/// <c>0x38</c>, with values <c>1=Water</c>, <c>2=Magma</c>, <c>3=Slime</c>.
/// On DBC miss the binary falls back to row ID 1 (water).
/// </summary>
/// <remarks>
/// <para>
/// The 3.3.5 DBC type field only distinguishes Water / Magma / Slime. The
/// wow-viewer renderer additionally distinguishes <see cref="AdtLiquidBasicType.Ocean"/>
/// (more opaque) from <see cref="AdtLiquidBasicType.Water"/> (less opaque) at
/// <c>LiquidRenderer.cs:92</c>. To preserve that visual distinction, this
/// table uses the DBC type field for Magma and Slime but falls back to a
/// hardcoded row-ID mapping for water-flavor rows (row 17 = Ocean, else Water).
/// This matches the behavior of <c>AdtLiquidReader.MapLiquidTypeId</c> at
/// <c>wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtLiquidReader.cs:275-284</c>
/// for the 17/19/20 case while still routing through the DBC.
/// </para>
/// <para>
/// Row IDs not present in the loaded DBC are recorded in
/// <see cref="MissingLiquidTypeIds"/> (deduplicated, one entry per unique
/// miss) and resolve to <see cref="AdtLiquidBasicType.Water"/> — the 3.3.5
/// <c>param_1 = 1</c> fallback.
/// </para>
/// </remarks>
public sealed class DbcLiquidTypeTable
{
    private const int DbcIdFieldIndex = 0x00 / 4;
    private const int DbcTypeFieldIndex = 0x38 / 4;

    private const uint DbcTypeWater = 1u;
    private const uint DbcTypeMagma = 2u;
    private const uint DbcTypeSlime = 3u;

    private const ushort LiquidTypeIdOcean = 17;
    private const ushort LiquidTypeIdMagma = 19;
    private const ushort LiquidTypeIdSlime = 20;

    private readonly Dictionary<int, AdtLiquidBasicType> _typeByLiquidTypeId = [];
    private readonly HashSet<int> _missingLiquidTypeIds = [];

    private DbcLiquidTypeTable()
    {
    }

    /// <summary>Number of DBC rows loaded into the table.</summary>
    public int RowCount => _typeByLiquidTypeId.Count;

    /// <summary>
    /// LiquidTypeIds that were queried via <see cref="ResolveBasicType"/> but
    /// were not present in the loaded DBC. One entry per unique miss.
    /// </summary>
    public IReadOnlyCollection<int> MissingLiquidTypeIds => _missingLiquidTypeIds;

    public static DbcLiquidTypeTable Load(string dbcPath)
    {
        ArgumentException.ThrowIfNullOrEmpty(dbcPath);
        byte[] data = File.ReadAllBytes(dbcPath);
        return LoadFromBytes(data);
    }

    public static DbcLiquidTypeTable LoadFromBytes(byte[] data)
    {
        ArgumentNullException.ThrowIfNull(data);
        DbcReader reader = DbcReader.Load(data);
        DbcLiquidTypeTable table = new();
        for (int rowIndex = 0; rowIndex < reader.Rows.Count; rowIndex++)
        {
            int id = reader.GetInt(rowIndex, DbcIdFieldIndex);
            uint typeField = reader.GetUInt(rowIndex, DbcTypeFieldIndex);
            table._typeByLiquidTypeId[id] = MapDbcTypeField(typeField);
        }
        return table;
    }

    public static DbcLiquidTypeTable LoadFromJson(string jsonPath)
    {
        ArgumentException.ThrowIfNullOrEmpty(jsonPath);
        string json = File.ReadAllText(jsonPath);
        return LoadFromJsonString(json);
    }

    public static DbcLiquidTypeTable LoadFromJsonString(string json)
    {
        ArgumentNullException.ThrowIfNull(json);
        DbcLiquidTypeFixture? fixture = JsonSerializer.Deserialize<DbcLiquidTypeFixture>(json);
        if (fixture is null)
            throw new InvalidDataException("DBC LiquidType JSON fixture deserialized to null.");
        if (fixture.Rows is null)
            return new DbcLiquidTypeTable();

        DbcLiquidTypeTable table = new();
        foreach (DbcLiquidTypeFixtureRow row in fixture.Rows)
        {
            table._typeByLiquidTypeId[row.Id] = MapDbcTypeField((uint)row.Type);
        }
        return table;
    }

    /// <summary>
    /// Resolve a MH2O <paramref name="liquidTypeId"/> to the canonical
    /// <see cref="AdtLiquidBasicType"/>. Records the id in
    /// <see cref="MissingLiquidTypeIds"/> on a miss (deduplicated) and
    /// returns <see cref="AdtLiquidBasicType.Water"/> per the 3.3.5
    /// <c>FUN_00439760</c> <c>param_1 = 1</c> fallback.
    /// </summary>
    public AdtLiquidBasicType ResolveBasicType(ushort liquidTypeId)
    {
        if (_typeByLiquidTypeId.TryGetValue(liquidTypeId, out AdtLiquidBasicType dbcType))
        {
            // DBC only distinguishes Water/Magma/Slime at the type-field level.
            // For water-flavor rows, apply the canonical row-ID mapping so
            // 17 -> Ocean (preserving the renderer's 0.7 opacity path) and
            // other water rows -> Water (0.45 opacity).
            if (dbcType == AdtLiquidBasicType.Water)
                return ResolveWaterByRowId(liquidTypeId);
            return dbcType;
        }

        _missingLiquidTypeIds.Add(liquidTypeId);
        return ResolveWaterByRowId(liquidTypeId);
    }

    private static AdtLiquidBasicType ResolveWaterByRowId(ushort liquidTypeId) => liquidTypeId switch
    {
        LiquidTypeIdOcean => AdtLiquidBasicType.Ocean,
        LiquidTypeIdMagma => AdtLiquidBasicType.Magma,
        LiquidTypeIdSlime => AdtLiquidBasicType.Slime,
        _ => AdtLiquidBasicType.Water,
    };

    private static AdtLiquidBasicType MapDbcTypeField(uint typeField) => typeField switch
    {
        DbcTypeWater => AdtLiquidBasicType.Water,
        DbcTypeMagma => AdtLiquidBasicType.Magma,
        DbcTypeSlime => AdtLiquidBasicType.Slime,
        _ => AdtLiquidBasicType.Water,
    };

    private sealed class DbcLiquidTypeFixture
    {
        [JsonPropertyName("rows")]
        public List<DbcLiquidTypeFixtureRow>? Rows { get; set; }
    }

    private sealed class DbcLiquidTypeFixtureRow
    {
        [JsonPropertyName("id")]
        public int Id { get; set; }

        [JsonPropertyName("name")]
        public string? Name { get; set; }

        [JsonPropertyName("type")]
        public int Type { get; set; }

        [JsonPropertyName("flags")]
        public int Flags { get; set; }
    }
}
