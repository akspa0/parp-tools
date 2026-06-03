using System.Text.Json;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests.Dbc;

public sealed class DbcLiquidTypeTableTests
{
    private const string FixtureJson = """
    {
      "rows": [
        { "id": 1,  "name": "Water (generic)",      "type": 1, "flags": 0 },
        { "id": 13, "name": "River (Dark)",          "type": 1, "flags": 0 },
        { "id": 14, "name": "Still Water (generic)", "type": 1, "flags": 0 },
        { "id": 17, "name": "Ocean (deep)",          "type": 1, "flags": 0 },
        { "id": 19, "name": "Magma (lava)",          "type": 2, "flags": 0 },
        { "id": 20, "name": "Slime (green)",         "type": 3, "flags": 0 }
      ]
    }
    """;

    [Fact]
    public void LoadFromJsonString_ParsesAllRows()
    {
        DbcLiquidTypeTable table = DbcLiquidTypeTable.LoadFromJsonString(FixtureJson);

        Assert.Equal(6, table.RowCount);
        Assert.Empty(table.MissingLiquidTypeIds);
    }

    [Theory]
    [InlineData((ushort)17, AdtLiquidBasicType.Ocean)]
    [InlineData((ushort)19, AdtLiquidBasicType.Magma)]
    [InlineData((ushort)20, AdtLiquidBasicType.Slime)]
    [InlineData((ushort)1, AdtLiquidBasicType.Water)]
    [InlineData((ushort)13, AdtLiquidBasicType.Water)]
    [InlineData((ushort)14, AdtLiquidBasicType.Water)]
    public void ResolveBasicType_KnownRowIds_MapToExpectedBasicType(ushort liquidTypeId, AdtLiquidBasicType expected)
    {
        DbcLiquidTypeTable table = DbcLiquidTypeTable.LoadFromJsonString(FixtureJson);

        Assert.Equal(expected, table.ResolveBasicType(liquidTypeId));
    }

    [Fact]
    public void ResolveBasicType_UnknownRowId_DefaultsToWaterAndRecordsMiss()
    {
        DbcLiquidTypeTable table = DbcLiquidTypeTable.LoadFromJsonString(FixtureJson);

        Assert.Equal(AdtLiquidBasicType.Water, table.ResolveBasicType(9999));
        Assert.Single(table.MissingLiquidTypeIds);
        Assert.Contains(9999, table.MissingLiquidTypeIds);
    }

    [Fact]
    public void ResolveBasicType_RepeatedMiss_RecordedOnce()
    {
        DbcLiquidTypeTable table = DbcLiquidTypeTable.LoadFromJsonString(FixtureJson);

        _ = table.ResolveBasicType(9999);
        _ = table.ResolveBasicType(9999);
        _ = table.ResolveBasicType(9999);

        Assert.Single(table.MissingLiquidTypeIds);
        Assert.Contains(9999, table.MissingLiquidTypeIds);
    }

    [Fact]
    public void ResolveBasicType_DistinctMisses_AllRecorded()
    {
        DbcLiquidTypeTable table = DbcLiquidTypeTable.LoadFromJsonString(FixtureJson);

        _ = table.ResolveBasicType(9999);
        _ = table.ResolveBasicType(9998);
        _ = table.ResolveBasicType(1234);

        Assert.Equal(3, table.MissingLiquidTypeIds.Count);
        Assert.Contains(9999, table.MissingLiquidTypeIds);
        Assert.Contains(9998, table.MissingLiquidTypeIds);
        Assert.Contains(1234, table.MissingLiquidTypeIds);
    }

    [Fact]
    public void LoadFromJsonString_EmptyFixture_ReturnsEmptyTable()
    {
        DbcLiquidTypeTable table = DbcLiquidTypeTable.LoadFromJsonString("""{ "rows": [] }""");

        Assert.Equal(0, table.RowCount);
        Assert.Equal(AdtLiquidBasicType.Ocean, table.ResolveBasicType(17));
        Assert.Single(table.MissingLiquidTypeIds);
        Assert.Contains(17, table.MissingLiquidTypeIds);
    }

    [Fact]
    public void LoadFromJsonString_NullRows_ReturnsEmptyTable()
    {
        DbcLiquidTypeTable table = DbcLiquidTypeTable.LoadFromJsonString("{}");

        Assert.Equal(0, table.RowCount);
        Assert.Equal(AdtLiquidBasicType.Ocean, table.ResolveBasicType(17));
        Assert.Equal(AdtLiquidBasicType.Water, table.ResolveBasicType(9999));
        Assert.Equal(2, table.MissingLiquidTypeIds.Count);
    }

    [Fact]
    public void LoadFromJsonString_NullJson_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => DbcLiquidTypeTable.LoadFromJsonString(null!));
    }

    [Fact]
    public void LoadFromJsonString_InvalidJson_Throws()
    {
        Assert.Throws<JsonException>(() => DbcLiquidTypeTable.LoadFromJsonString("not json"));
    }

    [Fact]
    public void LoadFromBytes_SynthesizedDbc_RoundTripsKnownRows()
    {
        byte[] dbcBytes = BuildSyntheticLiquidTypeDbc([
            (1, 1u), (13, 1u), (14, 1u), (17, 1u), (19, 2u), (20, 3u),
        ]);

        DbcLiquidTypeTable table = DbcLiquidTypeTable.LoadFromBytes(dbcBytes);

        Assert.Equal(6, table.RowCount);
        Assert.Equal(AdtLiquidBasicType.Ocean, table.ResolveBasicType(17));
        Assert.Equal(AdtLiquidBasicType.Magma, table.ResolveBasicType(19));
        Assert.Equal(AdtLiquidBasicType.Slime, table.ResolveBasicType(20));
        Assert.Equal(AdtLiquidBasicType.Water, table.ResolveBasicType(1));
        Assert.Equal(AdtLiquidBasicType.Water, table.ResolveBasicType(13));
        Assert.Equal(AdtLiquidBasicType.Water, table.ResolveBasicType(14));
        Assert.Equal(AdtLiquidBasicType.Water, table.ResolveBasicType(9999));
        Assert.Single(table.MissingLiquidTypeIds);
    }

    [Fact]
    public void Load_NullPath_Throws()
    {
        Assert.ThrowsAny<ArgumentException>(() => DbcLiquidTypeTable.Load(null!));
    }

    [Fact]
    public void LoadFromBytes_NullData_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => DbcLiquidTypeTable.LoadFromBytes(null!));
    }

    private static byte[] BuildSyntheticLiquidTypeDbc((int Id, uint TypeField)[] rows)
    {
        const int recordSize = 0x40;

        using MemoryStream ms = new();
        using BinaryWriter writer = new(ms);

        writer.Write(0x43424457u);                  // "WDBC" magic
        writer.Write((uint)rows.Length);            // record count
        writer.Write((uint)(recordSize / 4));       // field count
        writer.Write((uint)recordSize);             // record size
        writer.Write(0u);                            // string block size (empty)

        foreach ((int id, uint typeField) in rows)
        {
            byte[] record = new byte[recordSize];
            BitConverter.GetBytes(id).CopyTo(record, 0x00);
            BitConverter.GetBytes(typeField).CopyTo(record, 0x38);
            writer.Write(record);
        }

        return ms.ToArray();
    }
}
