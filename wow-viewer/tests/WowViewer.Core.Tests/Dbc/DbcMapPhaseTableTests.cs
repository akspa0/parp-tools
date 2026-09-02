using WowViewer.Core.IO.Dbc;

namespace WowViewer.Core.Tests.Dbc;

/// <summary>
/// Spec 203 T001. The rows here are the ones actually measured in MoP Beta 5.0.1.15464's
/// <c>Map.dbc</c> via <c>inspect dbc dump --table Map</c>, so the fixture is a sample of the real
/// client rather than a shape invented to match the implementation.
/// </summary>
public sealed class DbcMapPhaseTableTests
{
    private static DbcMapPhaseTable CreateMeasuredTable() => DbcMapPhaseTable.FromRows(
    [
        new MapPhaseRecord(0, "Azeroth", "Eastern Kingdoms", -1),
        new MapPhaseRecord(1, "Kalimdor", "Kalimdor", -1),
        new MapPhaseRecord(638, "Gilneas", "Gilneas", 654),
        new MapPhaseRecord(654, "Gilneas2", "Gilneas2", -1),
        new MapPhaseRecord(655, "GilneasPhase1", "GilneasPhase1", 654),
        new MapPhaseRecord(656, "GilneasPhase2", "GilneasPhase2", 654),
        new MapPhaseRecord(870, "HawaiiMainLand", "Pandaria", -1),
        new MapPhaseRecord(971, "JadeForestAllianceHubPhase", "Jade Forest Alliance Hub Phase", 870),
        new MapPhaseRecord(972, "JadeForestBattlefieldPhase", "Jade Forest Battlefield Phase", 870),
        new MapPhaseRecord(719, "MountHyjalPhase1", "Mount Hyjal Phase 1", 1),
    ]);

    [Fact]
    public void GetChildMapsOf_Pandaria_FindsBothJadeForestPhases()
    {
        IReadOnlyList<MapPhaseRecord> children = CreateMeasuredTable().GetChildMapsOf("HawaiiMainLand");

        Assert.Equal(2, children.Count);
        Assert.Contains(children, c => c.Directory == "JadeForestAllianceHubPhase");
        Assert.Contains(children, c => c.Directory == "JadeForestBattlefieldPhase");
    }

    [Fact]
    public void GetChildMapsOf_IsCaseInsensitiveOnTheDirectory()
    {
        Assert.Equal(2, CreateMeasuredTable().GetChildMapsOf("hawaiimainland").Count);
    }

    [Fact]
    public void GetChildMapsOf_MapWithNoChildren_ReturnsEmptyRatherThanThrowing()
    {
        Assert.DoesNotContain(CreateMeasuredTable().GetChildMapsOf("Kalimdor"), c => c.Directory == "Kalimdor");
        Assert.Empty(CreateMeasuredTable().GetChildMapsOf("Azeroth"));
    }

    [Fact]
    public void GetChildMapsOf_UnknownMap_ReturnsEmpty()
    {
        Assert.Empty(CreateMeasuredTable().GetChildMapsOf("NoSuchMap"));
    }

    [Fact]
    public void ChildMaps_ExcludeRowsWithoutAParent()
    {
        var children = CreateMeasuredTable().ChildMaps.ToList();

        Assert.All(children, c => Assert.True(c.HasParent));
        Assert.DoesNotContain(children, c => c.Directory == "Azeroth");
    }

    [Fact]
    public void ParentSentinel_NegativeOne_IsNotAParent()
    {
        Assert.True(CreateMeasuredTable().TryGetByDirectory("Azeroth", out MapPhaseRecord azeroth));
        Assert.False(azeroth.HasParent);
    }

    [Fact]
    public void AChildMapMayItselfBeRealTerrain_NotOnlyAPhase()
    {
        // Gilneas names Gilneas2 as its parent on 5.0.1. The caller decides what that means; the
        // table must not silently filter it out on a guess about naming.
        IReadOnlyList<MapPhaseRecord> children = CreateMeasuredTable().GetChildMapsOf("Gilneas2");

        Assert.Equal(3, children.Count);
        Assert.Contains(children, c => c.Directory == "Gilneas");
    }

    [Fact]
    public void EmptyTable_ResolvesNothingAndDoesNotThrow()
    {
        Assert.Empty(DbcMapPhaseTable.Empty.GetChildMapsOf("HawaiiMainLand"));
        Assert.False(DbcMapPhaseTable.Empty.TryGetByDirectory("HawaiiMainLand", out _));
        Assert.Equal(0, DbcMapPhaseTable.Empty.RowCount);
    }
}

/// <summary>
/// <c>Phase.dbc</c> rows measured on 5.0.1.15464. The point of these is the negative result:
/// the table names phases but cannot associate one with a map.
/// </summary>
public sealed class DbcPhaseTableTests
{
    private static DbcPhaseTable CreateMeasuredTable() => DbcPhaseTable.FromRows(
        [
            new PhaseRecord(50, "Gilneas Lev 6", 0),
            new PhaseRecord(52, "Gilneas Lev 8", 4),
            new PhaseRecord(101, "Gilneas City Unphased Terrain Swap", 4),
            new PhaseRecord(102, "Gilneas City Phase 1 Terrain Swap", 4),
            new PhaseRecord(106, "Gilneas City Phase 5 Terrain Swap", 0),
        ],
        [(171, 367), (172, 367), (169, 368), (170, 368), (171, 368)]);

    [Fact]
    public void TerrainSwapPhases_AreIdentifiedByName()
    {
        var swaps = CreateMeasuredTable().TerrainSwapPhases.ToList();

        Assert.Equal(3, swaps.Count);
        Assert.All(swaps, phase => Assert.Contains("Terrain Swap", phase.Name));
    }

    [Fact]
    public void TerrainSwap_IsNotDerivedFromFlags()
    {
        // Flags 4 appears on most terrain-swap rows but not all of them (phase 106 is flags 0), and
        // it also appears on rows that are not terrain swaps (phase 52). Treating flag 4 as "is a
        // terrain swap" would be wrong in both directions.
        DbcPhaseTable table = CreateMeasuredTable();

        Assert.True(table.TryGetPhase(106, out PhaseRecord unflaggedSwap));
        Assert.Equal(0, unflaggedSwap.Flags);
        Assert.True(unflaggedSwap.LooksLikeTerrainSwap);

        Assert.True(table.TryGetPhase(52, out PhaseRecord flaggedNonSwap));
        Assert.Equal(4, flaggedNonSwap.Flags);
        Assert.False(flaggedNonSwap.LooksLikeTerrainSwap);
    }

    [Fact]
    public void GetPhaseIdsInGroup_ReturnsEveryMember()
    {
        Assert.Equal([169, 170, 171], CreateMeasuredTable().GetPhaseIdsInGroup(368));
        Assert.Equal([171, 172], CreateMeasuredTable().GetPhaseIdsInGroup(367));
    }

    [Fact]
    public void GetPhaseIdsInGroup_UnknownGroup_ReturnsEmpty()
    {
        Assert.Empty(CreateMeasuredTable().GetPhaseIdsInGroup(99999));
    }

    [Fact]
    public void PhaseRecord_CarriesNoMapReference()
    {
        // The load-bearing negative result: Phase.dbc's 5.0.1 layout is ID, Name, Flags. Anything
        // that wants to know which map a phase belongs to must use Map.dbc.ParentMapID instead.
        Assert.Equal(3, typeof(PhaseRecord).GetProperties().Count(p => p.Name is "PhaseId" or "Name" or "Flags"));
        Assert.DoesNotContain(typeof(PhaseRecord).GetProperties(), p => p.Name.Contains("Map", StringComparison.OrdinalIgnoreCase));
    }
}
