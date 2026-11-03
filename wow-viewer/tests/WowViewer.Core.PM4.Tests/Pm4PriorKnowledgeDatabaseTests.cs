using System.Linq;
using WowViewer.Core.PM4.Matching;
using Xunit;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4PriorKnowledgeDatabaseTests
{
    [Fact]
    public void Database_Initializes_WithCuratedAndHarvestedAssets()
    {
        var db = Pm4PriorKnowledgeDatabase.Instance;
        Assert.NotNull(db);
        Assert.NotEmpty(db.AllAssets);

        // Verify key architectural assets are present
        Assert.True(db.TryGetAsset("ND_WINTERTAUREN_BRIDGE_01.WMO", out var bridgeAsset));
        Assert.NotNull(bridgeAsset);
        Assert.Equal("Wintertauren", bridgeAsset!.Family);
        Assert.Equal("WMO", bridgeAsset.AssetKind);

        Assert.True(db.TryGetAsset("PVP_TAURENGATE01.WMO", out var gateAsset));
        Assert.NotNull(gateAsset);
        Assert.Equal("Tauren", gateAsset!.Family);

        Assert.True(db.TryGetAsset("GUARDTOWER.WMO", out var towerAsset));
        Assert.NotNull(towerAsset);
    }

    [Fact]
    public void DetectFamily_AccuratelyClassifiesPathOrName()
    {
        Assert.Equal("Wintertauren", Pm4PriorKnowledgeDatabase.DetectFamily(@"WORLD\WMO\NORTHREND\BUILDINGS\WINTERTAUREN\ND_WINTERTAUREN_BRIDGE_01.WMO"));
        Assert.Equal("Wintertauren", Pm4PriorKnowledgeDatabase.DetectFamily(@"WORLD\EXPANSION02\DOODADS\WINTERTAUREN\WT_BENCH_01.M2"));
        Assert.Equal("Tauren", Pm4PriorKnowledgeDatabase.DetectFamily(@"WORLD\WMO\KALIMDOR\BUILDINGS\TAURENHUTS\TAURENHUTD.WMO"));
        Assert.Equal("Bridges", Pm4PriorKnowledgeDatabase.DetectFamily(@"WORLD\WMO\AZEROTH\BUILDINGS\WETLANDSFOOTBRIDGE01\WETLANDSFOOTBRIDGE01.WMO"));
        Assert.Equal("HumanArchitecture", Pm4PriorKnowledgeDatabase.DetectFamily(@"WORLD\WMO\AZEROTH\BUILDINGS\HUMANTWOSTORY\HUMANTWOSTORY.WMO"));
        Assert.Equal("Keeps & Castles", Pm4PriorKnowledgeDatabase.DetectFamily(@"WORLD\WMO\AZEROTH\BUILDINGS\BIG_KEEP\BIG_KEEP.WMO"));
    }

    [Fact]
    public void GetNeighborhoodAssets_ReturnsRelatedFamilyModels()
    {
        var db = Pm4PriorKnowledgeDatabase.Instance;
        var wintertaurenAssets = db.GetNeighborhoodAssets("ND_WINTERTAUREN_BRIDGE_01.WMO");

        Assert.NotEmpty(wintertaurenAssets);
        Assert.Contains(wintertaurenAssets, a => a.ModelName.Contains("WINTERTAUREN", System.StringComparison.OrdinalIgnoreCase) || a.ModelName.StartsWith("WT_", System.StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void GetDatabaseCandidates_ReturnsCandidatesForMatching()
    {
        var db = Pm4PriorKnowledgeDatabase.Instance;
        var candidates = db.GetDatabaseCandidates(targetFamily: "Wintertauren");

        Assert.NotEmpty(candidates);
        // The first candidate should be from Wintertauren family
        Assert.True(candidates[0].ModelPath.Contains("WINTERTAUREN", System.StringComparison.OrdinalIgnoreCase) || candidates[0].ModelName.StartsWith("WT_", System.StringComparison.OrdinalIgnoreCase));
    }
}

