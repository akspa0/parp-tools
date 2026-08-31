using System;
using System.Numerics;
using WowViewer.Core.Editor.Operations;
using WowViewer.Core.Editor.Staging;
using Xunit;

namespace WowViewer.Core.Tests;

public class MultiClientMapStagingTests
{
    [Fact]
    public void RegisterAndUnregisterClientSources_MaintainsCatalog()
    {
        var service = new MultiClientMapStagingService();

        var sourceAlpha = new ClientSourceDescriptor
        {
            ClientId = "alpha_053",
            DisplayName = "WoW 0.5.3 Alpha",
            BuildVersion = "3368",
            BasePath = @"H:\CLIENTS\alpha_053",
        };

        var sourceLk = new ClientSourceDescriptor
        {
            ClientId = "lk_335a",
            DisplayName = "WoW 3.3.5a WotLK",
            BuildVersion = "12340",
            BasePath = @"H:\CLIENTS\lk_335a",
        };

        Assert.True(service.RegisterClientSource(sourceAlpha));
        Assert.True(service.RegisterClientSource(sourceLk));
        Assert.Equal(2, service.MountedSources.Count);

        var retrieved = service.GetSource("alpha_053");
        Assert.NotNull(retrieved);
        Assert.Equal("3368", retrieved.BuildVersion);

        Assert.True(service.UnregisterClientSource("alpha_053"));
        Assert.Single(service.MountedSources);
        Assert.Null(service.GetSource("alpha_053"));
    }

    [Fact]
    public void StageChunkPayload_TracksProvenanceAndPositions()
    {
        var service = new MultiClientMapStagingService();
        service.RegisterClientSource(new ClientSourceDescriptor
        {
            ClientId = "cata_401",
            DisplayName = "WoW 4.0.1 Cataclysm",
            BuildVersion = "13164",
            BasePath = @"H:\CLIENTS\cata_401",
        });

        service.CreateRestorationMap("HyjalRestoration", "3.3.5a");

        var chunkRecord = new TransposedChunkRecord
        {
            RelativeGx = 0,
            RelativeGy = 0,
            Heights = new float[145],
            AreaId = 1234,
            HoleMask = 0,
        };

        var targetCoord = new GlobalChunkCoordinate(32 * 16 + 5, 48 * 16 + 7);

        bool success = service.StageChunkPayload(
            "cata_401",
            "Azeroth",
            sourceTileX: 32,
            sourceTileY: 48,
            sourceChunkX: 5,
            sourceChunkY: 7,
            targetCoord,
            chunkRecord,
            checksumSha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");

        Assert.True(success);
        Assert.Equal(1, service.GetStagedChunkCount());

        var stagedMap = service.ActiveRestorationMap;
        Assert.NotNull(stagedMap);
        Assert.True(stagedMap.StagedChunks.ContainsKey(targetCoord));

        var entry = stagedMap.StagedChunks[targetCoord];
        Assert.Equal("cata_401", entry.Provenance.SourceClientId);
        Assert.Equal("13164", entry.Provenance.SourceBuild);
        Assert.Equal(32, entry.Provenance.SourceTileX);
        Assert.Equal(48, entry.Provenance.SourceTileY);
        Assert.Equal(5, entry.Provenance.SourceChunkX);
        Assert.Equal(7, entry.Provenance.SourceChunkY);
        Assert.Equal("Azeroth", entry.Provenance.SourceMapName);
        Assert.Equal("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", entry.Provenance.ChecksumSha256);
    }

    [Fact]
    public void StagePlacement_AppendsPlacementsWithProvenance()
    {
        var service = new MultiClientMapStagingService();
        service.RegisterClientSource(new ClientSourceDescriptor
        {
            ClientId = "alpha_053",
            DisplayName = "WoW 0.5.3 Alpha",
            BuildVersion = "3368",
            BasePath = @"H:\CLIENTS\alpha_053",
        });

        var placement = new TransposedObjectPlacement
        {
            IsWmo = true,
            AssetPath = @"World\wmo\Dungeon\Crypt\Crypt.wmo",
            RelativePosition = new Vector3(100f, 200f, 300f),
            Rotation = new Vector3(0f, 45f, 0f),
            Scale = 1.0f,
            UniqueId = 99901,
        };

        bool success = service.StagePlacement("alpha_053", "Kalimdor", 42, 50, placement);
        Assert.True(success);
        Assert.Equal(1, service.GetStagedPlacementCount());

        var stagedPlacement = service.ActiveRestorationMap!.StagedPlacements[0];
        Assert.Equal("alpha_053", stagedPlacement.Provenance.SourceClientId);
        Assert.Equal(99901, stagedPlacement.Placement.UniqueId);
        Assert.Equal(@"World\wmo\Dungeon\Crypt\Crypt.wmo", stagedPlacement.Placement.AssetPath);
    }
}
