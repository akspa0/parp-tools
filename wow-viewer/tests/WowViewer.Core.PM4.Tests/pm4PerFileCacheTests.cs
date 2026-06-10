using System;
using System.Collections.Generic;
using System.Numerics;
using WowViewer.Core.PM4.Caching;
using WowViewer.Core.PM4.Models;
using Xunit;

namespace WowViewer.Core.PM4.Tests;

public class Pm4PerFileCacheTests
{
    [Fact]
    public void TryGet_OnEmptyCache_ReturnsFalse()
    {
        Pm4PerFileCache cache = new(capacity: 8);
        Assert.False(cache.TryGet("World/Maps/x/x_00_00.pm4", 100, 42, out _));
        Assert.Equal(0, cache.Count);
    }

    [Fact]
    public void Set_ThenTryGet_Hit()
    {
        Pm4PerFileCache cache = new(capacity: 8);
        Pm4CachedTile tile = MakeTile(0, 0);
        Pm4PerFileCacheEntry entry = new(FileLength: 1024, LastWriteTicks: 42, Tiles: new[] { tile });
        cache.Set("a.pm4", entry);

        Assert.True(cache.TryGet("a.pm4", 1024, 42, out Pm4PerFileCacheEntry? actual));
        Assert.Same(entry, actual);
    }

    [Fact]
    public void TryGet_StampMismatch_ReturnsFalse()
    {
        Pm4PerFileCache cache = new(capacity: 8);
        cache.Set("a.pm4", new Pm4PerFileCacheEntry(1024, 42, new[] { MakeTile(0, 0) }));

        Assert.False(cache.TryGet("a.pm4", 1024, 99, out _));
        Assert.False(cache.TryGet("a.pm4", 9999, 42, out _));
    }

    [Fact]
    public void Remove_DropsEntry()
    {
        Pm4PerFileCache cache = new(capacity: 8);
        cache.Set("a.pm4", new Pm4PerFileCacheEntry(1024, 42, new[] { MakeTile(0, 0) }));
        Assert.True(cache.Remove("a.pm4"));
        Assert.False(cache.TryGet("a.pm4", 1024, 42, out _));
    }

    [Fact]
    public void Clear_DropsEverything()
    {
        Pm4PerFileCache cache = new(capacity: 8);
        cache.Set("a.pm4", new Pm4PerFileCacheEntry(1024, 42, new[] { MakeTile(0, 0) }));
        cache.Set("b.pm4", new Pm4PerFileCacheEntry(1024, 43, new[] { MakeTile(0, 1) }));
        cache.Clear();
        Assert.Equal(0, cache.Count);
    }

    [Fact]
    public void Capacity_One_EvictsOnSecondInsert()
    {
        Pm4PerFileCache cache = new(capacity: 1);
        cache.Set("a.pm4", new Pm4PerFileCacheEntry(1024, 42, new[] { MakeTile(0, 0) }));
        cache.Set("b.pm4", new Pm4PerFileCacheEntry(1024, 43, new[] { MakeTile(0, 1) }));
        Assert.Equal(1, cache.Count);
        // Exactly one of a/b should still be present. With capacity=1 and
        // a single-eviction policy, "a" was the victim of "b"'s insert,
        // so only "b" remains.
        Assert.False(cache.TryGet("a.pm4", 1024, 42, out _));
        Assert.True(cache.TryGet("b.pm4", 1024, 43, out _));
    }

    [Fact]
    public void Capacity_Respected_OnRepeatedInserts()
    {
        Pm4PerFileCache cache = new(capacity: 3);
        for (int i = 0; i < 10; i++)
        {
            string path = $"file_{i}.pm4";
            cache.Set(path, new Pm4PerFileCacheEntry(1024, i, new[] { MakeTile(i, i) }));
        }
        Assert.Equal(3, cache.Count);
    }

    [Fact]
    public void Constructor_RejectsNonPositiveCapacity()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Pm4PerFileCache(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Pm4PerFileCache(-1));
    }

    [Fact]
    public void TryGet_StampFoldingSplitFlag_HitWhenReadMatchesWrite()
    {
        // Reproduces the spec 054 stamp-folding pattern: the viewer
        // packs (splitByMscnRef, splitByConnectivity) into a single long
        // and uses that as the cache stamp. A read with the same fold
        // must hit; a read with a different fold must miss.
        Pm4PerFileCache cache = new(capacity: 4);
        Pm4CachedTile tile = MakeTile(0, 0);
        const long splitStamp = (1L << 32) | 1L;  // both flags on
        cache.Set("a.pm4", new Pm4PerFileCacheEntry(1024, splitStamp, new[] { tile }));

        Assert.True(cache.TryGet("a.pm4", 1024, splitStamp, out Pm4PerFileCacheEntry? sameStamp));
        Assert.NotNull(sameStamp);

        // Read with a different stamp (only one flag on) must miss.
        const long differentStamp = (1L << 32) | 0L;
        Assert.False(cache.TryGet("a.pm4", 1024, differentStamp, out _));
    }

    private static Pm4CachedTile MakeTile(int x, int y) =>
        new(
            TileX: x,
            TileY: y,
            PositionRefs: System.Array.Empty<Vector3>(),
            Objects: new[]
            {
                new Pm4CachedObject(
                    SourcePath: $"file_{x}_{y}.pm4",
                    MshdField00: 0,
                    MshdRegionId: 0,
                    MshdField08: 0,
                    Ck24: 0x000001,
                    Ck24Type: 0x40,
                    ObjectPartId: 0,
                    LinkGroupObjectId: 0,
                    LinkedPositionRefCount: 0,
                    LinkedPositionRefSummary: default,
                    SurfaceCount: 1,
                    TotalIndexCount: 3,
                    DominantGroupKey: 0,
                    DominantAttributeMask: 0,
                    DominantMscnRefIndex: 0,
                    AverageSurfaceHeight: 0f,
                    PlacementAnchor: Vector3.Zero,
                    BaseRotationRadians: 0f,
                    PlanarSwapPlanarAxes: false,
                    PlanarInvertU: false,
                    PlanarInvertV: false,
                    BoundsMin: Vector3.Zero,
                    BoundsMax: Vector3.Zero,
                    ConnectorKeys: System.Array.Empty<Pm4CachedConnectorKey>(),
                    Lines: System.Array.Empty<Pm4CachedLineSegment>(),
                    Triangles: System.Array.Empty<Pm4CachedTriangle>())
            });
}
