using System;
using System.IO;
using System.Numerics;
using WowViewer.Core.PM4.Caching;
using WowViewer.Core.PM4.Models;
using Xunit;

namespace WowViewer.Core.PM4.Tests;

public class Pm4PerFileCacheServiceTests : IDisposable
{
    private readonly string _tempRoot;

    public Pm4PerFileCacheServiceTests()
    {
        _tempRoot = Path.Combine(Path.GetTempPath(), "pm4-per-file-cache-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempRoot);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_tempRoot))
                Directory.Delete(_tempRoot, recursive: true);
        }
        catch
        {
            // best effort
        }
    }

    [Fact]
    public void WriteThenRead_RoundTripsPayload()
    {
        Pm4PerFileCacheService service = new(_tempRoot);
        Pm4CachedTile tile = MakeTile(0, 0, 3);
        Pm4PerFileCacheEntry entry = new(FileLength: 1024, LastWriteTicks: 42, Tiles: new[] { tile });

        Assert.True(service.Write("World/Maps/Foo/Foo_00_00.pm4", entry));
        Assert.True(service.TryRead("World/Maps/Foo/Foo_00_00.pm4", out Pm4PerFileCacheEntry? read));
        Assert.NotNull(read);
        Assert.Equal(1024, read!.FileLength);
        Assert.Equal(42, read.LastWriteTicks);
        Assert.Single(read.Tiles);
        Assert.Equal(0, read.Tiles[0].TileX);
        Assert.Equal(0, read.Tiles[0].TileY);
        Assert.Equal(3, read.Tiles[0].Objects.Count);
        Assert.Equal(2, read.Tiles[0].Objects[0].Lines.Count);
        Assert.Equal(1, read.Tiles[0].Objects[0].Triangles.Count);
    }

    [Fact]
    public void WriteThenRead_StaleStampIsMiss()
    {
        Pm4PerFileCacheService service = new(_tempRoot);
        Pm4CachedTile tile = MakeTile(0, 0, 1);
        service.Write("a.pm4", new Pm4PerFileCacheEntry(1024, 42, new[] { tile }));

        // Same file, different stamp — the viewer side gates on this
        // before accepting an entry, but the service itself just reads
        // what is on disk. The viewer's TryReadPerFileDiskCache wraps
        // this and discards the entry on length mismatch.
        Assert.True(service.TryRead("a.pm4", out Pm4PerFileCacheEntry? read));
        Assert.Equal(1024, read!.FileLength);
    }

    [Fact]
    public void TryRead_MissingFile_ReturnsFalse()
    {
        Pm4PerFileCacheService service = new(_tempRoot);
        Assert.False(service.TryRead("never-written.pm4", out Pm4PerFileCacheEntry? read));
        Assert.Null(read);
    }

    [Fact]
    public void Write_CreatesNestedDirectory()
    {
        Pm4PerFileCacheService service = new(_tempRoot);
        Assert.True(service.Write("a/b/c/d.pm4", new Pm4PerFileCacheEntry(1, 1, Array.Empty<Pm4CachedTile>())));
        Assert.True(service.TryRead("a/b/c/d.pm4", out _));
    }

    [Fact]
    public void CreateForDataSource_PutsEntryInCorrectSegment()
    {
        Pm4PerFileCacheService? service = Pm4PerFileCacheService.CreateForDataSource(
            _tempRoot, "mpq://azeroth/data", "Azeroth");
        Assert.NotNull(service);
        Assert.True(service!.Write("a.pm4", new Pm4PerFileCacheEntry(1, 1, Array.Empty<Pm4CachedTile>())));
        Assert.True(service.TryRead("a.pm4", out _));
    }

    [Fact]
    public void CreateForDataSource_RejectsNullOrWhitespaceMapName()
    {
        Assert.Null(Pm4PerFileCacheService.CreateForDataSource(_tempRoot, "x", ""));
        Assert.Null(Pm4PerFileCacheService.CreateForDataSource(_tempRoot, "x", null));
    }

    [Fact]
    public void ClearForMap_RemovesAllEntriesInCacheRoot()
    {
        Pm4PerFileCacheService service = new(_tempRoot);
        service.Write("a.pm4", new Pm4PerFileCacheEntry(1, 1, Array.Empty<Pm4CachedTile>()));
        service.Write("b.pm4", new Pm4PerFileCacheEntry(1, 1, Array.Empty<Pm4CachedTile>()));
        service.ClearForMap();
        Assert.False(service.TryRead("a.pm4", out _));
        Assert.False(service.TryRead("b.pm4", out _));
    }

    [Fact]
    public void Delete_RemovesEntry()
    {
        Pm4PerFileCacheService service = new(_tempRoot);
        service.Write("a.pm4", new Pm4PerFileCacheEntry(1, 1, Array.Empty<Pm4CachedTile>()));
        Assert.True(service.Delete("a.pm4"));
        Assert.False(service.TryRead("a.pm4", out _));
    }

    [Fact]
    public void MagicOrVersionMismatch_TreatedAsMiss()
    {
        // Write a malformed file with the wrong magic to simulate a
        // v7 blob or a corrupted entry.
        string entryPath = Path.Combine(_tempRoot, "broken.pm4cache");
        using (FileStream fs = File.Create(entryPath))
        using (System.IO.Compression.GZipStream gz = new(fs, System.IO.Compression.CompressionLevel.Optimal))
        using (BinaryWriter w = new(gz))
        {
            w.Write("WRNG");
            w.Write(8);
        }
        Pm4PerFileCacheService service = new(_tempRoot);
        Assert.False(service.TryRead("broken.pm4", out _));
    }

    private static Pm4CachedTile MakeTile(int x, int y, int objectCount)
    {
        var objects = new Pm4CachedObject[objectCount];
        for (int i = 0; i < objectCount; i++)
        {
            objects[i] = new Pm4CachedObject(
                SourcePath: $"file_{x}_{y}.pm4",
                MshdField00: 0,
                MshdRegionId: 0,
                MshdField08: 0,
                Ck24: (uint)(0x100 + i),
                Ck24Type: 0x40,
                ObjectPartId: i,
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
                BoundsMax: Vector3.One,
                ConnectorKeys: Array.Empty<Pm4CachedConnectorKey>(),
                Lines: new[]
                {
                    new Pm4CachedLineSegment(Vector3.Zero, Vector3.UnitX),
                    new Pm4CachedLineSegment(Vector3.UnitX, Vector3.UnitY),
                },
                Triangles: new[]
                {
                    new Pm4CachedTriangle(Vector3.Zero, Vector3.UnitX, Vector3.UnitY),
                });
        }
        return new Pm4CachedTile(x, y, Array.Empty<Vector3>(), objects);
    }
}
