using System.Numerics;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4ObjectSegmentBuilderTests
{
    [Fact]
    public void Build_DevelopmentTile_ProducesStableUniqueSegmentIds()
    {
        Pm4ResearchDocument document = Pm4ResearchReader.ReadFile(Pm4TestPaths.DevelopmentTilePath);

        IReadOnlyList<Pm4BuiltObjectSegment> first = Pm4ObjectSegmentBuilder.Build(document, 0, 0);
        IReadOnlyList<Pm4BuiltObjectSegment> second = Pm4ObjectSegmentBuilder.Build(document, 0, 0);

        Assert.NotEmpty(first);
        Assert.Equal(first.Select(static segment => segment.Segment.SegmentId), second.Select(static segment => segment.Segment.SegmentId));
        Assert.Equal(first.Count, first.Select(static segment => segment.Segment.SegmentId).Distinct(StringComparer.Ordinal).Count());
        Assert.All(first, static segment =>
        {
            Assert.Contains("0_0", segment.Segment.TileCoordinates);
            Assert.Equal(Pm4SegmentSignalExtractor.CurrentSignalVersion, segment.Signal.SignalVersion);
            Assert.True(segment.Segment.SurfaceCount > 0);
            Assert.True(segment.Signal.TopologyStats.TotalIndexCount > 0);
        });
    }

    [Fact]
    public void Build_SyntheticCorpus_SplitsZeroCk24ConnectivityAndFlagsLow16Reuse()
    {
        Pm4ResearchDocument document = CreateSyntheticDocument();

        IReadOnlyList<Pm4BuiltObjectSegment> segments = Pm4ObjectSegmentBuilder.Build(document, 30, 48);

        Assert.Equal(3, segments.Count);

        List<Pm4BuiltObjectSegment> zeroSegments = segments.Where(static segment => segment.Segment.Ck24 == 0u).ToList();
        Assert.Single(zeroSegments);
        Pm4BuiltObjectSegment zeroSegment = zeroSegments[0];
        Assert.True(zeroSegment.Segment.ConfidenceFlags.HasFlag(Pm4SegmentConfidenceFlags.ZeroCk24Seed));
        Assert.True(zeroSegment.Segment.ConfidenceFlags.HasFlag(Pm4SegmentConfidenceFlags.HasUnlinkedSurfaces));
        Assert.True(zeroSegment.Segment.ConfidenceFlags.HasFlag(Pm4SegmentConfidenceFlags.MissingPositionRefs));
        Assert.Equal(2, zeroSegment.Segment.SurfaceCount);

        List<Pm4BuiltObjectSegment> reusedLow16Segments = segments
            .Where(static segment => segment.Segment.Ck24 != 0u)
            .Where(static segment => segment.Segment.ConfidenceFlags.HasFlag(Pm4SegmentConfidenceFlags.ReusedLow16ObjectId))
            .ToList();
        Assert.Equal(2, reusedLow16Segments.Count);
        Assert.Contains(reusedLow16Segments, static segment => segment.Segment.Ck24 == 0x42ABCDu);
        Assert.Contains(reusedLow16Segments, static segment => segment.Segment.Ck24 == 0x43ABCDu);
    }

    private static Pm4ResearchDocument CreateSyntheticDocument()
    {
        Vector3[] vertices =
        [
            new(0f, 0f, 0f),
            new(1f, 0f, 0f),
            new(0f, 1f, 0f),
            new(1f, 1f, 0f),
            new(10f, 0f, 0f),
            new(11f, 0f, 0f),
            new(10f, 1f, 0f),
            new(20f, 0f, 0f),
            new(21f, 0f, 0f),
            new(20f, 1f, 0f),
            new(30f, 0f, 0f),
            new(31f, 0f, 0f),
            new(30f, 1f, 0f),
            new(40f, 0f, 0f),
            new(41f, 0f, 0f),
            new(40f, 1f, 0f),
        ];

        uint[] indices =
        [
            0, 1, 2,
            1, 3, 2,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12,
            13, 14, 15,
        ];

        List<Pm4MsurEntry> surfaces =
        [
            CreateSurface(groupKey: 3, attributeMask: 0x10, height: 1f, msviFirstIndex: 0, mscnRefIndex: 0, ck24: 0x42ABCDu),
            CreateSurface(groupKey: 3, attributeMask: 0x10, height: 2f, msviFirstIndex: 3, mscnRefIndex: 0, ck24: 0x42ABCDu),
            CreateSurface(groupKey: 4, attributeMask: 0x12, height: 3f, msviFirstIndex: 6, mscnRefIndex: 1, ck24: 0x43ABCDu),
            CreateSurface(groupKey: 5, attributeMask: 0x03, height: 4f, msviFirstIndex: 9, mscnRefIndex: 2, ck24: 0u),
            CreateSurface(groupKey: 5, attributeMask: 0x03, height: 5f, msviFirstIndex: 12, mscnRefIndex: 2, ck24: 0u),
        ];

        List<Pm4MslkEntry> links =
        [
            new(0x12, 0, 0, 1779u, 0, 0, 0, 0, 0),
            new(0x12, 0, 0, 1779u, 0, 0, 0, 1, 0),
            new(0x03, 0, 0, 1880u, 0, 0, 0, 2, 0),
        ];

        List<Pm4MprlEntry> refs =
        [
            new(0, 1, 0, 0, new Vector3(0.5f, 0.5f, 0f), 0, 0),
            new(0, 2, 0, 0, new Vector3(0.5f, 0.6f, 0f), 0, 0),
            new(0, 3, 0, 0, new Vector3(10.5f, 0.5f, 0f), 0, 0),
            new(0, -1, 0, 0, new Vector3(20.5f, 0.5f, 0f), 0, 0),
            new(0, -1, 0, 0, new Vector3(30.5f, 0.5f, 0f), 0, 0),
        ];

        Pm4KnownChunkSet chunks = new(
            new Pm4MshdHeader(0, 3262, 0, 0, 0, 0, 0, 0),
            links,
            Array.Empty<Vector3>(),
            Array.Empty<uint>(),
            vertices,
            indices,
            surfaces,
            Array.Empty<Vector3>(),
            refs,
            Array.Empty<Pm4MprrEntry>(),
            null,
            Array.Empty<Pm4MdbiEntry>(),
            Array.Empty<Pm4MdbfEntry>(),
            Array.Empty<Pm4MdosEntry>(),
            Array.Empty<Pm4MdsfEntry>());

        return new Pm4ResearchDocument(
            "synthetic_30_48.pm4",
            0,
            Array.Empty<Pm4ChunkRecord>(),
            chunks,
            Array.Empty<string>());
    }

    private static Pm4MsurEntry CreateSurface(byte groupKey, byte attributeMask, float height, uint msviFirstIndex, uint mscnRefIndex, uint ck24)
    {
        uint packedParams = ck24 << 8;
        return new Pm4MsurEntry(groupKey, 3, attributeMask, 0, Vector3.UnitZ, height, msviFirstIndex, mscnRefIndex, packedParams);
    }
}
