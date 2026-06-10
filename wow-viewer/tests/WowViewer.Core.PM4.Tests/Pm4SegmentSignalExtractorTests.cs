using System.Numerics;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.PM4.Models;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4SegmentSignalExtractorTests
{
    [Fact]
    public void Extract_PopulatesSignalContractFromSegmentState()
    {
        Pm4ObjectSegment segment = new(
            "pm4seg-test",
            0x42ABCDu,
            0x42,
            0xABCD,
            ["30_48"],
            [3262u],
            2,
            6,
            [1779u],
            1779u,
            Pm4SegmentConfidenceFlags.None);

        Pm4CorrelationObjectState state = new(
            30,
            48,
            new Pm4ObjectGroupKey(30, 48, 0x42ABCDu),
            new Pm4CorrelationObjectDescriptor(0x42ABCDu, 0x42, 0, 1779u, 2, 3, 3, 0x10, 0u, 1.5f),
            new Vector3(0f, 0f, 0f),
            new Vector3(4f, 8f, 2f),
            new Vector3(2f, 4f, 1f),
            [new Vector2(0f, 0f), new Vector2(4f, 0f), new Vector2(4f, 8f), new Vector2(0f, 8f)],
            32f);

        Pm4LinkedPositionRefSummary anchorSummary = new(
            3,
            2,
            1,
            -2,
            5,
            10f,
            40f,
            25f);

        IReadOnlyList<Pm4ObjectSegmentSurface> surfaces =
        [
            new Pm4ObjectSegmentSurface(0, 3, 0x10, 3, 1f, 0, 0, 0x42ABCD00u, 0x42ABCDu, 0x42, 0xABCD, Vector3.UnitZ),
            new Pm4ObjectSegmentSurface(1, 4, 0x12, 3, 2f, 3, 1, 0x42ABCD00u, 0x42ABCDu, 0x42, 0xABCD, Vector3.UnitZ),
        ];

        Pm4SegmentSignalRecord signal = Pm4SegmentSignalExtractor.Extract(segment, state, anchorSummary, surfaces, new Dictionary<byte, Pm4Bounds3>());

        Assert.Equal(Pm4SegmentSignalExtractor.CurrentSignalVersion, signal.SignalVersion);
        Assert.NotNull(signal.Bounds);
        Assert.Equal(new Vector3(0f, 0f, 0f), signal.Bounds!.Min);
        Assert.Equal(new Vector3(4f, 8f, 2f), signal.Bounds.Max);
        Assert.Equal(1f, signal.HeightStats.MinimumPlaneDistance);
        Assert.Equal(2f, signal.HeightStats.MaximumPlaneDistance);
        Assert.Equal(1.5f, signal.HeightStats.AveragePlaneDistance);
        Assert.Equal(2, signal.SurfaceFamilyHistogram["ck24Type:0x42"]);
        Assert.Equal(1, signal.SurfaceFamilyHistogram["groupKey:0x03"]);
        Assert.Equal(1, signal.SurfaceFamilyHistogram["groupKey:0x04"]);
        Assert.Equal(1, signal.SurfaceFamilyHistogram["attributeMask:0x10"]);
        Assert.Equal(1, signal.SurfaceFamilyHistogram["attributeMask:0x12"]);
        Assert.Equal(3, signal.AnchorSignals.LinkedPositionRefCount);
        Assert.Equal(2, signal.AnchorSignals.NormalHeadingCount);
        Assert.Equal(1, signal.AnchorSignals.TerminatorCount);
        Assert.Equal(-2, signal.AnchorSignals.FloorMinimum);
        Assert.Equal(5, signal.AnchorSignals.FloorMaximum);
        Assert.Equal(25f, signal.AnchorSignals.HeadingMeanDegrees);
    }
}
