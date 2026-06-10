using System.Numerics;
using WowViewer.Core.PM4.Models;

namespace WowViewer.Core.PM4.Matching;

public static class Pm4SegmentSignalExtractor
{
    public const string CurrentSignalVersion = "pm4-segment-signal-v2";

    public static Pm4SegmentSignalRecord Extract(
        Pm4ObjectSegment segment,
        Pm4CorrelationObjectState correlationState,
        Pm4LinkedPositionRefSummary anchorSummary,
        IReadOnlyList<Pm4ObjectSegmentSurface> surfaces,
        IReadOnlyDictionary<byte, Pm4Bounds3> typedBounds)
    {
        float minPlaneDistance = 0f;
        float maxPlaneDistance = 0f;
        float averagePlaneDistance = 0f;
        if (surfaces.Count > 0)
        {
            minPlaneDistance = surfaces.Min(static surface => surface.PlaneDistance);
            maxPlaneDistance = surfaces.Max(static surface => surface.PlaneDistance);
            averagePlaneDistance = surfaces.Average(static surface => surface.PlaneDistance);
        }

        Dictionary<string, int> histogram = new(StringComparer.Ordinal)
        {
            [$"ck24Type:0x{segment.Ck24Type:X2}"] = segment.SurfaceCount,
        };

        foreach (IGrouping<byte, Pm4ObjectSegmentSurface> group in surfaces.GroupBy(static surface => surface.GroupKey).OrderBy(static group => group.Key))
            histogram[$"groupKey:0x{group.Key:X2}"] = group.Count();

        foreach (IGrouping<byte, Pm4ObjectSegmentSurface> group in surfaces.GroupBy(static surface => surface.AttributeMask).OrderBy(static group => group.Key))
            histogram[$"attributeMask:0x{group.Key:X2}"] = group.Count();

        // TypeFlags distribution — which surface roles this object has
        foreach (KeyValuePair<byte, Pm4Bounds3> kv in typedBounds)
        {
            string label = kv.Key switch
            {
                0x03 => "typeFlags:m2-top",
                0x10 => "typeFlags:interior-floor",
                0x12 => "typeFlags:exterior-solid",
                _ => $"typeFlags:0x{kv.Key:X2}",
            };
            histogram[label] = histogram.TryGetValue(label, out int existing) ? existing + 1 : 1;
        }

        return new Pm4SegmentSignalRecord(
            segment.SegmentId,
            new Pm4Bounds3(correlationState.BoundsMin, correlationState.BoundsMax),
            correlationState.FootprintHull.ToList(),
            new Pm4SegmentHeightStats(minPlaneDistance, maxPlaneDistance, averagePlaneDistance),
            histogram,
            new Pm4SegmentTopologyStats(
                segment.SurfaceCount,
                segment.TotalIndexCount,
                anchorSummary.TotalCount,
                anchorSummary.NormalCount),
            new Pm4SegmentAnchorSignals(
                anchorSummary.TotalCount,
                anchorSummary.NormalCount,
                anchorSummary.TerminatorCount,
                anchorSummary.FloorMin,
                anchorSummary.FloorMax,
                anchorSummary.HasNormalHeadings ? anchorSummary.HeadingMinDegrees : null,
                anchorSummary.HasNormalHeadings ? anchorSummary.HeadingMaxDegrees : null,
                anchorSummary.HasNormalHeadings ? anchorSummary.HeadingMeanDegrees : null),
            CurrentSignalVersion,
            null,
            typedBounds);
    }
}
