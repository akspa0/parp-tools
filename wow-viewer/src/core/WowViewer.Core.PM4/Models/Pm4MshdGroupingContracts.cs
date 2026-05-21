namespace WowViewer.Core.PM4.Models;

public sealed record Pm4MshdGroupingInfo(
    uint Field00,
    uint RegionId,
    uint Field08,
    bool IsEmptyStubRegion)
{
    public static Pm4MshdGroupingInfo Empty { get; } = new(0, 0, 0, false);
}
