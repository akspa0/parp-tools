using WowViewer.Core.PM4.Models;

namespace WowViewer.Core.PM4.Services;

public static class Pm4MshdGroupingService
{
    public static Pm4MshdGroupingInfo Describe(Pm4MshdHeader? header)
    {
        if (header is null)
            return Pm4MshdGroupingInfo.Empty;

        return new Pm4MshdGroupingInfo(
            header.Field00,
            header.RegionId,
            header.Field08,
            header.IsEmptyStubRegion);
    }
}
