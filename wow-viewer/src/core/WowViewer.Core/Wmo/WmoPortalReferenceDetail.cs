namespace WowViewer.Core.Wmo;

public sealed class WmoPortalReferenceDetail
{
    public WmoPortalReferenceDetail(int referenceIndex, int portalIndex, int groupIndex, short side)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(referenceIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(portalIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(groupIndex);

        ReferenceIndex = referenceIndex;
        PortalIndex = portalIndex;
        GroupIndex = groupIndex;
        Side = side;
    }

    public int ReferenceIndex { get; }

    public int PortalIndex { get; }

    public int GroupIndex { get; }

    public short Side { get; }
}