using System.Numerics;

namespace WowViewer.Core.Wmo;

public sealed class WmoPortalVertexDetail
{
    public WmoPortalVertexDetail(int portalVertexIndex, Vector3 position)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(portalVertexIndex);

        PortalVertexIndex = portalVertexIndex;
        Position = position;
    }

    public int PortalVertexIndex { get; }

    public Vector3 Position { get; }
}