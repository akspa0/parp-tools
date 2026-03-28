using System.Numerics;

namespace WowViewer.Core.Mdx;

public sealed class MdxGeosetAnimationSummary
{
    public MdxGeosetAnimationSummary(
        int index,
        uint geosetId,
        float staticAlpha,
        Vector3 staticColor,
        uint flags,
        MdxGeosetAnimationTrackSummary? alphaTrack,
        MdxGeosetAnimationTrackSummary? colorTrack)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);

        Index = index;
        GeosetId = geosetId;
        StaticAlpha = staticAlpha;
        StaticColor = staticColor;
        Flags = flags;
        AlphaTrack = alphaTrack;
        ColorTrack = colorTrack;
    }

    public int Index { get; }

    public uint GeosetId { get; }

    public float StaticAlpha { get; }

    public Vector3 StaticColor { get; }

    public uint Flags { get; }

    public bool UsesStaticColor => (Flags & 0x1u) != 0;

    public MdxGeosetAnimationTrackSummary? AlphaTrack { get; }

    public MdxGeosetAnimationTrackSummary? ColorTrack { get; }
}