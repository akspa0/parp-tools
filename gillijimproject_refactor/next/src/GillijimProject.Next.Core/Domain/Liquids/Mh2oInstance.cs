using System;

namespace GillijimProject.Next.Core.Domain.Liquids;

/// <summary>
/// MH2O liquid instance rectangle and vertex data.
/// Width/Height are in [1..8], offsets in [0..7], and XOffset+Width/YOffset+Height <= 8.
/// Vertex maps (when present) are sized (Width+1)*(Height+1).
/// </summary>
public sealed class Mh2oInstance
{
    /// <summary>LK LiquidType entry id (can be overridden via mapping).</summary>
    public ushort LiquidTypeId { get; init; }

    /// <summary>Liquid vertex format (case 0..3).</summary>
    public LiquidVertexFormat Lvf { get; init; }

    /// <summary>Minimum height level of this instance.</summary>
    public float MinHeightLevel { get; init; }

    /// <summary>Maximum height level of this instance.</summary>
    public float MaxHeightLevel { get; init; }

    /// <summary>Tile-space X offset (0..7).</summary>
    public byte XOffset { get; init; }

    /// <summary>Tile-space Y offset (0..7).</summary>
    public byte YOffset { get; init; }

    /// <summary>Width in tiles (1..8).</summary>
    public byte Width { get; init; }

    /// <summary>Height in tiles (1..8).</summary>
    public byte Height { get; init; }

    /// <summary>
    /// Optional exists bitmap for (Width*Height) tiles; null means all tiles exist.
    /// Packed row-major 1 bit per tile; length = ceil(W*H/8).
    /// </summary>
    public byte[]? ExistsBitmap { get; init; }

    /// <summary>Optional height map (present in cases with height data); length = (W+1)*(H+1).</summary>
    public float[]? HeightMap { get; init; }

    /// <summary>Optional depth map (present in cases with depth data); length = (W+1)*(H+1).</summary>
    public byte[]? DepthMap { get; init; }

    /// <summary>Number of vertices in the per-vertex maps: (W+1)*(H+1).</summary>
    public int VertexCount => (Width + 1) * (Height + 1);

    /// <summary>Validate dimensions and constraints for this instance.</summary>
    public void Validate()
    {
        if (Width is < 1 or > 8) throw new ArgumentOutOfRangeException(nameof(Width));
        if (Height is < 1 or > 8) throw new ArgumentOutOfRangeException(nameof(Height));
        if (XOffset > 7) throw new ArgumentOutOfRangeException(nameof(XOffset));
        if (YOffset > 7) throw new ArgumentOutOfRangeException(nameof(YOffset));
        if (XOffset + Width > 8) throw new ArgumentOutOfRangeException(null, "XOffset+Width must be <= 8");
        if (YOffset + Height > 8) throw new ArgumentOutOfRangeException(null, "YOffset+Height must be <= 8");

        int v = VertexCount;
        switch (Lvf)
        {
            case LiquidVertexFormat.HeightDepth:
                if (HeightMap is null || DepthMap is null)
                    throw new InvalidOperationException("HeightDepth requires both HeightMap and DepthMap");
                if (HeightMap.Length != v) throw new ArgumentException("HeightMap length mismatch", nameof(HeightMap));
                if (DepthMap.Length != v) throw new ArgumentException("DepthMap length mismatch", nameof(DepthMap));
                break;
            case LiquidVertexFormat.DepthOnly:
                if (DepthMap is null) throw new InvalidOperationException("DepthOnly requires DepthMap");
                if (DepthMap.Length != v) throw new ArgumentException("DepthMap length mismatch", nameof(DepthMap));
                if (HeightMap is not null && HeightMap.Length != v)
                    throw new ArgumentException("HeightMap length mismatch", nameof(HeightMap));
                break;
            case LiquidVertexFormat.HeightUv:
            case LiquidVertexFormat.HeightUvDepth:
                // TODO(PORT): UV handling not implemented initially.
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(Lvf));
        }

        if (ExistsBitmap is not null)
        {
            int expectedBits = Width * Height;
            int expectedBytes = (expectedBits + 7) / 8;
            if (ExistsBitmap.Length != expectedBytes)
                throw new ArgumentException("ExistsBitmap length mismatch", nameof(ExistsBitmap));
        }
    }
}
