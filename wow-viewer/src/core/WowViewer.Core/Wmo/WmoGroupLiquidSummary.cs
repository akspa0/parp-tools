using System.Numerics;

namespace WowViewer.Core.Wmo;

public sealed class WmoGroupLiquidSummary
{
    public WmoGroupLiquidSummary(
        string sourcePath,
        uint? version,
        int payloadSizeBytes,
        int xVertexCount,
        int yVertexCount,
        int xTileCount,
        int yTileCount,
        Vector3 corner,
        int materialId,
        int heightCount,
        float minHeight,
        float maxHeight,
        int tileFlagByteCount,
        int visibleTileCount,
        WmoLiquidBasicType liquidType)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(payloadSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(xVertexCount);
        ArgumentOutOfRangeException.ThrowIfNegative(yVertexCount);
        ArgumentOutOfRangeException.ThrowIfNegative(xTileCount);
        ArgumentOutOfRangeException.ThrowIfNegative(yTileCount);
        ArgumentOutOfRangeException.ThrowIfNegative(materialId);
        ArgumentOutOfRangeException.ThrowIfNegative(heightCount);
        ArgumentOutOfRangeException.ThrowIfNegative(tileFlagByteCount);
        ArgumentOutOfRangeException.ThrowIfNegative(visibleTileCount);

        SourcePath = sourcePath;
        Version = version;
        PayloadSizeBytes = payloadSizeBytes;
        XVertexCount = xVertexCount;
        YVertexCount = yVertexCount;
        XTileCount = xTileCount;
        YTileCount = yTileCount;
        Corner = corner;
        MaterialId = materialId;
        HeightCount = heightCount;
        MinHeight = minHeight;
        MaxHeight = maxHeight;
        TileFlagByteCount = tileFlagByteCount;
        VisibleTileCount = visibleTileCount;
        LiquidType = liquidType;
    }

    public string SourcePath { get; }

    public uint? Version { get; }

    public int PayloadSizeBytes { get; }

    public int XVertexCount { get; }

    public int YVertexCount { get; }

    public int XTileCount { get; }

    public int YTileCount { get; }

    public int TileCount => XTileCount * YTileCount;

    public Vector3 Corner { get; }

    public int MaterialId { get; }

    public int HeightCount { get; }

    public float MinHeight { get; }

    public float MaxHeight { get; }

    public int TileFlagByteCount { get; }

    public bool HasCompleteTileFlags => TileFlagByteCount >= TileCount;

    public int VisibleTileCount { get; }

    public WmoLiquidBasicType LiquidType { get; }
}
