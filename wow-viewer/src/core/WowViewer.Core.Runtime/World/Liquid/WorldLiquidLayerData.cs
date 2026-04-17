using WowViewer.Core.Maps;

namespace WowViewer.Core.Runtime.World.Liquid;

public sealed class WorldLiquidLayerData
{
    public WorldLiquidLayerData(
        ushort liquidTypeId,
        AdtLiquidBasicType basicType,
        AdtLiquidVertexFormat vertexFormat,
        float minHeight,
        float maxHeight,
        int xOffset,
        int yOffset,
        int width,
        int height,
        int visibleTileCount,
        bool hasDepthData,
        bool hasHeightData,
        bool hasUvData)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(xOffset);
        ArgumentOutOfRangeException.ThrowIfNegative(yOffset);
        ArgumentOutOfRangeException.ThrowIfNegative(width);
        ArgumentOutOfRangeException.ThrowIfNegative(height);
        ArgumentOutOfRangeException.ThrowIfNegative(visibleTileCount);

        LiquidTypeId = liquidTypeId;
        BasicType = basicType;
        VertexFormat = vertexFormat;
        MinHeight = minHeight;
        MaxHeight = maxHeight;
        XOffset = xOffset;
        YOffset = yOffset;
        Width = width;
        Height = height;
        VisibleTileCount = visibleTileCount;
        HasDepthData = hasDepthData;
        HasHeightData = hasHeightData;
        HasUvData = hasUvData;
    }

    public ushort LiquidTypeId { get; }

    public AdtLiquidBasicType BasicType { get; }

    public AdtLiquidVertexFormat VertexFormat { get; }

    public float MinHeight { get; }

    public float MaxHeight { get; }

    public int XOffset { get; }

    public int YOffset { get; }

    public int Width { get; }

    public int Height { get; }

    public int VisibleTileCount { get; }

    public bool HasDepthData { get; }

    public bool HasHeightData { get; }

    public bool HasUvData { get; }
}