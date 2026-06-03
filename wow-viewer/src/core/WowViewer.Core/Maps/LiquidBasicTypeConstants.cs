namespace WowViewer.Core.Maps;

/// <summary>
/// Constants for the <see cref="TerrainTileTensorPack.LiquidBasicType257"/>
/// canonical resolved liquid type field. Values are the
/// <see cref="AdtLiquidBasicType"/> enum encoded as a single byte (0..3),
/// with <see cref="NoLiquid"/> as a sentinel for "no liquid present at this
/// vertex" (0xFF = 255).
/// </summary>
public static class LiquidBasicTypeConstants
{
    public const byte NoLiquid = 0xFF;
    public const int MaxBasicType = 3;
}
