using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

internal static class AlphaLiquidTypeCodec
{
    private const uint VisibleLiquidMask = 0x3Cu;

    public static uint GetWriterChunkFlags(AdtLiquidBasicType _) => VisibleLiquidMask;

    public static byte GetWriterTileTypeNibble(AdtLiquidBasicType basicType)
    {
        return basicType switch
        {
            AdtLiquidBasicType.Ocean => 0x02,
            AdtLiquidBasicType.Magma => 0x03,
            AdtLiquidBasicType.Slime => 0x04,
            _ => 0x01,
        };
    }

    public static int ClassifyCoarseType(byte[]? tileFlags, uint mcnkFlags)
    {
        return ResolveBasicType(tileFlags, mcnkFlags) switch
        {
            AdtLiquidBasicType.Magma => 2,
            AdtLiquidBasicType.Slime => 3,
            _ => 1,
        };
    }

    public static AdtLiquidBasicType ResolveBasicType(byte[]? tileFlags, uint mcnkFlags)
    {
        byte tileType = GetVisibleTileTypeNibble(tileFlags);
        if (tileType != 0)
        {
            return tileType switch
            {
                0x02 => AdtLiquidBasicType.Ocean,
                0x03 => AdtLiquidBasicType.Magma,
                0x04 => AdtLiquidBasicType.Slime,
                _ => AdtLiquidBasicType.Water,
            };
        }

        if ((mcnkFlags & 0x20u) != 0)
            return AdtLiquidBasicType.Slime;

        if ((mcnkFlags & 0x10u) != 0)
            return AdtLiquidBasicType.Magma;

        if ((mcnkFlags & 0x08u) != 0)
            return AdtLiquidBasicType.Ocean;

        return AdtLiquidBasicType.Water;
    }

    private static byte GetVisibleTileTypeNibble(byte[]? tileFlags)
    {
        if (tileFlags is not { Length: >= 64 })
            return 0;

        for (int index = 0; index < 64; index++)
        {
            byte lowNibble = (byte)(tileFlags[index] & 0x0F);
            if (lowNibble != 0x0F)
                return lowNibble;
        }

        return 0;
    }
}