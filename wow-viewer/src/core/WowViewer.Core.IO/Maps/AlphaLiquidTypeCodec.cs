using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

internal static class AlphaLiquidTypeCodec
{
    private const uint VisibleLiquidMask = 0x3Cu;

    public static uint GetWriterChunkFlags(AdtLiquidBasicType _) => VisibleLiquidMask;

    public static byte GetWriterTileTypeNibble(AdtLiquidBasicType basicType)
    {
        // MCLQ's raw lower nibble is not the AdtLiquidBasicType ordinal.
        return basicType switch
        {
            AdtLiquidBasicType.Ocean => 0x01,
            AdtLiquidBasicType.Magma => 0x06,
            AdtLiquidBasicType.Slime => 0x03,
            _ => 0x04, // River, rendered as water.
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
        if (tileFlags is { Length: >= 64 })
        {
            for (int index = 0; index < 64; index++)
            {
                byte tileType = (byte)(tileFlags[index] & 0x0F);
                if (tileType != 0x0F)
                    return McnkFlagDecoder.DecodeWithMclqTileNibble(mcnkFlags, tileType);
            }
        }

        return McnkFlagDecoder.Decode(mcnkFlags);
    }
}
