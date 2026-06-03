using WowViewer.Core.IO.Liquids;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Builds the canonical <see cref="TerrainTileTensorPack.LiquidBasicType257"/>
/// field at harvest time. The viewer MUST consume this resolved type at render
/// time and MUST NOT re-resolve liquid type from MCNK flags or MH2O
/// LiquidTypeId on the fly.
/// </summary>
/// <remarks>
/// <para>Resolution priority (matching the unified-liquid-mask priority in
/// <c>BuildUnifiedLiquid</c>):</para>
/// <list type="number">
///   <item><description>MH2O presence/type mask (LK WotLK+) — already
///     resolved by <c>AdtLiquidReader.MapLiquidTypeId</c> via the
///     <c>DbcLiquidTypeTable</c>.</description></item>
///   <item><description>MCLQ presence/type mask (pre-WotLK) — the Alpha
///     builder stores <see cref="AdtLiquidBasicType"/> here; the LK builder
///     stores raw <c>MclqLiquidType</c> enum values which this helper
///     degrades to <see cref="AdtLiquidBasicType.Water"/> on out-of-range
///     values (per the MCLQ encoding table in spec 040 §3).</description></item>
///   <item><description>MCNK flag fallback — per-chunk
///     <see cref="McnkFlagDecoder.Decode"/> applied to all 17×17 vertices
///     in the chunk. Used when neither MH2O nor MCLQ is present but the
///     MCNK header marks the chunk as liquid.</description></item>
/// </list>
/// </remarks>
public static class LiquidBasicTypePackBuilder
{
    private const int TileHeightmapSize = 257;
    private const int TileChunks = 16;
    private const int VerticesPerChunk = 17;

    public static byte[,]? Build(
        bool[,]? mh2oPresence,
        int[,]? mh2oType,
        bool[,]? mclqPresence,
        int[,]? mclqType,
        int[,]? mcnkFlags16)
    {
        if (mh2oPresence is not null)
            return BuildFromMh2o(mh2oPresence, mh2oType);

        if (mclqPresence is not null)
            return BuildFromMclq(mclqPresence, mclqType);

        if (mcnkFlags16 is not null)
            return BuildFromMcnkFlags(mcnkFlags16);

        return null;
    }

    private static byte[,] BuildFromMh2o(bool[,] presence, int[,]? type)
    {
        int h = presence.GetLength(0);
        int w = presence.GetLength(1);
        byte[,] result = new byte[h, w];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                if (!presence[y, x])
                {
                    result[y, x] = LiquidBasicTypeConstants.NoLiquid;
                    continue;
                }

                int raw = type is not null ? type[y, x] : 0;
                result[y, x] = (byte)Math.Clamp(raw, 0, LiquidBasicTypeConstants.MaxBasicType);
            }
        }
        return result;
    }

    private static byte[,] BuildFromMclq(bool[,] presence, int[,]? type)
    {
        int h = presence.GetLength(0);
        int w = presence.GetLength(1);
        byte[,] result = new byte[h, w];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                if (!presence[y, x])
                {
                    result[y, x] = LiquidBasicTypeConstants.NoLiquid;
                    continue;
                }

                int raw = type is not null ? type[y, x] : 0;
                result[y, x] = ConvertMclqLiquidTypeToBasicType(raw);
            }
        }
        return result;
    }

    private static byte ConvertMclqLiquidTypeToBasicType(int raw) => raw switch
    {
        (int)MclqLiquidType.Ocean => (byte)AdtLiquidBasicType.Ocean,
        (int)MclqLiquidType.Slime => (byte)AdtLiquidBasicType.Slime,
        (int)MclqLiquidType.River => (byte)AdtLiquidBasicType.Water,
        (int)MclqLiquidType.Magma => (byte)AdtLiquidBasicType.Magma,
        (int)MclqLiquidType.DontRender => LiquidBasicTypeConstants.NoLiquid,
        _ => (byte)AdtLiquidBasicType.Water,
    };

    private static byte[,] BuildFromMcnkFlags(int[,] mcnkFlags16)
    {
        byte[,] result = new byte[TileHeightmapSize, TileHeightmapSize];
        for (int y = 0; y < TileHeightmapSize; y++)
        {
            int chunkY = Math.Min(y / VerticesPerChunk, TileChunks - 1);
            for (int x = 0; x < TileHeightmapSize; x++)
            {
                int chunkX = Math.Min(x / VerticesPerChunk, TileChunks - 1);
                uint flags = (uint)mcnkFlags16[chunkY, chunkX];
                if (flags == 0)
                {
                    result[y, x] = LiquidBasicTypeConstants.NoLiquid;
                    continue;
                }

                result[y, x] = (byte)McnkFlagDecoder.Decode(flags);
            }
        }
        return result;
    }
}
