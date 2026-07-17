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
///   <item><description>MCLQ presence/type mask (pre-WotLK) — later ADT
///     builders store raw <c>MclqLiquidType</c> enum values which this helper
///     degrades to <see cref="AdtLiquidBasicType.Water"/> on out-of-range
///     values (per the MCLQ encoding table in spec 040 §3). Alpha already
///     carries resolved basic types and normalizes them in
///     <c>AlphaTensorPackBuilder</c>.</description></item>
///   <item><description>MCNK flag fallback — per-chunk
///     <see cref="McnkFlagDecoder.Decode"/> applied to all 17×17 vertices
///     in the chunk. Used when neither MH2O nor MCLQ is present but the
///     MCNK header marks the chunk as liquid.</description></item>
///   <item><description>WL* fallback — a terrain-gated recovered surface uses
///     its parsed header/container family type only where neither explicit
///     MH2O nor MCLQ coverage owns the destination pixel.</description></item>
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

    /// <summary>
    /// Overlays WL* type evidence at the same per-pixel priority used by the
    /// unified liquid-height mask: MH2O, then MCLQ, then recovered WL*.
    /// MCNK flags are classification-only and intentionally do not suppress a
    /// visible, terrain-gated WL* surface.
    /// </summary>
    public static byte[,]? OverlayWlFallbackTypes(
        byte[,]? resolvedTypes,
        float[,]? wlMask,
        byte[,]? wlTypes,
        float[,]? mh2oSurfaceHeight,
        bool[,]? mh2oPresence,
        float[,]? mclqSurfaceHeight,
        bool[,]? mclqPresence)
    {
        if (wlMask is null || wlTypes is null)
            return resolvedTypes;

        int height = wlMask.GetLength(0);
        int width = wlMask.GetLength(1);
        if (height == 0 || width == 0
            || wlTypes.GetLength(0) != height
            || wlTypes.GetLength(1) != width)
        {
            throw new ArgumentException("WL mask and basic-type grids must have matching non-empty dimensions.");
        }

        if (resolvedTypes is not null
            && (resolvedTypes.GetLength(0) != height || resolvedTypes.GetLength(1) != width))
        {
            throw new ArgumentException("Resolved liquid type grid must match the WL grid.", nameof(resolvedTypes));
        }

        byte[,] result = resolvedTypes is null
            ? CreateNoLiquidGrid(height, width)
            : (byte[,])resolvedTypes.Clone();

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                byte wlType = wlTypes[y, x];
                if (!(wlMask[y, x] > 0f)
                    || wlType > LiquidBasicTypeConstants.MaxBasicType
                    || HasMh2oCoverage(mh2oSurfaceHeight, mh2oPresence, x, y, width, height)
                    || HasMclqCoverage(mclqSurfaceHeight, mclqPresence, x, y, width, height))
                {
                    continue;
                }

                result[y, x] = wlType;
            }
        }

        return result;
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

    private static byte[,] CreateNoLiquidGrid(int height, int width)
    {
        byte[,] result = new byte[height, width];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
                result[y, x] = LiquidBasicTypeConstants.NoLiquid;
        }

        return result;
    }

    private static bool HasMh2oCoverage(
        float[,]? surfaceHeight,
        bool[,]? presence,
        int x,
        int y,
        int targetWidth,
        int targetHeight)
    {
        if (surfaceHeight is null || presence is null
            || surfaceHeight.GetLength(0) != presence.GetLength(0)
            || surfaceHeight.GetLength(1) != presence.GetLength(1))
        {
            return false;
        }

        int sourceY = ScaleCoordinate(y, targetHeight, presence.GetLength(0));
        int sourceX = ScaleCoordinate(x, targetWidth, presence.GetLength(1));
        return presence[sourceY, sourceX];
    }

    private static bool HasMclqCoverage(
        float[,]? surfaceHeight,
        bool[,]? presence,
        int x,
        int y,
        int targetWidth,
        int targetHeight)
    {
        if (surfaceHeight is null || presence is null
            || surfaceHeight.GetLength(0) != presence.GetLength(0)
            || surfaceHeight.GetLength(1) != presence.GetLength(1)
            || presence.GetLength(0) == 0
            || presence.GetLength(1) == 0)
        {
            return false;
        }

        int sourceHeight = presence.GetLength(0);
        int sourceWidth = presence.GetLength(1);
        if (sourceHeight == 1 || sourceWidth == 1)
            return presence[Math.Min(y, sourceHeight - 1), Math.Min(x, sourceWidth - 1)];

        float sourceX = x * (sourceWidth - 1f) / Math.Max(targetWidth - 1f, 1f);
        float sourceY = y * (sourceHeight - 1f) / Math.Max(targetHeight - 1f, 1f);
        int ix = Math.Clamp((int)sourceX, 0, sourceWidth - 2);
        int iy = Math.Clamp((int)sourceY, 0, sourceHeight - 2);
        return presence[iy, ix]
            || presence[iy, ix + 1]
            || presence[iy + 1, ix]
            || presence[iy + 1, ix + 1];
    }

    private static int ScaleCoordinate(int coordinate, int sourceSize, int targetSize)
    {
        if (sourceSize <= 1 || targetSize <= 1)
            return 0;

        return Math.Clamp((int)MathF.Round(coordinate * (targetSize - 1f) / (sourceSize - 1f)), 0, targetSize - 1);
    }
}
