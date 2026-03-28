using System.Buffers.Binary;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoGroupLiquidSummaryReader
{
    private const int HeaderSize = 30;
    private const int VertexStride = 8;

    public static WmoGroupLiquidSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoGroupLiquidSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        (uint? version, byte[] mogp) = WmoGroupReaderCommon.ReadGroupPayload(stream, sourcePath);
        return ReadMogpPayload(mogp, sourcePath, version);
    }

    internal static WmoGroupLiquidSummary ReadMogpPayload(byte[] mogp, string sourcePath, uint? version)
    {
        ArgumentNullException.ThrowIfNull(mogp);

        uint groupFlags = BinaryPrimitives.ReadUInt32LittleEndian(mogp.AsSpan(0x08, 4));
        int headerSizeBytes = WmoGroupReaderCommon.FindHeaderSize(mogp);
        byte[]? liquidPayload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogp, headerSizeBytes, WmoChunkIds.Mliq);
        if (liquidPayload is null)
            throw new InvalidDataException("WMO group liquid summary requires an MLIQ subchunk.");

        if (liquidPayload.Length < HeaderSize)
            throw new InvalidDataException($"MLIQ payload is too short ({liquidPayload.Length} bytes). Expected at least {HeaderSize} bytes.");

        int xVertexCount = BinaryPrimitives.ReadInt32LittleEndian(liquidPayload.AsSpan(0, 4));
        int yVertexCount = BinaryPrimitives.ReadInt32LittleEndian(liquidPayload.AsSpan(4, 4));
        int xTileCount = BinaryPrimitives.ReadInt32LittleEndian(liquidPayload.AsSpan(8, 4));
        int yTileCount = BinaryPrimitives.ReadInt32LittleEndian(liquidPayload.AsSpan(12, 4));
        if (xVertexCount <= 0 || yVertexCount <= 0)
            throw new InvalidDataException($"MLIQ vertex dimensions must be positive, but found {xVertexCount}x{yVertexCount}.");

        int vertexCount = checked(xVertexCount * yVertexCount);
        int vertexBytes = checked(vertexCount * VertexStride);
        if (liquidPayload.Length < HeaderSize + vertexBytes)
            throw new InvalidDataException($"MLIQ payload is too short for {vertexCount} vertices.");

        float minHeight = float.MaxValue;
        float maxHeight = float.MinValue;
        int vertexOffset = HeaderSize;
        for (int index = 0; index < vertexCount; index++)
        {
            float height = BitConverter.ToSingle(liquidPayload, vertexOffset + 4);
            minHeight = Math.Min(minHeight, height);
            maxHeight = Math.Max(maxHeight, height);
            vertexOffset += VertexStride;
        }

        int tileCount = checked(Math.Max(0, xTileCount) * Math.Max(0, yTileCount));
        int tileFlagByteCount = Math.Min(Math.Max(0, liquidPayload.Length - HeaderSize - vertexBytes), tileCount);
        int visibleTileCount = 0;
        int firstVisibleNibble = -1;
        for (int index = 0; index < tileFlagByteCount; index++)
        {
            int nibble = liquidPayload[HeaderSize + vertexBytes + index] & 0x0F;
            if (nibble == 0x0F)
                continue;

            visibleTileCount++;
            if (firstVisibleNibble < 0)
                firstVisibleNibble = nibble;
        }

        return new WmoGroupLiquidSummary(
            sourcePath,
            version,
            liquidPayload.Length,
            xVertexCount,
            yVertexCount,
            xTileCount,
            yTileCount,
            corner: WmoGroupReaderCommon.ReadVector3(liquidPayload.AsSpan(16, 12)),
            materialId: BinaryPrimitives.ReadUInt16LittleEndian(liquidPayload.AsSpan(28, 2)),
            heightCount: vertexCount,
            minHeight,
            maxHeight,
            tileFlagByteCount,
            visibleTileCount,
            liquidType: InferLiquidType(firstVisibleNibble, groupFlags));
    }

    private static WmoLiquidBasicType InferLiquidType(int firstVisibleNibble, uint groupFlags)
    {
        WmoLiquidBasicType type = firstVisibleNibble switch
        {
            2 or 6 => WmoLiquidBasicType.Magma,
            3 or 7 => WmoLiquidBasicType.Slime,
            _ => WmoLiquidBasicType.Water,
        };

        bool isOcean = (groupFlags & 0x80000) != 0;
        if (isOcean && type == WmoLiquidBasicType.Water)
            return WmoLiquidBasicType.Ocean;

        return type;
    }
}
