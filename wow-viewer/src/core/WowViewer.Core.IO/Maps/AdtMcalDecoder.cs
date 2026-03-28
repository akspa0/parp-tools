using System.Buffers.Binary;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class AdtMcalDecoder
{
    private const uint CompressedAlphaFlag = 0x200u;

    public static AdtMcalDecodedLayer? DecodeLayer(
        byte[] mcalData,
        AdtTextureLayerDescriptor layer,
        AdtTextureLayerDescriptor? nextLayer,
        bool useBigAlpha,
        bool doNotFixAlphaMap,
        AdtMcalDecodeProfile profile)
    {
        ArgumentNullException.ThrowIfNull(mcalData);
        ArgumentNullException.ThrowIfNull(layer);

        int offset = unchecked((int)layer.AlphaOffset);
        if ((uint)offset >= (uint)mcalData.Length)
            return null;

        int? nextOffset = ResolveNextOffset(layer, nextLayer, mcalData.Length);
        bool layerBigAlpha = ResolveLayerBigAlpha(offset, nextOffset, mcalData.Length, useBigAlpha);
        if (!nextOffset.HasValue)
        {
            int expectedSpan = layerBigAlpha ? 4096 : 2048;
            int expectedNext = offset + expectedSpan;
            if (expectedNext > offset && expectedNext <= mcalData.Length)
                nextOffset = expectedNext;
        }

        int maxLength = nextOffset.HasValue
            ? Math.Max(0, nextOffset.Value - offset)
            : Math.Max(0, mcalData.Length - offset);
        if (maxLength <= 0)
            return null;

        uint effectiveFlags = layer.Flags;
        if (profile == AdtMcalDecodeProfile.LichKingStrict && ShouldForceCompressedAlpha(effectiveFlags, maxLength))
            effectiveFlags |= CompressedAlphaFlag;

        return profile == AdtMcalDecodeProfile.Cataclysm400
            ? DecodeCataclysm400(mcalData, layer, offset, effectiveFlags, layerBigAlpha, doNotFixAlphaMap, maxLength)
            : DecodeLichKingStrict(mcalData, layer, offset, effectiveFlags, layerBigAlpha, doNotFixAlphaMap, maxLength);
    }

    internal static IReadOnlyList<AdtTextureLayerDescriptor> ReadTextureLayers(byte[] payload)
    {
        ArgumentNullException.ThrowIfNull(payload);

        const int entrySize = 16;
        int count = payload.Length / entrySize;
        if (count == 0)
            return Array.Empty<AdtTextureLayerDescriptor>();

        AdtTextureLayerDescriptor[] layers = new AdtTextureLayerDescriptor[count];
        for (int index = 0; index < count; index++)
        {
            int offset = index * entrySize;
            layers[index] = new AdtTextureLayerDescriptor(
                index,
                BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(offset, 4)),
                BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(offset + 4, 4)),
                BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(offset + 8, 4)),
                BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(offset + 12, 4)));
        }

        return layers;
    }

    private static AdtMcalDecodedLayer DecodeLichKingStrict(
        byte[] mcalData,
        AdtTextureLayerDescriptor layer,
        int offset,
        uint flags,
        bool useBigAlpha,
        bool doNotFixAlphaMap,
        int maxLength)
    {
        if ((flags & CompressedAlphaFlag) != 0)
        {
            (byte[] alpha, int bytesConsumed) = ReadCompressedAlpha(mcalData, offset, maxLength);
            return new AdtMcalDecodedLayer(layer.Index, layer.TextureId, layer.Flags, offset, bytesConsumed, AdtMcalAlphaEncoding.Compressed, appliedFixup: false, alpha);
        }

        if (useBigAlpha)
        {
            int available = Math.Min(4096, Math.Min(maxLength, mcalData.Length - offset));
            byte[] alpha = new byte[64 * 64];
            if (available > 0)
                Buffer.BlockCopy(mcalData, offset, alpha, 0, available);

            return new AdtMcalDecodedLayer(layer.Index, layer.TextureId, layer.Flags, offset, available, AdtMcalAlphaEncoding.BigAlpha, appliedFixup: false, alpha);
        }

        (byte[] packedAlpha, int packedBytesConsumed) = ReadPackedAlpha(mcalData, offset, maxLength);
        bool appliedFixup = !doNotFixAlphaMap;
        if (appliedFixup)
            ApplyLegacyEdgeFix(packedAlpha);

        return new AdtMcalDecodedLayer(layer.Index, layer.TextureId, layer.Flags, offset, packedBytesConsumed, AdtMcalAlphaEncoding.Packed4Bit, appliedFixup, packedAlpha);
    }

    private static AdtMcalDecodedLayer DecodeCataclysm400(
        byte[] mcalData,
        AdtTextureLayerDescriptor layer,
        int offset,
        uint flags,
        bool useBigAlpha,
        bool doNotFixAlphaMap,
        int maxLength)
    {
        if ((flags & CompressedAlphaFlag) != 0)
        {
            (byte[] alpha, int bytesConsumed) = ReadCompressedAlpha(mcalData, offset, maxLength);
            return new AdtMcalDecodedLayer(layer.Index, layer.TextureId, layer.Flags, offset, bytesConsumed, AdtMcalAlphaEncoding.Compressed, appliedFixup: false, alpha);
        }

        if (!useBigAlpha)
            return DecodeLichKingStrict(mcalData, layer, offset, flags, useBigAlpha: false, doNotFixAlphaMap, maxLength);

        int available = Math.Max(0, Math.Min(maxLength, mcalData.Length - offset));
        if (available <= 0)
            return null!;

        if (!doNotFixAlphaMap && available >= 63 * 63 && available < 64 * 64)
        {
            byte[] alpha = ExpandFixedBigAlpha(mcalData, offset, available);
            return new AdtMcalDecodedLayer(layer.Index, layer.TextureId, layer.Flags, offset, available, AdtMcalAlphaEncoding.BigAlphaFixed, appliedFixup: true, alpha);
        }

        byte[] bigAlpha = new byte[64 * 64];
        Buffer.BlockCopy(mcalData, offset, bigAlpha, 0, Math.Min(bigAlpha.Length, available));
        return new AdtMcalDecodedLayer(layer.Index, layer.TextureId, layer.Flags, offset, available, AdtMcalAlphaEncoding.BigAlpha, appliedFixup: false, bigAlpha);
    }

    private static (byte[] Alpha, int BytesConsumed) ReadCompressedAlpha(byte[] source, int offset, int maxLength)
    {
        byte[] alpha = new byte[64 * 64];
        int sourceEnd = Math.Min(source.Length, offset + maxLength);
        int readPos = offset;
        int writePos = 0;

        while (writePos < alpha.Length && readPos < sourceEnd)
        {
            byte control = source[readPos++];
            bool fill = (control & 0x80) != 0;
            int count = control & 0x7F;
            if (count == 0)
                continue;

            if (fill)
            {
                if (readPos >= sourceEnd)
                    break;

                byte value = source[readPos++];
                int copyCount = Math.Min(count, alpha.Length - writePos);
                Array.Fill(alpha, value, writePos, copyCount);
                writePos += copyCount;
                continue;
            }

            int literalCount = Math.Min(count, Math.Min(alpha.Length - writePos, sourceEnd - readPos));
            if (literalCount <= 0)
                break;

            Buffer.BlockCopy(source, readPos, alpha, writePos, literalCount);
            readPos += literalCount;
            writePos += literalCount;
        }

        return (alpha, readPos - offset);
    }

    private static (byte[] Alpha, int BytesConsumed) ReadPackedAlpha(byte[] source, int offset, int maxLength)
    {
        byte[] alpha = new byte[64 * 64];
        int sourceEnd = Math.Min(source.Length, offset + maxLength);
        int readPos = offset;
        int writePos = 0;

        for (int row = 0; row < 64 && readPos < sourceEnd; row++)
        {
            for (int column = 0; column < 32 && readPos < sourceEnd; column++)
            {
                byte packed = source[readPos++];
                byte low = (byte)((packed & 0x0F) * 17);
                byte high = (byte)(((packed >> 4) & 0x0F) * 17);

                alpha[writePos++] = low;
                alpha[writePos++] = column == 31 ? low : high;
            }
        }

        return (alpha, readPos - offset);
    }

    private static void ApplyLegacyEdgeFix(byte[] alpha)
    {
        for (int row = 0; row < 64; row++)
            alpha[(row * 64) + 63] = alpha[(row * 64) + 62];

        Buffer.BlockCopy(alpha, 62 * 64, alpha, 63 * 64, 64);
        alpha[(64 * 64) - 1] = alpha[(62 * 64) + 62];
    }

    private static byte[] ExpandFixedBigAlpha(byte[] source, int offset, int available)
    {
        byte[] alpha = new byte[64 * 64];
        int readPos = offset;
        int sourceEnd = Math.Min(source.Length, offset + available);

        for (int row = 0; row < 63 && readPos < sourceEnd; row++)
        {
            int rowStart = row * 64;
            int rowBytes = Math.Min(63, sourceEnd - readPos);
            if (rowBytes > 0)
                Buffer.BlockCopy(source, readPos, alpha, rowStart, rowBytes);

            readPos += rowBytes;
            alpha[rowStart + 63] = alpha[rowStart + Math.Max(0, rowBytes - 1)];
        }

        Buffer.BlockCopy(alpha, 62 * 64, alpha, 63 * 64, 64);
        return alpha;
    }

    private static int? ResolveNextOffset(AdtTextureLayerDescriptor currentLayer, AdtTextureLayerDescriptor? nextLayer, int mcalLength)
    {
        if (nextLayer is null)
            return null;

        int currentOffset = unchecked((int)currentLayer.AlphaOffset);
        int nextOffset = unchecked((int)nextLayer.AlphaOffset);
        if (nextOffset <= currentOffset || nextOffset > mcalLength)
            return null;

        return nextOffset;
    }

    private static bool ResolveLayerBigAlpha(int currentOffset, int? nextOffset, int mcalLength, bool defaultBigAlpha)
    {
        if (currentOffset < 0 || currentOffset >= mcalLength)
            return defaultBigAlpha;

        int span = nextOffset.HasValue
            ? nextOffset.Value - currentOffset
            : mcalLength - currentOffset;
        if (span >= 4096)
            return true;

        if (span > 0 && span <= 2048)
            return false;

        return defaultBigAlpha;
    }

    private static bool ShouldForceCompressedAlpha(uint flags, int maxLength)
    {
        if ((flags & CompressedAlphaFlag) != 0)
            return false;

        return maxLength > 0 && maxLength < 2048;
    }
}