using System.Buffers.Binary;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class AdtTextureChunkReader
{
    private const int RootMcnkHeaderSize = 128;
    private const int RootMcnkMcalSizeOffset = 0x28;
    private const int McnrConsumedSize = 0x1C0;

    public static AdtTextureChunk Read(int chunkIndex, byte[] payload, MapFileKind kind, IReadOnlyList<string> textureNames)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(chunkIndex);
        ArgumentNullException.ThrowIfNull(payload);
        ArgumentNullException.ThrowIfNull(textureNames);

        if (kind is not (MapFileKind.Adt or MapFileKind.AdtTex))
            throw new InvalidDataException($"ADT texture chunk reader requires a root ADT or _tex0.adt payload, but found {kind}.");

        AdtMcalDecodeProfile decodeProfile = kind == MapFileKind.AdtTex
            ? AdtMcalDecodeProfile.Cataclysm400
            : AdtMcalDecodeProfile.LichKingStrict;
        bool defaultBigAlpha = kind == MapFileKind.AdtTex;

        ParsedTextureChunkData parsed = ParseMcnkTextureData(payload, kind);
        List<AdtTextureChunkLayer> layers = [];
        for (int layerIndex = 0; layerIndex < parsed.Layers.Count; layerIndex++)
        {
            AdtTextureLayerDescriptor layer = parsed.Layers[layerIndex];
            AdtTextureLayerDescriptor? nextLayer = layerIndex + 1 < parsed.Layers.Count ? parsed.Layers[layerIndex + 1] : null;
            AdtMcalDecodedLayer? decoded = null;
            if (layerIndex > 0 && parsed.McalData is { Length: > 0 })
            {
                decoded = AdtMcalDecoder.DecodeLayer(
                    parsed.McalData,
                    layer,
                    nextLayer,
                    defaultBigAlpha,
                    parsed.DoNotFixAlphaMap,
                    decodeProfile);
            }

            string? texturePath = layer.TextureId < textureNames.Count
                ? textureNames[checked((int)layer.TextureId)]
                : null;
            layers.Add(new AdtTextureChunkLayer(
                layer.Index,
                layer.TextureId,
                texturePath,
                layer.Flags,
                layer.AlphaOffset,
                layer.EffectId,
                decoded));
        }

        return new AdtTextureChunk(
            chunkIndex,
            parsed.ChunkX ?? chunkIndex % 16,
            parsed.ChunkY ?? chunkIndex / 16,
            parsed.DoNotFixAlphaMap,
            parsed.McalData?.Length ?? 0,
            parsed.ShadowMap,
            layers);
    }

    private static ParsedTextureChunkData ParseMcnkTextureData(byte[] payload, MapFileKind kind)
    {
        uint flags = 0;
        int startOffset = 0;
        int? headerMcalPayloadSize = null;
        int? chunkX = null;
        int? chunkY = null;
        if (TryReadEmbeddedMcnkHeader(payload, kind, out flags, out headerMcalPayloadSize, out chunkX, out chunkY))
        {
            startOffset = RootMcnkHeaderSize;
        }

        List<AdtTextureLayerDescriptor> layers = [];
        byte[]? mcalData = null;
        byte[]? shadowMap = null;
        int position = startOffset;
        while (position <= payload.Length - ChunkHeader.SizeInBytes)
        {
            if (!ChunkHeaderReader.TryRead(payload.AsSpan(position, ChunkHeader.SizeInBytes), out ChunkHeader header))
                break;

            if (!TryConvertChunkPayloadSize(header.Size, out int declaredSize))
                break;

            long consumedSize = header.Id == AdtChunkIds.Mcnr
                ? Math.Max((long)declaredSize, McnrConsumedSize)
                : declaredSize;
            long nextOffset = (long)position + ChunkHeader.SizeInBytes + consumedSize;
            if (nextOffset > payload.Length || nextOffset <= position)
                break;

            int dataOffset = position + ChunkHeader.SizeInBytes;
            if (header.Id == AdtChunkIds.Mcly)
            {
                byte[] layerPayload = new byte[declaredSize];
                Buffer.BlockCopy(payload, dataOffset, layerPayload, 0, declaredSize);
                layers.AddRange(AdtMcalDecoder.ReadTextureLayers(layerPayload));
            }
            else if (header.Id == AdtChunkIds.Mcal)
            {
                int mcalPayloadSize = declaredSize;
                if (headerMcalPayloadSize.HasValue
                    && headerMcalPayloadSize.Value > mcalPayloadSize
                    && headerMcalPayloadSize.Value <= payload.Length - dataOffset)
                {
                    mcalPayloadSize = headerMcalPayloadSize.Value;
                    nextOffset = (long)position + ChunkHeader.SizeInBytes + mcalPayloadSize;
                }

                mcalData = new byte[mcalPayloadSize];
                Buffer.BlockCopy(payload, dataOffset, mcalData, 0, mcalPayloadSize);
            }
            else if (header.Id == AdtChunkIds.Mcsh)
            {
                shadowMap = ReadShadowMap(payload, dataOffset, declaredSize);
            }

            position = checked((int)nextOffset);
        }

        return new ParsedTextureChunkData(layers, mcalData, shadowMap, (flags & 0x8000u) != 0, chunkX, chunkY);
    }

    private static byte[]? ReadShadowMap(byte[] payload, int dataOffset, int declaredSize)
    {
        if (declaredSize < 8 || dataOffset < 0 || dataOffset + declaredSize > payload.Length)
            return null;

        byte[] shadow = new byte[64 * 64];
        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                int sourceByteIndex = dataOffset + (y * 8) + (x / 8);
                if (sourceByteIndex >= dataOffset + declaredSize)
                    break;

                int bitIndex = x % 8;
                bool isShadowed = (payload[sourceByteIndex] & (1 << bitIndex)) != 0;
                shadow[(y * 64) + x] = isShadowed ? (byte)255 : (byte)0;
            }
        }

        return shadow;
    }

    private static bool TryReadEmbeddedMcnkHeader(
        byte[] payload,
        MapFileKind kind,
        out uint flags,
        out int? headerMcalPayloadSize,
        out int? chunkX,
        out int? chunkY)
    {
        flags = 0;
        headerMcalPayloadSize = null;
        chunkX = null;
        chunkY = null;

        if (payload.Length < RootMcnkHeaderSize + ChunkHeader.SizeInBytes)
            return false;

        if (!ChunkHeaderReader.TryRead(payload.AsSpan(RootMcnkHeaderSize, ChunkHeader.SizeInBytes), out ChunkHeader firstSubchunk))
            return false;

        if (!IsKnownTextureSubchunk(firstSubchunk.Id))
            return false;

        if (!TryConvertChunkPayloadSize(firstSubchunk.Size, out int declaredSize))
            return false;

        long consumedSize = firstSubchunk.Id == AdtChunkIds.Mcnr
            ? Math.Max((long)declaredSize, McnrConsumedSize)
            : declaredSize;
        long firstSubchunkEnd = (long)RootMcnkHeaderSize + ChunkHeader.SizeInBytes + consumedSize;
        if (firstSubchunkEnd > payload.Length)
            return false;

        flags = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0, 4));
        if (kind == MapFileKind.Adt || kind == MapFileKind.AdtTex)
        {
            uint rawChunkX = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0x04, 4));
            uint rawChunkY = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0x08, 4));
            if (rawChunkX > int.MaxValue || rawChunkY > int.MaxValue)
                return false;

            chunkX = (int)rawChunkX;
            chunkY = (int)rawChunkY;

            uint sizeMcal = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(RootMcnkMcalSizeOffset, 4));
            if (sizeMcal >= ChunkHeader.SizeInBytes)
            {
                uint payloadSize = sizeMcal - ChunkHeader.SizeInBytes;
                if (payloadSize <= int.MaxValue)
                    headerMcalPayloadSize = (int)payloadSize;
            }
        }

        return true;
    }

    private static bool TryConvertChunkPayloadSize(uint rawSize, out int size)
    {
        if (rawSize > int.MaxValue)
        {
            size = 0;
            return false;
        }

        size = (int)rawSize;
        return true;
    }

    private static bool IsKnownTextureSubchunk(FourCC id)
    {
        return id == AdtChunkIds.Mcvt
            || id == AdtChunkIds.Mcnr
            || id == AdtChunkIds.Mcly
            || id == AdtChunkIds.Mcal
            || id == AdtChunkIds.Mcsh
            || id == AdtChunkIds.Mccv
            || id == AdtChunkIds.Mclq
            || id == AdtChunkIds.Mcrd
            || id == AdtChunkIds.Mcrw;
    }

    private sealed class ParsedTextureChunkData
    {
        public ParsedTextureChunkData(
            IReadOnlyList<AdtTextureLayerDescriptor> layers,
            byte[]? mcalData,
            byte[]? shadowMap,
            bool doNotFixAlphaMap,
            int? chunkX,
            int? chunkY)
        {
            Layers = layers;
            McalData = mcalData;
            ShadowMap = shadowMap;
            DoNotFixAlphaMap = doNotFixAlphaMap;
            ChunkX = chunkX;
            ChunkY = chunkY;
        }

        public IReadOnlyList<AdtTextureLayerDescriptor> Layers { get; }

        public byte[]? McalData { get; }

        public byte[]? ShadowMap { get; }

        public bool DoNotFixAlphaMap { get; }

        public int? ChunkX { get; }

        public int? ChunkY { get; }
    }
}
