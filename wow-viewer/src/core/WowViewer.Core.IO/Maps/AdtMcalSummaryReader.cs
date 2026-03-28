using System.Buffers.Binary;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class AdtMcalSummaryReader
{
    private const int RootMcnkHeaderSize = 128;
    private const int RootMcnkMcalSizeOffset = 0x28;
    private const int McnrConsumedSize = 0x1C0;

    public static AdtMcalSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, Path.GetFullPath(path));
        return Read(stream, fileSummary);
    }

    public static AdtMcalSummary Read(Stream stream, MapFileSummary fileSummary)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(fileSummary);

        if (fileSummary.Kind is not (MapFileKind.Adt or MapFileKind.AdtTex))
            throw new InvalidDataException($"ADT MCAL summary requires a root ADT or _tex0.adt file, but found {fileSummary.Kind}.");

        AdtMcalDecodeProfile decodeProfile = fileSummary.Kind == MapFileKind.AdtTex
            ? AdtMcalDecodeProfile.Cataclysm400
            : AdtMcalDecodeProfile.LichKingStrict;
        bool defaultBigAlpha = fileSummary.Kind == MapFileKind.AdtTex;

        int mcnkWithLayerTableCount = 0;
        int overlayLayerCount = 0;
        int decodedLayerCount = 0;
        int missingPayloadLayerCount = 0;
        int decodeFailureCount = 0;
        int compressedLayerCount = 0;
        int bigAlphaLayerCount = 0;
        int bigAlphaFixedLayerCount = 0;
        int packedLayerCount = 0;

        foreach (MapChunkLocation mcnkChunk in fileSummary.Chunks.Where(static chunk => chunk.Id == MapChunkIds.Mcnk))
        {
            byte[] payload = MapSummaryReaderCommon.ReadChunkPayload(stream, mcnkChunk);
            ParsedMcnkTextureData parsed = ParseMcnkTextureData(payload, fileSummary.Kind);
            if (parsed.Layers.Count == 0)
                continue;

            mcnkWithLayerTableCount++;
            for (int layerIndex = 1; layerIndex < parsed.Layers.Count; layerIndex++)
            {
                overlayLayerCount++;
                if (parsed.McalData is not { Length: > 0 })
                {
                    missingPayloadLayerCount++;
                    continue;
                }

                AdtTextureLayerDescriptor layer = parsed.Layers[layerIndex];
                AdtTextureLayerDescriptor? nextLayer = layerIndex + 1 < parsed.Layers.Count ? parsed.Layers[layerIndex + 1] : null;
                AdtMcalDecodedLayer? decodedLayer = AdtMcalDecoder.DecodeLayer(
                    parsed.McalData,
                    layer,
                    nextLayer,
                    defaultBigAlpha,
                    parsed.DoNotFixAlphaMap,
                    decodeProfile);
                if (decodedLayer is null)
                {
                    if (unchecked((int)layer.AlphaOffset) >= 0 && unchecked((int)layer.AlphaOffset) < parsed.McalData.Length)
                        decodeFailureCount++;
                    else
                        missingPayloadLayerCount++;

                    continue;
                }

                decodedLayerCount++;
                switch (decodedLayer.Encoding)
                {
                    case AdtMcalAlphaEncoding.Compressed:
                        compressedLayerCount++;
                        break;
                    case AdtMcalAlphaEncoding.BigAlpha:
                        bigAlphaLayerCount++;
                        break;
                    case AdtMcalAlphaEncoding.BigAlphaFixed:
                        bigAlphaFixedLayerCount++;
                        break;
                    case AdtMcalAlphaEncoding.Packed4Bit:
                        packedLayerCount++;
                        break;
                }
            }
        }

        return new AdtMcalSummary(
            fileSummary.SourcePath,
            fileSummary.Kind,
            decodeProfile,
            mcnkWithLayerTableCount,
            overlayLayerCount,
            decodedLayerCount,
            missingPayloadLayerCount,
            decodeFailureCount,
            compressedLayerCount,
            bigAlphaLayerCount,
            bigAlphaFixedLayerCount,
            packedLayerCount);
    }

    private static ParsedMcnkTextureData ParseMcnkTextureData(byte[] payload, MapFileKind kind)
    {
        uint flags = 0;
        int startOffset = 0;
        int? headerMcalPayloadSize = null;
        if (kind == MapFileKind.Adt && payload.Length >= RootMcnkHeaderSize)
        {
            flags = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0, 4));
            uint sizeMcal = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(RootMcnkMcalSizeOffset, 4));
            if (sizeMcal >= ChunkHeader.SizeInBytes)
                headerMcalPayloadSize = checked((int)(sizeMcal - ChunkHeader.SizeInBytes));

            startOffset = RootMcnkHeaderSize;
        }

        List<AdtTextureLayerDescriptor> layers = [];
        byte[]? mcalData = null;
        int position = startOffset;
        while (position <= payload.Length - ChunkHeader.SizeInBytes)
        {
            if (!ChunkHeaderReader.TryRead(payload.AsSpan(position, ChunkHeader.SizeInBytes), out ChunkHeader header))
                break;

            int declaredSize = checked((int)header.Size);
            int consumedSize = header.Id == AdtChunkIds.Mcnr
                ? Math.Max(declaredSize, McnrConsumedSize)
                : declaredSize;
            long nextOffset = (long)position + ChunkHeader.SizeInBytes + consumedSize;
            if (nextOffset > payload.Length)
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
                if (headerMcalPayloadSize.HasValue && headerMcalPayloadSize.Value > mcalPayloadSize && dataOffset + headerMcalPayloadSize.Value <= payload.Length)
                {
                    mcalPayloadSize = headerMcalPayloadSize.Value;
                    nextOffset = (long)position + ChunkHeader.SizeInBytes + mcalPayloadSize;
                }

                mcalData = new byte[mcalPayloadSize];
                Buffer.BlockCopy(payload, dataOffset, mcalData, 0, mcalPayloadSize);
            }

            position = checked((int)nextOffset);
        }

        return new ParsedMcnkTextureData(layers, mcalData, (flags & 0x8000u) != 0);
    }

    private sealed class ParsedMcnkTextureData
    {
        public ParsedMcnkTextureData(IReadOnlyList<AdtTextureLayerDescriptor> layers, byte[]? mcalData, bool doNotFixAlphaMap)
        {
            Layers = layers;
            McalData = mcalData;
            DoNotFixAlphaMap = doNotFixAlphaMap;
        }

        public IReadOnlyList<AdtTextureLayerDescriptor> Layers { get; }

        public byte[]? McalData { get; }

        public bool DoNotFixAlphaMap { get; }
    }
}