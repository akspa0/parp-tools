using System.Buffers.Binary;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class AdtMcnkSummaryReader
{
    private const int RootMcnkHeaderSize = 128;
    private const int RootMcnkSubchunkOffset = 0x80;
    private const uint LiquidFlagMask = 0x3Cu;
    private const uint MccvFlagMask = 0x40u;
    private const int MclyEntrySize = 16;
    private const int McnrConsumedSize = 0x1C0;

    private static readonly HashSet<FourCC> KnownSubchunks =
    [
        AdtChunkIds.Mcvt,
        AdtChunkIds.Mcnr,
        AdtChunkIds.Mcly,
        AdtChunkIds.Mcal,
        AdtChunkIds.Mcsh,
        AdtChunkIds.Mccv,
        AdtChunkIds.Mclq,
        AdtChunkIds.Mcrd,
        AdtChunkIds.Mcrw,
    ];

    public static AdtMcnkSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, Path.GetFullPath(path));
        return Read(stream, fileSummary);
    }

    public static AdtMcnkSummary Read(Stream stream, MapFileSummary fileSummary)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(fileSummary);

        if (fileSummary.Kind is not (MapFileKind.Adt or MapFileKind.AdtTex or MapFileKind.AdtObj))
            throw new InvalidDataException($"ADT MCNK summary requires an ADT-family file, but found {fileSummary.Kind}.");

        List<MapChunkLocation> mcnkChunks = fileSummary.Chunks.Where(static chunk => chunk.Id == MapChunkIds.Mcnk).ToList();
        HashSet<ulong> indices = [];
        HashSet<uint> areaIds = [];

        int zeroLengthMcnkCount = 0;
        int headerLikeMcnkCount = 0;
        int duplicateIndexCount = 0;
        int chunksWithHoles = 0;
        int chunksWithLiquidFlags = 0;
        int chunksWithMccvFlag = 0;
        int chunksWithMcvt = 0;
        int chunksWithMcnr = 0;
        int chunksWithMcly = 0;
        int chunksWithMcal = 0;
        int chunksWithMcsh = 0;
        int chunksWithMccv = 0;
        int chunksWithMclq = 0;
        int chunksWithMcrd = 0;
        int chunksWithMcrw = 0;
        int totalLayerCount = 0;
        int maxLayerCount = 0;
        int chunksWithMultipleLayers = 0;
        int mccvFlagWithoutPayloadCount = 0;
        int liquidFlagWithoutPayloadCount = 0;

        foreach (MapChunkLocation mcnkChunk in mcnkChunks)
        {
            byte[] payload = MapSummaryReaderCommon.ReadChunkPayload(stream, mcnkChunk);
            if (payload.Length == 0)
            {
                zeroLengthMcnkCount++;
                continue;
            }

            bool hasRootHeader = fileSummary.Kind == MapFileKind.Adt && payload.Length >= RootMcnkHeaderSize;
            uint flags = 0;
            int layerCountFromHeader = 0;
            int subchunkStart = 0;
            if (hasRootHeader)
            {
                headerLikeMcnkCount++;
                flags = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0x00, 4));
                uint indexX = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0x04, 4));
                uint indexY = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0x08, 4));
                layerCountFromHeader = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0x0C, 4)));
                uint areaId = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0x34, 4));
                ushort holes = BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(0x3C, 2));
                if (!indices.Add(((ulong)indexX << 32) | indexY))
                    duplicateIndexCount++;

                areaIds.Add(areaId);

                if (holes != 0)
                    chunksWithHoles++;

                if ((flags & LiquidFlagMask) != 0)
                    chunksWithLiquidFlags++;

                if ((flags & MccvFlagMask) != 0)
                    chunksWithMccvFlag++;

                subchunkStart = RootMcnkSubchunkOffset;
            }

            McnkChunkSignals signals = ScanMcnkSubchunks(payload, subchunkStart);
            if (signals.HasMcvt)
                chunksWithMcvt++;

            if (signals.HasMcnr)
                chunksWithMcnr++;

            if (signals.HasMcly)
                chunksWithMcly++;

            if (signals.HasMcal)
                chunksWithMcal++;

            if (signals.HasMcsh)
                chunksWithMcsh++;

            if (signals.HasMccv)
                chunksWithMccv++;

            if (signals.HasMclq)
                chunksWithMclq++;

            if (signals.HasMcrd)
                chunksWithMcrd++;

            if (signals.HasMcrw)
                chunksWithMcrw++;

            int layerCount = signals.LayerCount > 0 ? signals.LayerCount : layerCountFromHeader;
            totalLayerCount += layerCount;
            maxLayerCount = Math.Max(maxLayerCount, layerCount);
            if (layerCount > 1)
                chunksWithMultipleLayers++;

            if ((flags & MccvFlagMask) != 0 && !signals.HasMccv)
                mccvFlagWithoutPayloadCount++;

            if ((flags & LiquidFlagMask) != 0 && !signals.HasMclq)
                liquidFlagWithoutPayloadCount++;
        }

        return new AdtMcnkSummary(
            fileSummary.SourcePath,
            fileSummary.Kind,
            mcnkCount: mcnkChunks.Count,
            zeroLengthMcnkCount,
            headerLikeMcnkCount,
            distinctIndexCount: indices.Count,
            duplicateIndexCount,
            distinctAreaIdCount: areaIds.Count,
            chunksWithHoles,
            chunksWithLiquidFlags,
            chunksWithMccvFlag,
            chunksWithMcvt,
            chunksWithMcnr,
            chunksWithMcly,
            chunksWithMcal,
            chunksWithMcsh,
            chunksWithMccv,
            chunksWithMclq,
            chunksWithMcrd,
            chunksWithMcrw,
            totalLayerCount,
            maxLayerCount,
            chunksWithMultipleLayers,
            mccvFlagWithoutPayloadCount,
            liquidFlagWithoutPayloadCount);
    }

    private static McnkChunkSignals ScanMcnkSubchunks(byte[] payload, int startOffset)
    {
        McnkChunkSignals signals = default;
        int position = startOffset;
        while (position <= payload.Length - ChunkHeader.SizeInBytes)
        {
            if (!ChunkHeaderReader.TryRead(payload.AsSpan(position, ChunkHeader.SizeInBytes), out ChunkHeader header))
                break;

            if (!KnownSubchunks.Contains(header.Id))
                break;

            int consumedSize = checked((int)header.Size);
            if (header.Id == AdtChunkIds.Mcnr)
                consumedSize = Math.Max(consumedSize, McnrConsumedSize);

            long nextOffset = (long)position + ChunkHeader.SizeInBytes + consumedSize;
            if (nextOffset > payload.Length)
                break;

            if (header.Id == AdtChunkIds.Mcvt)
                signals.HasMcvt = true;
            else if (header.Id == AdtChunkIds.Mcnr)
                signals.HasMcnr = true;
            else if (header.Id == AdtChunkIds.Mcly)
            {
                signals.HasMcly = true;
                signals.LayerCount += checked((int)header.Size / MclyEntrySize);
            }
            else if (header.Id == AdtChunkIds.Mcal)
                signals.HasMcal = true;
            else if (header.Id == AdtChunkIds.Mcsh)
                signals.HasMcsh = true;
            else if (header.Id == AdtChunkIds.Mccv)
                signals.HasMccv = true;
            else if (header.Id == AdtChunkIds.Mclq)
                signals.HasMclq = true;
            else if (header.Id == AdtChunkIds.Mcrd)
                signals.HasMcrd = true;
            else if (header.Id == AdtChunkIds.Mcrw)
                signals.HasMcrw = true;

            position = checked((int)nextOffset);
        }

        return signals;
    }

    private struct McnkChunkSignals
    {
        public bool HasMcvt;
        public bool HasMcnr;
        public bool HasMcly;
        public bool HasMcal;
        public bool HasMcsh;
        public bool HasMccv;
        public bool HasMclq;
        public bool HasMcrd;
        public bool HasMcrw;
        public int LayerCount;
    }
}
