using System.Buffers.Binary;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoMddlReader
{
    public static WmoDetailDoodadDocument? ReadDetailDoodads(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using FileStream stream = File.OpenRead(path);
        return ReadDetailDoodads(stream, Path.GetFullPath(path));
    }

    public static WmoDetailDoodadDocument? ReadDetailDoodads(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        var (_, chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        return Read(stream, chunks);
    }

    public static WmoDetailDoodadDocument? Read(Stream stream, IReadOnlyList<ChunkSpan> chunks)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(chunks);

        byte[]? payload = WmoRootReaderCommon.TryReadChunkPayload(stream, chunks, WmoChunkIds.Mddl);
        if (payload is null || payload.Length < 6)
            return null;

        return Read(payload);
    }

    public static WmoDetailDoodadDocument Read(ReadOnlySpan<byte> payload)
    {
        if (payload.Length < 6)
            throw new InvalidDataException($"MDDL chunk payload is too short ({payload.Length} bytes). Expected at least 6 bytes.");

        float minTriangleArea = BinaryPrimitives.ReadSingleLittleEndian(payload[..4]);
        ushort layerCount = BinaryPrimitives.ReadUInt16LittleEndian(payload.Slice(4, 2));

        int offset = 6;
        List<WmoDetailDoodadLayer> layers = new(layerCount);

        for (int layerIndex = 0; layerIndex < layerCount && offset + 2 <= payload.Length; layerIndex++)
        {
            byte density = payload[offset++];
            byte doodadCount = payload[offset++];
            List<WmoDetailDoodadEntry> doodads = new(doodadCount);

            for (int d = 0; d < doodadCount && offset + 5 <= payload.Length; d++)
            {
                uint doodadId = BinaryPrimitives.ReadUInt32LittleEndian(payload.Slice(offset, 4));
                offset += 4;
                byte weight = payload[offset++];
                doodads.Add(new WmoDetailDoodadEntry(doodadId, weight));
            }

            layers.Add(new WmoDetailDoodadLayer(density, doodads));
        }

        List<WmoGroupDetailDoodadData> groups = [];
        while (offset + 6 <= payload.Length)
        {
            ushort groupIndex = BinaryPrimitives.ReadUInt16LittleEndian(payload.Slice(offset, 2));
            offset += 2;
            uint dataSize = BinaryPrimitives.ReadUInt32LittleEndian(payload.Slice(offset, 4));
            offset += 4;

            int actualSize = (int)Math.Min((long)dataSize, payload.Length - offset);
            byte[] rawBytes = payload.Slice(offset, actualSize).ToArray();
            offset += actualSize;

            List<WmoDetailDoodadDecodedCommand> commands = DecodeGroupData(rawBytes);
            groups.Add(new WmoGroupDetailDoodadData(groupIndex, rawBytes, commands));
        }

        return new WmoDetailDoodadDocument(minTriangleArea, layers, groups);
    }

    public static List<WmoDetailDoodadDecodedCommand> DecodeGroupData(ReadOnlySpan<byte> data)
    {
        List<WmoDetailDoodadDecodedCommand> commands = [];
        int offset = 0;

        while (offset + 2 <= data.Length)
        {
            ushort layerIndex = BinaryPrimitives.ReadUInt16LittleEndian(data.Slice(offset, 2));
            offset += 2;
            if (layerIndex == 0xFFFF)
                break;

            while (offset + 2 <= data.Length)
            {
                ushort rawBatchIndex = BinaryPrimitives.ReadUInt16LittleEndian(data.Slice(offset, 2));
                offset += 2;
                if (rawBatchIndex == 0xFFFF)
                    break;

                bool rollAll = (rawBatchIndex & 0x8000) != 0;
                ushort batchIndex = (ushort)(rawBatchIndex & 0x7FFF);

                if (rollAll)
                {
                    commands.Add(new WmoDetailDoodadDecodedCommand(
                        layerIndex,
                        batchIndex,
                        RollAllLocations: true,
                        LocRangeIndex: 0,
                        SingleLocation: false,
                        Locations: []));
                    continue;
                }

                int locRangeIndex = 0;
                while (offset < data.Length)
                {
                    byte locRangePart = data[offset++];
                    locRangeIndex += locRangePart & 0x7F;
                    if (locRangePart == 0xFF)
                        break;
                    if (locRangePart == 0x7F)
                        continue;

                    bool singleLoc = (locRangePart & 0x80) != 0;
                    int loc = 0;
                    List<int> locs = [];

                    while (offset < data.Length)
                    {
                        byte locPart = data[offset++];
                        loc += locPart;
                        if (locPart == 0xFF)
                            break;
                        if (locPart == 0xFE)
                            continue;

                        locs.Add(loc);
                    }

                    commands.Add(new WmoDetailDoodadDecodedCommand(
                        layerIndex,
                        batchIndex,
                        RollAllLocations: false,
                        LocRangeIndex: locRangeIndex,
                        SingleLocation: singleLoc,
                        Locations: locs));
                }
            }
        }

        return commands;
    }
}

