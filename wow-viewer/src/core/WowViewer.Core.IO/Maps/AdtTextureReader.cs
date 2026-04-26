using System.Buffers.Binary;
using System.Text;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class AdtTextureReader
{
    private const int RootMcnkHeaderSize = 128;
    private const int McinEntrySize = 16;
    private const int ExpectedChunkCount = 256;
    private const int MhdrMcinOffset = 4;
    private const int MhdrMtexOffset = 8;

    public static AdtTextureFile Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, Path.GetFullPath(path));
        return Read(stream, fileSummary);
    }

    public static AdtTextureFile Read(Stream stream, MapFileSummary fileSummary)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(fileSummary);

        if (fileSummary.Kind is not (MapFileKind.Adt or MapFileKind.AdtTex))
            throw new InvalidDataException($"ADT texture reader requires a root ADT or _tex0.adt file, but found {fileSummary.Kind}.");

        AdtMcalDecodeProfile decodeProfile = fileSummary.Kind == MapFileKind.AdtTex
            ? AdtMcalDecodeProfile.Cataclysm400
            : AdtMcalDecodeProfile.LichKingStrict;

        if (fileSummary.Kind == MapFileKind.AdtTex
            && TryReadSplitTextureFile(stream, out IReadOnlyList<string> splitTextureNames, out IReadOnlyList<byte[]> splitChunkPayloads))
        {
            List<AdtTextureChunk> splitChunks = new(splitChunkPayloads.Count);
            for (int chunkIndex = 0; chunkIndex < splitChunkPayloads.Count; chunkIndex++)
                splitChunks.Add(AdtTextureChunkReader.Read(chunkIndex, splitChunkPayloads[chunkIndex], fileSummary.Kind, splitTextureNames));

            return new AdtTextureFile(fileSummary.SourcePath, fileSummary.Kind, decodeProfile, splitTextureNames, splitChunks);
        }

        IReadOnlyList<string> textureNames = MapSummaryReaderCommon.ReadStringEntries(
            MapSummaryReaderCommon.ReadChunkPayload(stream, fileSummary, MapChunkIds.Mtex));

        List<AdtTextureChunk> chunks = [];
        int resolvedChunkIndex = 0;
        foreach (MapChunkLocation mcnkChunk in ResolveTextureChunkLocations(stream, fileSummary))
        {
            byte[] payload = MapSummaryReaderCommon.ReadChunkPayload(stream, mcnkChunk);
            chunks.Add(AdtTextureChunkReader.Read(resolvedChunkIndex, payload, fileSummary.Kind, textureNames));
            resolvedChunkIndex++;
        }

        return new AdtTextureFile(fileSummary.SourcePath, fileSummary.Kind, decodeProfile, textureNames, chunks);
    }

    private static bool TryReadSplitTextureFile(Stream stream, out IReadOnlyList<string> textureNames, out IReadOnlyList<byte[]> chunkPayloads)
    {
        textureNames = Array.Empty<string>();
        chunkPayloads = Array.Empty<byte[]>();

        if (!stream.CanSeek)
            return false;

        byte[] bytes = ReadAllBytes(stream);
        int mhdrOffset = FindChunk(bytes, "MHDR");
        if (mhdrOffset < 0 || mhdrOffset + ChunkHeader.SizeInBytes > bytes.Length)
            return TryReadTopLevelSplitTextureFile(bytes, out textureNames, out chunkPayloads);

        int mhdrDataOffset = mhdrOffset + ChunkHeader.SizeInBytes;
        textureNames = ParseMtexViaMhdr(bytes, mhdrDataOffset);
        chunkPayloads = ResolveSplitMcnkPayloads(bytes, mhdrDataOffset);
        return textureNames.Count > 0 || chunkPayloads.Count > 0;
    }

    private static IReadOnlyList<MapChunkLocation> ResolveTextureChunkLocations(Stream stream, MapFileSummary fileSummary)
    {
        List<MapChunkLocation> topLevelChunks = fileSummary.Chunks
            .Where(static chunk => chunk.Id == MapChunkIds.Mcnk)
            .ToList();

        if (topLevelChunks.Count >= ExpectedChunkCount || !fileSummary.HasChunk(MapChunkIds.Mcin))
            return topLevelChunks;

        MapChunkLocation mcinChunk = fileSummary.Chunks.First(chunk => chunk.Id == MapChunkIds.Mcin);
        byte[] mcinPayload = MapSummaryReaderCommon.ReadChunkPayload(stream, mcinChunk);
        if (mcinPayload.Length < McinEntrySize)
            return topLevelChunks;

        List<MapChunkLocation> resolvedChunks = new(ExpectedChunkCount);
        for (int index = 0; index < ExpectedChunkCount && ((index + 1) * McinEntrySize) <= mcinPayload.Length; index++)
        {
            int entryOffset = index * McinEntrySize;
            uint chunkOffset = BinaryPrimitives.ReadUInt32LittleEndian(mcinPayload.AsSpan(entryOffset, 4));
            if (chunkOffset == 0)
                continue;

            long headerOffset = chunkOffset;
            if (!TryReadChunkHeader(stream, headerOffset, out ChunkHeader header))
                continue;

            int minimumPayloadSize = fileSummary.Kind == MapFileKind.Adt ? RootMcnkHeaderSize : 0;
            if (header.Id != MapChunkIds.Mcnk || header.Size < minimumPayloadSize)
                continue;

            long dataOffset = headerOffset + ChunkHeader.SizeInBytes;
            if (dataOffset > stream.Length || dataOffset + header.Size > stream.Length)
                continue;

            resolvedChunks.Add(new MapChunkLocation(MapChunkIds.Mcnk, header.Size, headerOffset, dataOffset));
        }

        return resolvedChunks.Count > topLevelChunks.Count ? resolvedChunks : topLevelChunks;
    }

    private static bool TryReadChunkHeader(Stream stream, long headerOffset, out ChunkHeader header)
    {
        long previousPosition = stream.Position;
        try
        {
            if (headerOffset < 0 || headerOffset > stream.Length - ChunkHeader.SizeInBytes)
            {
                header = default;
                return false;
            }

            Span<byte> headerBytes = stackalloc byte[ChunkHeader.SizeInBytes];
            stream.Position = headerOffset;
            stream.ReadExactly(headerBytes);
            return ChunkHeaderReader.TryRead(headerBytes, out header);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static byte[] ReadAllBytes(Stream stream)
    {
        long previousPosition = stream.Position;
        try
        {
            stream.Position = 0;
            byte[] bytes = new byte[stream.Length];
            stream.ReadExactly(bytes);
            return bytes;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static int FindChunk(byte[] bytes, string fourCC)
    {
        string reversed = new string(fourCC.Reverse().ToArray());
        for (int position = 0; position + ChunkHeader.SizeInBytes <= bytes.Length;)
        {
            string chunkId = Encoding.ASCII.GetString(bytes, position, 4);
            int size = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(position + 4, 4));
            if (size < 0)
                break;

            if (chunkId == reversed)
                return position;

            int next = position + ChunkHeader.SizeInBytes + size + ((size & 1) == 1 ? 1 : 0);
            if (next <= position)
                break;

            position = next;
        }

        return -1;
    }

    private static IReadOnlyList<string> ParseMtexViaMhdr(byte[] bytes, int mhdrDataOffset)
    {
        if (mhdrDataOffset + MhdrMtexOffset + 4 > bytes.Length)
            return Array.Empty<string>();

        int mtexOffset = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(mhdrDataOffset + MhdrMtexOffset, 4));
        if (mtexOffset == 0)
            return Array.Empty<string>();

        int mtexAbsoluteOffset = mhdrDataOffset + mtexOffset;
        if (mtexAbsoluteOffset + ChunkHeader.SizeInBytes > bytes.Length)
            return Array.Empty<string>();

        string mtexSignature = Encoding.ASCII.GetString(bytes, mtexAbsoluteOffset, 4);
        if (!string.Equals(mtexSignature, "XETM", StringComparison.Ordinal))
            return Array.Empty<string>();

        int mtexSize = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(mtexAbsoluteOffset + 4, 4));
        int mtexDataOffset = mtexAbsoluteOffset + ChunkHeader.SizeInBytes;
        if (mtexSize <= 0 || mtexDataOffset + mtexSize > bytes.Length)
            return Array.Empty<string>();

        return ParseNullStrings(bytes, mtexDataOffset, mtexSize);
    }

    private static IReadOnlyList<string> ParseNullStrings(byte[] bytes, int offset, int size)
    {
        List<string> values = [];
        int start = offset;
        int end = offset + size;
        for (int index = offset; index < end; index++)
        {
            if (bytes[index] != 0)
                continue;

            if (index > start)
                values.Add(Encoding.ASCII.GetString(bytes, start, index - start));

            start = index + 1;
        }

        return values;
    }

    private static bool TryReadTopLevelSplitTextureFile(byte[] bytes, out IReadOnlyList<string> textureNames, out IReadOnlyList<byte[]> chunkPayloads)
    {
        List<string> names = [];
        List<byte[]> payloads = [];

        int position = 0;
        while (position + ChunkHeader.SizeInBytes <= bytes.Length)
        {
            string signature = Encoding.ASCII.GetString(bytes, position, 4);
            int size = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(position + 4, 4));
            if (size < 0 || position + ChunkHeader.SizeInBytes + size > bytes.Length)
                break;

            int dataOffset = position + ChunkHeader.SizeInBytes;
            if (string.Equals(signature, "XETM", StringComparison.Ordinal))
            {
                names.AddRange(ParseNullStrings(bytes, dataOffset, size));
            }
            else if (string.Equals(signature, "KNCM", StringComparison.Ordinal))
            {
                byte[] payload = new byte[size];
                Buffer.BlockCopy(bytes, dataOffset, payload, 0, size);
                payloads.Add(payload);
            }

            int next = position + ChunkHeader.SizeInBytes + size;
            if (next <= position)
                break;

            position = next;
        }

        textureNames = names;
        chunkPayloads = payloads;
        return names.Count > 0 || payloads.Count > 0;
    }

    private static IReadOnlyList<byte[]> ResolveSplitMcnkPayloads(byte[] bytes, int mhdrDataOffset)
    {
        List<int> offsets;
        if (mhdrDataOffset + MhdrMcinOffset + 4 > bytes.Length)
            return Array.Empty<byte[]>();

        int mcinOffset = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(mhdrDataOffset + MhdrMcinOffset, 4));
        if (mcinOffset == 0)
        {
            offsets = ReadMcnkOffsetsByChunkScan(bytes);
        }
        else
        {
            int mcinAbsoluteOffset = mhdrDataOffset + mcinOffset;
            if (mcinAbsoluteOffset + ChunkHeader.SizeInBytes > bytes.Length)
                return Array.Empty<byte[]>();

            string mcinSignature = Encoding.ASCII.GetString(bytes, mcinAbsoluteOffset, 4);
            if (!string.Equals(mcinSignature, "NICM", StringComparison.Ordinal))
                return Array.Empty<byte[]>();

            int mcinSize = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(mcinAbsoluteOffset + 4, 4));
            int mcinDataOffset = mcinAbsoluteOffset + ChunkHeader.SizeInBytes;
            if (mcinSize <= 0 || mcinDataOffset + mcinSize > bytes.Length)
                return Array.Empty<byte[]>();

            offsets = [];
            for (int index = 0; index < ExpectedChunkCount && ((index + 1) * McinEntrySize) <= mcinSize; index++)
            {
                int entryOffset = mcinDataOffset + (index * McinEntrySize);
                int chunkOffset = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(entryOffset, 4));
                if (chunkOffset > 0)
                    offsets.Add(chunkOffset);
            }
        }

        if (offsets.Count == 0)
            return Array.Empty<byte[]>();

        List<byte[]> payloads = [];
        int entryCount = Math.Min(ExpectedChunkCount, offsets.Count);
        for (int index = 0; index < entryCount; index++)
        {
            int offset = offsets[index];
            if (offset <= 0 || offset + ChunkHeader.SizeInBytes > bytes.Length)
                continue;

            string signature = Encoding.ASCII.GetString(bytes, offset, 4);
            if (!string.Equals(signature, "KNCM", StringComparison.Ordinal))
                continue;

            int size = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(offset + 4, 4));
            if (size <= 0 || offset + ChunkHeader.SizeInBytes + size > bytes.Length)
                continue;

            byte[] payload = new byte[size];
            Buffer.BlockCopy(bytes, offset + ChunkHeader.SizeInBytes, payload, 0, size);
            payloads.Add(payload);
        }

        return payloads;
    }

    private static List<int> ReadMcnkOffsetsByChunkScan(byte[] bytes)
    {
        List<int> offsets = [];
        int position = 0;
        while (position + ChunkHeader.SizeInBytes <= bytes.Length)
        {
            string signature = Encoding.ASCII.GetString(bytes, position, 4);
            int size = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(position + 4, 4));
            if (size < 0)
                break;

            if (string.Equals(signature, "KNCM", StringComparison.Ordinal))
                offsets.Add(position);

            int next = position + ChunkHeader.SizeInBytes + size + ((size & 1) == 1 ? 1 : 0);
            if (next <= position)
                break;

            position = next;
        }

        return offsets;
    }
}