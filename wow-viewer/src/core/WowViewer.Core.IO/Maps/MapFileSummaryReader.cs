using System.Buffers.Binary;
using WowViewer.Core.Chunks;
using WowViewer.Core.Files;
using WowViewer.Core.Maps;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.IO.Files;

namespace WowViewer.Core.IO.Maps;

public static class MapFileSummaryReader
{
    public static MapFileSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static MapFileSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        try
        {
            IReadOnlyList<ChunkSpan> chunkSpans = ReadTopLevelChunks(stream, sourcePath);
            MapChunkLocation[] chunks = chunkSpans
                .Select(static chunk => new MapChunkLocation(chunk.Header.Id, chunk.Header.Size, chunk.HeaderOffset, chunk.DataOffset))
                .ToArray();

            uint? version = TryReadVersion(stream, chunkSpans);
            WowFileDetection detection = WowFileDetector.Detect(sourcePath, chunkSpans, version);
            MapFileKind kind = ToMapFileKind(detection.Kind);
            return new MapFileSummary(sourcePath, kind, version, chunks);
        }
        catch (InvalidDataException) when (TryReadAlphaWdtSummary(stream, sourcePath, out MapFileSummary? alphaSummary))
        {
            return alphaSummary!;
        }
        catch (InvalidDataException) when (TryReadAdtPrefixSummary(stream, sourcePath, out MapFileSummary? adtSummary))
        {
            return adtSummary!;
        }
    }

    private static IReadOnlyList<ChunkSpan> ReadTopLevelChunks(Stream stream, string sourcePath)
    {
        if (!sourcePath.EndsWith(".adt", StringComparison.OrdinalIgnoreCase))
            return ChunkedFileReader.ReadTopLevelChunks(stream);

        try
        {
            return ChunkedFileReader.ReadTopLevelChunks(stream, padOddChunkSizes: false);
        }
        catch (InvalidDataException)
        {
            return ChunkedFileReader.ReadTopLevelChunks(stream, padOddChunkSizes: true);
        }
    }

    private static bool TryReadAdtPrefixSummary(Stream stream, string sourcePath, out MapFileSummary? summary)
    {
        summary = null;

        if (!stream.CanSeek || !sourcePath.EndsWith(".adt", StringComparison.OrdinalIgnoreCase))
            return false;

        long previousPosition = stream.Position;
        try
        {
            List<ChunkSpan> chunks = [];
            Span<byte> headerBytes = stackalloc byte[ChunkHeader.SizeInBytes];
            stream.Position = 0;

            while (stream.Position <= stream.Length - ChunkHeader.SizeInBytes)
            {
                long headerOffset = stream.Position;
                stream.ReadExactly(headerBytes);
                if (!ChunkHeaderReader.TryRead(headerBytes, out ChunkHeader header))
                    break;

                long dataOffset = stream.Position;
                long payloadEndOffset = unchecked(dataOffset + (long)header.Size);
                if (payloadEndOffset > stream.Length)
                    break;

                chunks.Add(new ChunkSpan(header, headerOffset, dataOffset));

                stream.Position = payloadEndOffset;
            }

            if (chunks.Count < 2)
                return false;

            uint? version = TryReadVersion(stream, chunks);
            WowFileDetection detection = WowFileDetector.Detect(sourcePath, chunks, version);
            MapFileKind kind = ToMapFileKind(detection.Kind);
            if (kind is not (MapFileKind.Adt or MapFileKind.AdtV23 or MapFileKind.AdtV23Error or MapFileKind.AdtTex or MapFileKind.AdtObj or MapFileKind.AdtLod))
                return false;

            summary = new MapFileSummary(
                sourcePath,
                kind,
                version,
                chunks.Select(static chunk => new MapChunkLocation(chunk.Header.Id, chunk.Header.Size, chunk.HeaderOffset, chunk.DataOffset)).ToArray());
            return true;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static bool TryReadAlphaWdtSummary(Stream stream, string sourcePath, out MapFileSummary? summary)
    {
        summary = null;

        if (!stream.CanSeek || !sourcePath.EndsWith(".wdt", StringComparison.OrdinalIgnoreCase))
            return false;

        long previousPosition = stream.Position;
        try
        {
            if (!TryReadChunkLocation(stream, 0, MapChunkIds.Mver, out MapChunkLocation mver, out long nextOffset))
                return false;

            if (!TryReadChunkLocation(stream, nextOffset, MapChunkIds.Mphd, out MapChunkLocation mphd, out nextOffset))
                return false;

            if (!TryReadChunkLocation(stream, nextOffset, MapChunkIds.Main, out MapChunkLocation main, out _))
                return false;

            uint? version = ReadVersion(stream, mver);
            List<MapChunkLocation> chunks = [mver, mphd, main];
            if (TryReadAlphaReferencedChunkLocation(stream, mphd, 4, MapChunkIds.Mdnm, out MapChunkLocation mdnm))
                chunks.Add(mdnm);

            if (TryReadAlphaReferencedChunkLocation(stream, mphd, 12, MapChunkIds.Monm, out MapChunkLocation monm))
                chunks.Add(monm);

            summary = new MapFileSummary(sourcePath, MapFileKind.Wdt, version, chunks.ToArray());
            return true;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static bool TryReadAlphaReferencedChunkLocation(Stream stream, MapChunkLocation mphd, int mphdPayloadOffset, FourCC expectedId, out MapChunkLocation chunk)
    {
        chunk = default;

        if (mphdPayloadOffset < 0 || mphd.Size < mphdPayloadOffset + sizeof(int))
            return false;

        long previousPosition = stream.Position;
        try
        {
            stream.Position = mphd.DataOffset + mphdPayloadOffset;
            Span<byte> offsetBytes = stackalloc byte[sizeof(int)];
            stream.ReadExactly(offsetBytes);
            int chunkOffset = BinaryPrimitives.ReadInt32LittleEndian(offsetBytes);
            if (chunkOffset < 0 || chunkOffset > stream.Length - ChunkHeader.SizeInBytes)
                return false;

            return TryReadChunkLocation(stream, chunkOffset, expectedId, out chunk, out _);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static bool TryReadChunkLocation(Stream stream, long headerOffset, FourCC expectedId, out MapChunkLocation chunk, out long nextOffset)
    {
        chunk = default;
        nextOffset = headerOffset;

        if (headerOffset < 0 || headerOffset > stream.Length - ChunkHeader.SizeInBytes)
            return false;

        long previousPosition = stream.Position;
        try
        {
            stream.Position = headerOffset;

            Span<byte> headerBytes = stackalloc byte[ChunkHeader.SizeInBytes];
            stream.ReadExactly(headerBytes);
            if (!ChunkHeaderReader.TryRead(headerBytes, out ChunkHeader header) || header.Id != expectedId)
                return false;

            long dataOffset = stream.Position;
            long payloadEndOffset = unchecked(dataOffset + (long)header.Size);
            if (payloadEndOffset > stream.Length)
                return false;

            chunk = new MapChunkLocation(header.Id, header.Size, headerOffset, dataOffset);
            nextOffset = payloadEndOffset;
            if ((header.Size & 1) != 0 && nextOffset < stream.Length)
                nextOffset++;

            return true;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static uint? ReadVersion(Stream stream, MapChunkLocation chunk)
    {
        if (chunk.Size < sizeof(uint))
            return null;

        long previousPosition = stream.Position;
        try
        {
            stream.Position = chunk.DataOffset;
            Span<byte> bytes = stackalloc byte[sizeof(uint)];
            stream.ReadExactly(bytes);
            return BinaryPrimitives.ReadUInt32LittleEndian(bytes);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static uint? TryReadVersion(Stream stream, IReadOnlyList<ChunkSpan> chunks)
    {
        if (chunks.Count == 0)
            return null;

        if (chunks[0].Header.Id != MapChunkIds.Mver && chunks[0].Header.Id != MapChunkIds.Ahdr)
            return null;

        return ChunkedFileReader.TryReadUInt32(stream, chunks[0]);
    }

    private static MapFileKind ToMapFileKind(WowFileKind kind)
    {
        return kind switch
        {
            WowFileKind.Wdt => MapFileKind.Wdt,
            WowFileKind.Adt => MapFileKind.Adt,
            WowFileKind.AdtV23 => MapFileKind.AdtV23,
            WowFileKind.AdtV23Error => MapFileKind.AdtV23Error,
            WowFileKind.AdtTex => MapFileKind.AdtTex,
            WowFileKind.AdtObj => MapFileKind.AdtObj,
            WowFileKind.AdtLod => MapFileKind.AdtLod,
            _ => MapFileKind.Unknown,
        };
    }
}