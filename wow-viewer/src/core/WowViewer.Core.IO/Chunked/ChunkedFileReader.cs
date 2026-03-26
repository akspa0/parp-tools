using System.Buffers.Binary;
using WowViewer.Core.Chunks;

namespace WowViewer.Core.IO.Chunked;

public static class ChunkedFileReader
{
    public static IReadOnlyList<ChunkSpan> ReadTopLevelChunks(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return ReadTopLevelChunks(stream);
    }

    public static IReadOnlyList<ChunkSpan> ReadTopLevelChunks(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (!stream.CanSeek)
            throw new ArgumentException("Chunked file reading requires a seekable stream.", nameof(stream));

        List<ChunkSpan> chunks = [];
        Span<byte> headerBytes = stackalloc byte[ChunkHeader.SizeInBytes];
        stream.Position = 0;

        while (stream.Position <= stream.Length - ChunkHeader.SizeInBytes)
        {
            long headerOffset = stream.Position;
            stream.ReadExactly(headerBytes);

            if (!ChunkHeaderReader.TryRead(headerBytes, out ChunkHeader header))
                throw new InvalidDataException($"Could not decode chunk header at offset {headerOffset}.");

            long dataOffset = stream.Position;
            long endOffset = checked(dataOffset + header.Size);
            if (endOffset > stream.Length)
                throw new InvalidDataException($"Chunk {header.Id} at offset {headerOffset} overruns the stream length.");

            chunks.Add(new ChunkSpan(header, headerOffset, dataOffset));
            stream.Position = endOffset;
        }

        return chunks;
    }

    public static uint? TryReadUInt32(Stream stream, ChunkSpan chunk)
    {
        ArgumentNullException.ThrowIfNull(stream);

        if (!stream.CanSeek)
            throw new ArgumentException("Chunk payload reading requires a seekable stream.", nameof(stream));

        if (chunk.Header.Size < sizeof(uint))
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
}