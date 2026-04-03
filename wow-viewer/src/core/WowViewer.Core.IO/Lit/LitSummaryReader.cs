using System.Buffers.Binary;
using WowViewer.Core.Lit;

namespace WowViewer.Core.IO.Lit;

public static class LitSummaryReader
{
    private const int HeaderSize = 8;
    private const int LightListEntrySize = 64;

    public static LitSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static LitSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        if (!stream.CanSeek)
            throw new ArgumentException("LIT summary reading requires a seekable stream.", nameof(stream));

        if (stream.Length < HeaderSize)
            throw new InvalidDataException("LIT file is too short to contain a header.");

        long previousPosition = stream.Position;
        try
        {
            stream.Position = 0;
            byte[] header = new byte[HeaderSize];
            stream.ReadExactly(header);

            uint versionNumber = BinaryPrimitives.ReadUInt32LittleEndian(header.AsSpan(0, 4));
            int lightCount = BinaryPrimitives.ReadInt32LittleEndian(header.AsSpan(4, 4));
            bool usesSinglePartialEntry = lightCount == -1;
            int listEntryCount = Math.Max(lightCount, 0);

            long requiredListBytes = (long)listEntryCount * LightListEntrySize;
            if (stream.Length < HeaderSize + requiredListBytes)
            {
                throw new InvalidDataException(
                    $"LIT file declares {listEntryCount} light-list entries but only contains {stream.Length - HeaderSize} bytes after the header.");
            }

            bool hasDefaultFirstEntry = false;
            int namedEntryCount = 0;

            for (int entryIndex = 0; entryIndex < listEntryCount; entryIndex++)
            {
                byte[] entry = new byte[LightListEntrySize];
                stream.ReadExactly(entry);

                int chunkX = BinaryPrimitives.ReadInt32LittleEndian(entry.AsSpan(0, 4));
                int chunkY = BinaryPrimitives.ReadInt32LittleEndian(entry.AsSpan(4, 4));
                int chunkRadius = BinaryPrimitives.ReadInt32LittleEndian(entry.AsSpan(8, 4));
                if (entryIndex == 0 && chunkX == -1 && chunkY == -1 && chunkRadius == -1)
                    hasDefaultFirstEntry = true;

                if (HasMeaningfulName(entry.AsSpan(0x20, 0x20)))
                    namedEntryCount++;
            }

            int remainingPayloadBytes = checked((int)(stream.Length - stream.Position));
            return new LitSummary(
                sourcePath,
                versionNumber,
                lightCount,
                listEntryCount,
                usesSinglePartialEntry,
                hasDefaultFirstEntry,
                namedEntryCount,
                remainingPayloadBytes);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static bool HasMeaningfulName(ReadOnlySpan<byte> bytes)
    {
        foreach (byte value in bytes)
        {
            if (value != 0 && value != 0xFD)
                return true;
        }

        return false;
    }
}