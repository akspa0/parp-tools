using System.Buffers.Binary;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Reads MCRF (Map Chunk References) subchunks from within MCNK payloads.
/// MCRF contains uint32 indices: first N are doodad (M2) indices, remainder are WMO indices.
/// The counts come from the MCNK header (nDoodads, nMapObjRefs).
/// </summary>
public static class AdtMcrfReader
{
    /// <summary>
    /// Parses MCRF data from an MCNK subchunk payload (without FourCC header).
    /// </summary>
    public static AdtMcrfData Read(byte[] payload, int doodadCount, int wmoCount)
    {
        if (payload.Length < 4)
            return new AdtMcrfData();

        int totalIndices = payload.Length / 4;
        int[] allIndices = new int[totalIndices];
        for (int i = 0; i < totalIndices; i++)
        {
            allIndices[i] = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(i * 4, 4));
        }

        int actualDoodadCount = Math.Min(doodadCount, totalIndices);
        int actualWmoCount = Math.Min(wmoCount, totalIndices - actualDoodadCount);

        int[] doodadIndices = actualDoodadCount > 0
            ? allIndices.AsSpan(0, actualDoodadCount).ToArray()
            : Array.Empty<int>();

        int[] wmoIndices = actualWmoCount > 0
            ? allIndices.AsSpan(actualDoodadCount, actualWmoCount).ToArray()
            : Array.Empty<int>();

        return new AdtMcrfData
        {
            DoodadIndices = doodadIndices,
            WmoIndices = wmoIndices
        };
    }

    /// <summary>
    /// Locates MCRF subchunk data offset within an MCNK payload.
    /// Returns the offset to the MCRF payload (after FourCC+size), or -1 if not found.
    /// </summary>
    public static int LocateMcrfOffset(ReadOnlySpan<byte> payload)
    {
        const int subchunkOffset = 0x80;
        int position = subchunkOffset;

        while (position <= payload.Length - ChunkHeader.SizeInBytes)
        {
            if (!ChunkHeaderReader.TryRead(payload.Slice(position, ChunkHeader.SizeInBytes), out ChunkHeader header))
                break;

            int declaredSize = unchecked((int)header.Size);
            if (declaredSize < 0 || declaredSize > payload.Length)
                break;
            long nextOffset = (long)position + ChunkHeader.SizeInBytes + declaredSize;
            if (nextOffset > payload.Length)
                break;

            if (header.Id == AdtChunkIds.Mcrf)
            {
                if (header.Size < 4)
                    return -1;
                return position + ChunkHeader.SizeInBytes;
            }

            position = unchecked((int)nextOffset);
            if (position < 0) break;
        }

        return -1;
    }
}

/// <summary>
/// Parsed MCRF reference data for a single terrain chunk.
/// </summary>
public sealed class AdtMcrfData
{
    /// <summary>Doodad (M2) indices referenced by this chunk.</summary>
    public int[] DoodadIndices { get; init; } = Array.Empty<int>();

    /// <summary>WMO indices referenced by this chunk.</summary>
    public int[] WmoIndices { get; init; } = Array.Empty<int>();
}
