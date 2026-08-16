using WowViewer.Core.IO.Chunked;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.AssetReferences;

/// <summary>
/// Pulls the asset paths a world object claims: the doodad models it places and the textures its
/// materials name.
/// </summary>
/// <remarks>
/// The existing summary readers report counts, not paths, so this reads the same chunks through the
/// same shared root-chunk helpers rather than duplicating any parsing. MODN and MOTX are both
/// null-terminated string blobs; MODD entries index into MODN by byte offset.
/// </remarks>
public static class WmoReferenceExtractor
{
    private const int ModdEntrySize = 40;
    private const uint ModdNameOffsetMask = 0x00FFFFFFu;

    /// <summary>
    /// Extracts every doodad and texture path named by a world object root file.
    /// Returns paths as authored, without normalisation, so a case or separator difference stays
    /// visible to later comparison rather than being silently repaired here.
    /// </summary>
    public static IReadOnlyList<(AssetReferenceKind Kind, string Path)> Extract(Stream stream, string sourcePath)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        (uint? _, IReadOnlyList<ChunkSpan> chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        List<(AssetReferenceKind, string)> references = [];

        AddDoodadNames(stream, chunks, references);
        AddTextureNames(stream, chunks, references);
        return references;
    }

    private static void AddDoodadNames(
        Stream stream,
        IReadOnlyList<ChunkSpan> chunks,
        List<(AssetReferenceKind, string)> references)
    {
        // A world object with no doodads is ordinary, so both chunks are optional here even though
        // the summary reader requires them.
        byte[]? modn = WmoRootReaderCommon.TryReadChunkPayload(stream, chunks, WmoChunkIds.Modn);
        byte[]? modd = WmoRootReaderCommon.TryReadChunkPayload(stream, chunks, WmoChunkIds.Modd);
        if (modn is null || modd is null || modd.Length < ModdEntrySize)
            return;

        HashSet<string> seen = new(StringComparer.OrdinalIgnoreCase);
        int entryCount = modd.Length / ModdEntrySize;
        for (int index = 0; index < entryCount; index++)
        {
            uint nameOffset = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(
                modd.AsSpan(index * ModdEntrySize, 4)) & ModdNameOffsetMask;
            string name = WmoRootReaderCommon.ResolveStringAtOffset(modn, nameOffset);
            if (name.Length == 0 || !seen.Add(name))
                continue;

            references.Add((AssetReferenceKind.PlacedDoodad, name));
        }
    }

    private static void AddTextureNames(
        Stream stream,
        IReadOnlyList<ChunkSpan> chunks,
        List<(AssetReferenceKind, string)> references)
    {
        byte[]? motx = WmoRootReaderCommon.TryReadChunkPayload(stream, chunks, WmoChunkIds.Motx);
        if (motx is null || motx.Length == 0)
            return;

        HashSet<string> seen = new(StringComparer.OrdinalIgnoreCase);
        foreach (string entry in EnumerateNullTerminatedStrings(motx))
        {
            if (!seen.Add(entry))
                continue;

            references.Add((AssetReferenceKind.WorldObjectTexture, entry));
        }
    }

    /// <summary>
    /// Walks a null-terminated string blob. Padding runs of zeroes produce no entries, and a final
    /// unterminated run is still yielded so a truncated table does not silently lose its last name.
    /// </summary>
    private static IEnumerable<string> EnumerateNullTerminatedStrings(byte[] payload)
    {
        int start = -1;
        for (int index = 0; index < payload.Length; index++)
        {
            if (payload[index] != 0)
            {
                if (start < 0)
                    start = index;

                continue;
            }

            if (start >= 0)
            {
                yield return System.Text.Encoding.UTF8.GetString(payload, start, index - start);
                start = -1;
            }
        }

        if (start >= 0)
            yield return System.Text.Encoding.UTF8.GetString(payload, start, payload.Length - start);
    }
}
