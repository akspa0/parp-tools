using System.Buffers.Binary;
using WowViewer.Core.Chunks;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Reads the MTXF (Map Texture Flags) chunk from an ADT root file.
/// MTXF contains per-texture flags (animated, etc.) parallel to the MTEX texture list.
/// </summary>
public static class AdtMtxfReader
{
    /// <summary>
    /// Reads MTXF flags from a root ADT file.
    /// Returns null if the chunk is not present.
    /// </summary>
    public static AdtMtxfData? Read(string adtPath)
    {
        using FileStream stream = File.OpenRead(adtPath);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, Path.GetFullPath(adtPath));
        if (fileSummary.Kind != MapFileKind.Adt)
            return null;

        return Read(stream, fileSummary);
    }

    /// <summary>
    /// Reads MTXF flags given a pre-scanned file summary.
    /// </summary>
    public static AdtMtxfData? Read(Stream stream, MapFileSummary fileSummary)
    {
        if (!fileSummary.HasChunk(MapChunkIds.Mtxf))
            return null;

        MapChunkLocation mtxfChunk = fileSummary.Chunks.First(c => c.Id == MapChunkIds.Mtxf);
        long previousPosition = stream.Position;
        try
        {
            stream.Position = mtxfChunk.DataOffset;
            byte[] payload = new byte[mtxfChunk.Size];
            stream.ReadExactly(payload);

            return Parse(payload);
        }
        catch
        {
            return null;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    /// <summary>
    /// Parses MTXF payload bytes.
    /// Each entry is 2 bytes (uint16) where bit 0 = animated.
    /// Some formats have an extended array of 4-byte transform IDs after the flags.
    /// </summary>
    public static AdtMtxfData? Parse(byte[] payload)
    {
        if (payload.Length < 2)
            return null;

        int entryCount = payload.Length / 2;
        ushort[] flags = new ushort[entryCount];
        for (int i = 0; i < entryCount; i++)
        {
            flags[i] = BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(i * 2, 2));
        }

        // Check if there is an extended transform ID array
        // The extended array has the same count as flags but 4 bytes each
        int[]? transformIds = null;
        int extendedSize = entryCount * 4;
        int totalExtendedSize = (entryCount * 2) + extendedSize;
        if (payload.Length >= totalExtendedSize)
        {
            transformIds = new int[entryCount];
            int offset = entryCount * 2;
            for (int i = 0; i < entryCount; i++)
            {
                transformIds[i] = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(offset + (i * 4), 4));
            }
        }

        return new AdtMtxfData
        {
            Flags = flags,
            TransformIds = transformIds
        };
    }
}

/// <summary>
/// MTXF texture flags parallel to the MTEX texture list.
/// </summary>
public sealed class AdtMtxfData
{
    /// <summary>Per-texture flags where bit 0 = animated.</summary>
    public ushort[] Flags { get; init; } = Array.Empty<ushort>();

    /// <summary>Optional per-texture transform IDs (extended format).</summary>
    public int[]? TransformIds { get; init; }
}
