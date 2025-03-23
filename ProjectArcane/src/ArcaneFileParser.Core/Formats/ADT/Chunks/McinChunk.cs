using System;
using System.IO;
using ArcaneFileParser.Core.Common;

namespace ArcaneFileParser.Core.Formats.ADT.Chunks;

/// <summary>
/// Map Chunk Index chunk containing offsets and sizes for MCNK chunks.
/// </summary>
public class McinChunk : ChunkBase
{
    public override string ChunkId => "MCIN";

    /// <summary>
    /// The expected size of the MCIN chunk (256 entries * 16 bytes per entry).
    /// </summary>
    private const uint ExpectedSize = 256 * 16;

    /// <summary>
    /// Structure representing a single map chunk entry.
    /// </summary>
    public struct MapChunkEntry
    {
        /// <summary>
        /// Gets the offset to the MCNK chunk.
        /// </summary>
        public uint Offset { get; set; }

        /// <summary>
        /// Gets the size of the MCNK chunk.
        /// </summary>
        public uint Size { get; set; }

        /// <summary>
        /// Gets the flags for this chunk.
        /// </summary>
        public uint Flags { get; set; }

        /// <summary>
        /// Gets the async ID for this chunk.
        /// </summary>
        public uint AsyncId { get; set; }
    }

    /// <summary>
    /// Gets the array of map chunk entries (16x16 grid).
    /// </summary>
    public MapChunkEntry[] Entries { get; } = new MapChunkEntry[256];

    public override void Parse(BinaryReader reader, uint size)
    {
        if (size != ExpectedSize)
        {
            throw new InvalidDataException($"MCIN chunk size must be {ExpectedSize} bytes, found {size} bytes");
        }

        // Read all chunk entries
        for (int i = 0; i < 256; i++)
        {
            Entries[i] = new MapChunkEntry
            {
                Offset = reader.ReadUInt32(),
                Size = reader.ReadUInt32(),
                Flags = reader.ReadUInt32(),
                AsyncId = reader.ReadUInt32()
            };
        }
    }

    protected override void WriteContent(BinaryWriter writer)
    {
        // Write all chunk entries
        foreach (var entry in Entries)
        {
            writer.Write(entry.Offset);
            writer.Write(entry.Size);
            writer.Write(entry.Flags);
            writer.Write(entry.AsyncId);
        }
    }

    /// <summary>
    /// Gets the chunk entry at the specified coordinates.
    /// </summary>
    /// <param name="x">The X coordinate (0-15).</param>
    /// <param name="y">The Y coordinate (0-15).</param>
    /// <returns>The chunk entry at the specified coordinates.</returns>
    public MapChunkEntry GetEntry(int x, int y)
    {
        if (x < 0 || x > 15 || y < 0 || y > 15)
            throw new ArgumentOutOfRangeException($"Coordinates must be between 0 and 15 (got {x},{y})");

        int index = y * 16 + x;
        return Entries[index];
    }

    public override string ToHumanReadable()
    {
        int validChunks = 0;
        for (int i = 0; i < 256; i++)
        {
            if (Entries[i].Size > 0)
                validChunks++;
        }

        return $"Valid Chunks: {validChunks}/256";
    }
} 