using System.Collections.Generic;
using System.IO;
using System.Text;
using ArcaneFileParser.Core.Common;

namespace ArcaneFileParser.Core.Formats.ADT.Chunks;

/// <summary>
/// Map WMO Filename Offsets chunk containing offsets into the MWMO chunk.
/// </summary>
public class MwidChunk : ChunkBase
{
    public override string ChunkId => "MWID";

    /// <summary>
    /// Gets the list of offsets into the MWMO chunk.
    /// </summary>
    public List<uint> Offsets { get; } = new();

    /// <summary>
    /// Gets the number of WMO filename offsets.
    /// </summary>
    public int Count => Offsets.Count;

    public override void Parse(BinaryReader reader, uint size)
    {
        var startPosition = reader.BaseStream.Position;
        var endPosition = startPosition + size;

        // Clear existing data
        Offsets.Clear();

        // Each offset is 4 bytes
        var offsetCount = size / 4;

        // Read all offsets
        for (int i = 0; i < offsetCount; i++)
        {
            Offsets.Add(reader.ReadUInt32());
        }
    }

    protected override void WriteContent(BinaryWriter writer)
    {
        // Write each offset
        foreach (var offset in Offsets)
        {
            writer.Write(offset);
        }
    }

    /// <summary>
    /// Gets an offset by index.
    /// </summary>
    /// <param name="index">Index of the offset.</param>
    /// <returns>The offset if found, 0 otherwise.</returns>
    public uint GetOffset(int index)
    {
        if (index < 0 || index >= Offsets.Count)
            return 0;

        return Offsets[index];
    }

    /// <summary>
    /// Adds a new offset to the list.
    /// </summary>
    /// <param name="offset">The offset to add.</param>
    public void AddOffset(uint offset)
    {
        Offsets.Add(offset);
    }

    public override string ToHumanReadable()
    {
        var builder = new StringBuilder();
        builder.AppendLine($"Offset Count: {Offsets.Count}");
        
        builder.AppendLine("\nOffset List:");
        for (int i = 0; i < Offsets.Count; i++)
        {
            builder.AppendLine($"[{i}] 0x{Offsets[i]:X8}");
        }

        return builder.ToString();
    }
} 