using System;
using System.IO;
using ArcaneFileParser.Core.Common;

namespace ArcaneFileParser.Core.Formats.ADT.Chunks;

/// <summary>
/// Map Header chunk containing flags and offsets for ADT files.
/// </summary>
public class MhdrChunk : ChunkBase
{
    public override string ChunkId => "MHDR";

    /// <summary>
    /// Flags indicating various ADT properties.
    /// </summary>
    [Flags]
    public enum AdtFlags : uint
    {
        None = 0x0,
        HasMCCV = 0x1,          // Contains vertex colors (MCCV chunks)
        HasMH2O = 0x2,          // Contains water data (MH2O chunks)
        HasMCSE = 0x4,          // Contains sound emitters (MCSE chunks)
        HasMCLQ = 0x8,          // Contains legacy liquid data (MCLQ chunks)
        UseMCLQ = 0x10,         // Use legacy liquid data (MCLQ) instead of MH2O
        IsBigAlpha = 0x20,      // Use 4096 bytes for MCAL instead of 2048
        IsHighRes = 0x40,       // High resolution heightmap
        IsUnknown80 = 0x80,     // Unknown flag
        IsUnknown100 = 0x100,   // Unknown flag
        IsUnknown200 = 0x200,   // Unknown flag
        HasMCSH = 0x400,        // Contains shadow maps (MCSH chunks)
        IsMapped = 0x800,       // Is mapped (impass data)
        HasMCCV2 = 0x1000,      // Contains vertex colors v2 (MCCV chunks)
        HasMCLV = 0x2000,       // Contains light values (MCLV chunks)
        HasMCAL2 = 0x4000,      // Contains alpha maps v2 (MCAL chunks)
        HasMCLW = 0x8000,       // Contains light weights (MCLW chunks)
    }

    /// <summary>
    /// Gets the flags indicating various ADT properties.
    /// </summary>
    public AdtFlags Flags { get; private set; }

    /// <summary>
    /// Gets the offset to the MCIN chunk.
    /// </summary>
    public uint McInOffset { get; private set; }

    /// <summary>
    /// Gets the offset to the MTEX chunk.
    /// </summary>
    public uint MtexOffset { get; private set; }

    /// <summary>
    /// Gets the offset to the MMDX chunk.
    /// </summary>
    public uint MmdxOffset { get; private set; }

    /// <summary>
    /// Gets the offset to the MMID chunk.
    /// </summary>
    public uint MmidOffset { get; private set; }

    /// <summary>
    /// Gets the offset to the MWMO chunk.
    /// </summary>
    public uint MwmoOffset { get; private set; }

    /// <summary>
    /// Gets the offset to the MWID chunk.
    /// </summary>
    public uint MwidOffset { get; private set; }

    /// <summary>
    /// Gets the offset to the MDDF chunk.
    /// </summary>
    public uint MddfOffset { get; private set; }

    /// <summary>
    /// Gets the offset to the MODF chunk.
    /// </summary>
    public uint ModfOffset { get; private set; }

    /// <summary>
    /// Gets the offset to the MH2O chunk.
    /// </summary>
    public uint Mh2oOffset { get; private set; }

    /// <summary>
    /// Gets the offset to the MTXF chunk.
    /// </summary>
    public uint MtxfOffset { get; private set; }

    public override void Parse(BinaryReader reader, uint size)
    {
        if (size != 64)
        {
            throw new InvalidDataException($"MHDR chunk size must be 64 bytes, found {size} bytes");
        }

        Flags = (AdtFlags)reader.ReadUInt32();
        McInOffset = reader.ReadUInt32();
        MtexOffset = reader.ReadUInt32();
        MmdxOffset = reader.ReadUInt32();
        MmidOffset = reader.ReadUInt32();
        MwmoOffset = reader.ReadUInt32();
        MwidOffset = reader.ReadUInt32();
        MddfOffset = reader.ReadUInt32();
        ModfOffset = reader.ReadUInt32();
        Mh2oOffset = reader.ReadUInt32();
        MtxfOffset = reader.ReadUInt32();

        // Skip remaining bytes (padding)
        reader.BaseStream.Position += 24;
    }

    protected override void WriteContent(BinaryWriter writer)
    {
        writer.Write((uint)Flags);
        writer.Write(McInOffset);
        writer.Write(MtexOffset);
        writer.Write(MmdxOffset);
        writer.Write(MmidOffset);
        writer.Write(MwmoOffset);
        writer.Write(MwidOffset);
        writer.Write(MddfOffset);
        writer.Write(ModfOffset);
        writer.Write(Mh2oOffset);
        writer.Write(MtxfOffset);

        // Write padding
        for (int i = 0; i < 24; i++)
        {
            writer.Write((byte)0);
        }
    }

    public override string ToHumanReadable()
    {
        return $"Flags: {Flags}\n" +
               $"MCIN Offset: 0x{McInOffset:X8}\n" +
               $"MTEX Offset: 0x{MtexOffset:X8}\n" +
               $"MMDX Offset: 0x{MmdxOffset:X8}\n" +
               $"MMID Offset: 0x{MmidOffset:X8}\n" +
               $"MWMO Offset: 0x{MwmoOffset:X8}\n" +
               $"MWID Offset: 0x{MwidOffset:X8}\n" +
               $"MDDF Offset: 0x{MddfOffset:X8}\n" +
               $"MODF Offset: 0x{ModfOffset:X8}\n" +
               $"MH2O Offset: 0x{Mh2oOffset:X8}\n" +
               $"MTXF Offset: 0x{MtxfOffset:X8}";
    }
} 