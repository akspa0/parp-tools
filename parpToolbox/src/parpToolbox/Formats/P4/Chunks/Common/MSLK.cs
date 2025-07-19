namespace ParpToolbox.Formats.P4.Chunks.Common;

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;

/// <summary>
/// Ported MSLK entry from legacy Core.v2 with minimal dependencies removed.
/// Represents a single link object in a PM4 scene graph.
/// </summary>
public sealed class MslkEntry : IBinarySerializable
{
    // ---- Raw fields (one-to-one with on-disk layout) --------------------
    public byte Unknown_0x00 { get; private set; }
    public byte Unknown_0x01 { get; private set; }
    public ushort Unknown_0x02 { get; private set; }
    public uint Unknown_0x04 { get; private set; }
    public int  MspiFirstIndex { get; private set; }   // 24-bit signed
    public byte MspiIndexCount { get; private set; }
    public uint LinkIdRaw { get; private set; }        // 32-bit tile coordinate field
    public ushort Unknown_0x10 { get; private set; }
    public ushort Unknown_0x12 { get; private set; }

    public const int StructSize = 20;

    /// <summary>
    /// High-word parent/container identifier used for hierarchical grouping.
    /// </summary>
    public uint ParentIndex => Unknown_0x04;

    /// <summary>
    /// Authoritative group-key for geometry objects. Equals low-word of MSUR.SurfaceKey.
    /// </summary>
    public ushort ReferenceIndex => Unknown_0x10;

    /// <summary>
    /// High 16 bits of Unknown_0x10 for grouping analysis.
    /// </summary>
    public ushort ReferenceIndexHigh => (ushort)(Unknown_0x10 >> 16);

    /// <summary>
    /// Low 16 bits of Unknown_0x10 for grouping analysis.
    /// </summary>
    public ushort ReferenceIndexLow => (ushort)(Unknown_0x10 & 0xFFFF);

    /// <summary>
    /// Padding field from LinkIdRaw (should always be 0xFFFF).
    /// </summary>
    public ushort LinkIdPadding => (ushort)(LinkIdRaw >> 16);

    /// <summary>
    /// Tile Y coordinate decoded from LinkIdRaw using legacy logic (YYXX pattern).
    /// </summary>
    public byte LinkIdTileY => (byte)((LinkIdRaw >> 8) & 0xFF);

    /// <summary>
    /// Tile X coordinate decoded from LinkIdRaw using legacy logic (YYXX pattern).
    /// </summary>
    public byte LinkIdTileX => (byte)(LinkIdRaw & 0xFF);

    /// <summary>
    /// Attempts to decode tile coordinates from LinkIdRaw using legacy LinkIdDecoder logic.
    /// Returns true if the pattern matches (high 16 bits == 0xFFFF), false otherwise.
    /// </summary>
    public bool TryDecodeTileCoordinates(out int tileX, out int tileY)
    {
        tileX = tileY = 0;
        ushort high = (ushort)(LinkIdRaw >> 16);
        ushort low = (ushort)(LinkIdRaw & 0xFFFF);
        if (high != 0xFFFF) return false; // unknown schema

        // low word stored as YYXX (little-endian). Split bytes.
        byte yy = (byte)(low >> 8);
        byte xx = (byte)(low & 0xFF);
        tileX = xx;
        tileY = yy;
        return true;
    }

    /// <summary>
    /// Returns true if this entry has valid tile coordinates (LinkIdRaw follows 0xFFFFYYXX pattern).
    /// </summary>
    public bool HasValidTileCoordinates => (LinkIdRaw >> 16) == 0xFFFF;

    /// <summary>
    /// Composite tile coordinate key constructed from Y and X tile positions.
    /// </summary>
    public ushort TileCoordinate => (ushort)((LinkIdTileY << 8) | LinkIdTileX);

    /// <summary>
    /// Legacy compatibility: composite key matching MSUR SurfaceKey low 16 bits.
    /// </summary>
    public ushort LinkSubKey => TileCoordinate;

    // ---- Convenience decoded accessors ----------------------------------
    public bool HasGeometry => MspiFirstIndex >= 0 && MspiIndexCount > 0;

    // ----------------------------------------------------------------------
    public void LoadBinaryData(byte[] inData)
    {
        using var ms = new MemoryStream(inData ?? throw new ArgumentNullException(nameof(inData)));
        using var br = new BinaryReader(ms);
        Load(br);
    }

    public void Load(BinaryReader br)
    {
        if (br.BaseStream.Position + StructSize > br.BaseStream.Length)
            throw new EndOfStreamException("MSLK entry truncated");

        Unknown_0x00 = br.ReadByte();
        Unknown_0x01 = br.ReadByte();
        Unknown_0x02 = br.ReadUInt16();
        Unknown_0x04 = br.ReadUInt32();
        MspiFirstIndex = ReadInt24(br);
        MspiIndexCount = br.ReadByte();
        LinkIdRaw = br.ReadUInt32();
        Unknown_0x10 = br.ReadUInt16();
        Unknown_0x12 = br.ReadUInt16();
    }

    public byte[] Serialize(long offset = 0)
    {
        using var ms = new MemoryStream(StructSize);
        using var bw = new BinaryWriter(ms);
        Write(bw);
        return ms.ToArray();
    }

    public uint GetSize() => StructSize;

    private void Write(BinaryWriter bw)
    {
        bw.Write(Unknown_0x00);
        bw.Write(Unknown_0x01);
        bw.Write(Unknown_0x02);
        bw.Write(Unknown_0x04);
        WriteInt24(bw, MspiFirstIndex);
        bw.Write(MspiIndexCount);
        bw.Write(LinkIdRaw);
        bw.Write(Unknown_0x10);
        bw.Write(Unknown_0x12);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int ReadInt24(BinaryReader br)
    {
        int value = br.ReadByte() | (br.ReadByte() << 8) | (br.ReadByte() << 16);
        // sign-extend if MSB of byte2 set
        if ((value & 0x00800000) != 0)
            value |= unchecked((int)0xFF000000);
        return value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void WriteInt24(BinaryWriter bw, int value)
    {
        bw.Write((byte)(value & 0xFF));
        bw.Write((byte)((value >> 8) & 0xFF));
        bw.Write((byte)((value >> 16) & 0xFF));
    }
}

/// <summary>
/// Ported MSLK chunk – container for <see cref="MslkEntry"/> records.
/// </summary>
public sealed class MslkChunk : IIffChunk, IBinarySerializable
{
    public const string Signature = "MSLK";

    private readonly List<MslkEntry> _entries = new();
    public IReadOnlyList<MslkEntry> Entries => _entries;

    public string GetSignature() => Signature;

    public uint GetSize() => (uint)(_entries.Count * MslkEntry.StructSize);

    public void LoadBinaryData(byte[] inData)
    {
        using var ms = new MemoryStream(inData ?? throw new ArgumentNullException(nameof(inData)));
        using var br = new BinaryReader(ms);
        Load(br);
    }

    public void Load(BinaryReader br)
    {
        var remaining = br.BaseStream.Length - br.BaseStream.Position;
        int count = (int)(remaining / MslkEntry.StructSize);
        for (int i = 0; i < count; i++)
        {
            var e = new MslkEntry();
            e.Load(br);
            _entries.Add(e);
        }
    }

    public byte[] Serialize(long offset = 0)
    {
        using var ms = new MemoryStream(_entries.Count * MslkEntry.StructSize);
        using var bw = new BinaryWriter(ms);
        foreach (var e in _entries) e.Serialize(); // write via Serialize helper
        return ms.ToArray();
    }
}
