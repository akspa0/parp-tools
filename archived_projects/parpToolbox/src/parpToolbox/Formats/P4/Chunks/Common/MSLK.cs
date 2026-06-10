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
    public byte Flags_0x00 { get; private set; }         // Still under investigation
    public byte Type_0x01 { get; private set; }          // Still under investigation
    public ushort SortKey_0x02 { get; private set; }       // Still under investigation
    public uint ParentId { get; private set; }         // Parent/container identifier for hierarchical grouping
    public int MspiFirstIndex { get; private set; }    // 24-bit signed index into MSPI
    public byte MspiIndexCount { get; private set; }
    public uint TileCoordsRaw { get; private set; }      // Raw 32-bit tile coordinate field (0xFFFFYYXX)
    public ushort SurfaceRefIndex { get; private set; }  // Reference to a surface (e.g., in MSUR)
    public ushort Unknown_0x12 { get; private set; }     // Still under investigation

    public const int StructSize = 20;

    /// <summary>
    /// High-word parent/container identifier used for hierarchical grouping.
    /// </summary>
    public uint ParentIndex => ParentId;

    /// <summary>
    /// Authoritative group-key for geometry objects. Corresponds to a surface reference.
    /// </summary>
    public ushort ReferenceIndex => SurfaceRefIndex;

    /// <summary>
    /// High 16 bits of the surface reference index for grouping analysis.
    /// </summary>
    public ushort ReferenceIndexHigh => (ushort)(SurfaceRefIndex >> 16);

    /// <summary>
    /// Low 16 bits of the surface reference index for grouping analysis.
    /// </summary>
    public ushort ReferenceIndexLow => (ushort)(SurfaceRefIndex & 0xFFFF);

    /// <summary>
    /// Padding field from the raw tile coordinates (should always be 0xFFFF).
    /// </summary>
    public ushort LinkIdPadding => (ushort)(TileCoordsRaw >> 16);

    /// <summary>
    /// Tile Y coordinate decoded from the raw tile coordinates using the YYXX pattern.
    /// </summary>
    public byte LinkIdTileY => (byte)((TileCoordsRaw >> 8) & 0xFF);

    /// <summary>
    /// Tile X coordinate decoded from the raw tile coordinates using the YYXX pattern.
    /// </summary>
    public byte LinkIdTileX => (byte)(TileCoordsRaw & 0xFF);

    /// <summary>
    /// Attempts to decode tile coordinates from the raw tile coordinate data.
    /// Returns true if the pattern matches (high 16 bits == 0xFFFF), false otherwise.
    /// </summary>
    public bool TryDecodeTileCoordinates(out int tileX, out int tileY)
    {
        tileX = tileY = 0;
        ushort high = (ushort)(TileCoordsRaw >> 16);
        ushort low = (ushort)(TileCoordsRaw & 0xFFFF);
        if (high != 0xFFFF) return false; // unknown schema

        // low word stored as YYXX (little-endian). Split bytes.
        byte yy = (byte)(low >> 8);
        byte xx = (byte)(low & 0xFF);
        tileX = xx;
        tileY = yy;
        return true;
    }

    /// <summary>
    /// Returns true if this entry has valid tile coordinates (raw data follows 0xFFFFYYXX pattern).
    /// </summary>
    public bool HasValidTileCoordinates => (TileCoordsRaw >> 16) == 0xFFFF;

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
        Load(br, (uint)inData.Length);
    }

    public void Load(BinaryReader br)
    {
        // This variant is for interface compliance. It's unsafe; prefer the size-aware version.
        Load(br, (uint)StructSize);
    }

    public void Load(BinaryReader br, uint chunkSize)
    {
        if (chunkSize < StructSize)
            throw new EndOfStreamException("MSLK entry truncated");

        Flags_0x00 = br.ReadByte();
        Type_0x01 = br.ReadByte();
        SortKey_0x02 = br.ReadUInt16();
        ParentId = br.ReadUInt32();
        MspiFirstIndex = ReadInt24(br);
        MspiIndexCount = br.ReadByte();
        TileCoordsRaw = br.ReadUInt32();
        SurfaceRefIndex = br.ReadUInt16();
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
        bw.Write(Flags_0x00);
        bw.Write(Type_0x01);
        bw.Write(SortKey_0x02);
        bw.Write(ParentId);
        WriteInt24(bw, MspiFirstIndex);
        bw.Write(MspiIndexCount);
        bw.Write(TileCoordsRaw);
        bw.Write(SurfaceRefIndex);
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
        Load(br, (uint)inData.Length);
    }

    public void Load(BinaryReader br)
    {
        // This variant is for interface compliance. It's unsafe; prefer the size-aware version.
        Load(br, (uint)(br.BaseStream.Length - br.BaseStream.Position));
    }

    public void Load(BinaryReader br, uint chunkSize)
    {
        int count = (int)(chunkSize / MslkEntry.StructSize);
        for (int i = 0; i < count; i++)
        {
            var e = new MslkEntry();
            e.Load(br, MslkEntry.StructSize);
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
