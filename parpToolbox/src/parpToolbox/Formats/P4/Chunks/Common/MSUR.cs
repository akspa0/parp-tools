namespace ParpToolbox.Formats.P4.Chunks.Common;

using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;

/// <summary>
/// MSUR – Surface metadata chunk. Each entry is 32 bytes (confirmed July-2025 spec). The earlier 64-byte
/// assumption was incorrect and caused misalignment. This implementation matches the authoritative
/// `WoWToolbox.Core.v2` definition so downstream grouping logic receives valid indices.
/// Unknown / padding bytes are kept for diagnostic purposes.
/// </summary>
public sealed class MsurChunk : IIffChunk, IBinarySerializable
{
    public const string Signature = "MSUR";

    public sealed class Entry
    {
        // Raw fields (disk layout)
        public byte FlagsOrUnknown_0x00;   // GroupKey / flags
        public byte IndexCount;            // _0x01 – triangle index count
        public byte Unknown_0x02;          // _0x02 – attribute mask
        public byte Padding_0x03;          // _0x03 – padding
        public float Nx;                   // _0x04
        public float Ny;                   // _0x08
        public float Nz;                   // _0x0C
        public float Height;               // _0x10 – plane D or surface height
        public uint MsviFirstIndex;        // _0x14
        public uint MdosIndex;             // _0x18
        public uint PackedParams;          // _0x1C

        // Convenience accessors expected by adapters/exporters
        public byte SurfaceGroupKey => FlagsOrUnknown_0x00;
        public bool IsM2Bucket => SurfaceGroupKey == 0x00;
        public byte SurfaceAttributeMask => Unknown_0x02;
        public bool IsLiquidCandidate => (Unknown_0x02 & 0x80) != 0;
        public uint SurfaceKey => PackedParams; // 32-bit composite key

        // High/low 16-bit portions (preferred for grouping)
        public ushort SurfaceKeyHigh16 => (ushort)(PackedParams >> 16);
        public ushort SurfaceKeyLow16  => (ushort)(PackedParams & 0xFFFF);
    }

    private readonly List<Entry> _entries = new();
    public IReadOnlyList<Entry> Entries => _entries;

    public string GetSignature() => Signature;
    public uint GetSize() => (uint)(_entries.Count * 32);

    public void LoadBinaryData(byte[] inData)
    {
        using var ms = new MemoryStream(inData ?? throw new ArgumentNullException(nameof(inData)));
        using var br = new BinaryReader(ms);
        Load(br);
    }

    public void Load(BinaryReader br)
    {
        while (br.BaseStream.Position + 32 <= br.BaseStream.Length)
        {
            var e = new Entry
            {
                FlagsOrUnknown_0x00 = br.ReadByte(),
                IndexCount = br.ReadByte(),
                Unknown_0x02 = br.ReadByte(),
                Padding_0x03 = br.ReadByte(),
                Nx = br.ReadSingle(),
                Ny = br.ReadSingle(),
                Nz = br.ReadSingle(),
                Height = br.ReadSingle(),
                MsviFirstIndex = br.ReadUInt32(),
                MdosIndex = br.ReadUInt32(),
                PackedParams = br.ReadUInt32()
            };
            _entries.Add(e);
        }
    }

    public byte[] Serialize(long offset = 0) => throw new NotSupportedException();
}
