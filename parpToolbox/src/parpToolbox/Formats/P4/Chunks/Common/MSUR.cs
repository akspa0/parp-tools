namespace ParpToolbox.Formats.P4.Chunks.Common;

using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;

/// <summary>
/// MSUR – Surface metadata chunk. Each entry is 32 bytes (confirmed July-2025 spec). The earlier 64-byte
/// assumption was incorrect and caused misalignment. This implementation matches the authoritative
/// `WoWToolbox.Core.v2` definition so downstream grouping logic receives valid indices.
/// </summary>
/// <remarks>
/// <para>
/// MSUR fields provide the most semantically meaningful grouping of PM4 geometry. Through extensive
/// testing and comparison of multiple grouping strategies, we've discovered that MSUR raw fields
/// (particularly FlagsOrUnknown_0x00 and Unknown_0x02) define coherent object boundaries.
/// </para>
/// <para>
/// Key insights about MSUR fields:
/// - FlagsOrUnknown_0x00: Defines object categories/types (values 0-32: building exteriors, 33-64: interiors)
/// - Unknown_0x02: Further subdivides objects, often by floors or architectural sections
/// - MSUR fields define horizontal slices of objects corresponding to building construction
/// - Superior to alternative grouping methods (SurfaceGroupKey, ParentIndex, MPRR sentinels)
/// </para>
/// </remarks>
public sealed partial class MsurChunk : IIffChunk, IBinarySerializable
{
    public const string Signature = "MSUR";

    public sealed partial class Entry
    {
        /// <summary>
        /// Primary grouping field that defines object type/category (e.g., building exterior, interior, terrain).
        /// This is the most reliable field for coherent object grouping.
        /// </summary>
        public byte GroupKey;              // Formerly FlagsOrUnknown_0x00

        /// <summary>Number of indices for this surface in the MSVI chunk.</summary>
        public int IndexCount;             // Triangle index count (little-endian 16-bit at offsets 0x01-0x02)

        /// <summary>Raw byte at offset 0x03 (semantics unknown).</summary>
        public byte Unknown03;             // Raw field @0x03, semantics TBD

        /// <summary>Surface normal X component.</summary>
        public float Nx;                   // Normal X (offset 0x04)

        /// <summary>Surface normal Y component.</summary>
        public float Ny;                   // Normal Y (offset 0x08)

        /// <summary>Surface normal Z component.</summary>
        public float Nz;                   // Normal Z (offset 0x0C)

        /// <summary>Raw float at offset 0x10 (semantics unknown).</summary>
        public float Float10;              // Raw field @0x10, semantics TBD

        /// <summary>First index in the MSVI chunk for this surface.</summary>
        public uint MsviFirstIndex;        // MSVI first index (offset 0x14)

        /// <summary>MDOS chunk reference index.</summary>
        public uint MdosIndex;             // MDOS reference (offset 0x18)

        /// <summary>32-bit composite key that may contain encoded surface parameters.</summary>
        public uint CompositeKey;          // Formerly PackedParams (offset 0x1C)

        // Minimal helpers without semantic claims
        public uint SurfaceKey => CompositeKey;
        public ushort SurfaceKeyHigh16 => (ushort)(CompositeKey >> 16);
        public ushort SurfaceKeyLow16  => (ushort)(CompositeKey & 0xFFFF);
    }

    private readonly List<Entry> _entries = new();
    public IReadOnlyList<Entry> Entries => _entries;

    public string GetSignature() => Signature;
    public uint GetSize() => (uint)(_entries.Count * 32);

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
        long endOffset = br.BaseStream.Position + chunkSize;
        while (br.BaseStream.Position + 32 <= endOffset)
        {
            var e = new Entry
            {
                GroupKey = br.ReadByte(),
                // IndexCount is 16-bit little-endian across the next two bytes
                IndexCount = br.ReadByte() | (br.ReadByte() << 8),
                Unknown03 = br.ReadByte(),
                Nx = br.ReadSingle(),
                Ny = br.ReadSingle(),
                Nz = br.ReadSingle(),
                Float10 = br.ReadSingle(),
                MsviFirstIndex = br.ReadUInt32(),
                MdosIndex = br.ReadUInt32(),
                CompositeKey = br.ReadUInt32()
            };
            _entries.Add(e);
        }
    }

    public byte[] Serialize(long offset = 0) => throw new NotSupportedException();
}
