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
public sealed class MsurChunk : IIffChunk, IBinarySerializable
{
    public const string Signature = "MSUR";

    public sealed class Entry
    {
        /// <summary>
        /// Primary grouping field that appears to define object type/category.
        /// Values in range 0-32 typically represent building exteriors,
        /// values 33-64 are often building interiors, and values 65-128 appear to be terrain elements.
        /// This is the most reliable field for coherent object grouping.
        /// </summary>
        public byte FlagsOrUnknown_0x00;   // GroupKey / flags (offset 0x00)
        
        /// <summary>
        /// Number of indices for this surface in the MSVI chunk.
        /// </summary>
        public byte IndexCount;            // Triangle index count (offset 0x01)
        
        /// <summary>
        /// Secondary grouping field that further subdivides objects. Within each FlagsOrUnknown_0x00 group,
        /// this field typically increases sequentially. Jumps in values often indicate new floors
        /// or major architectural changes. Values of 0 often indicate base elements.
        /// Bit 0x80 appears to indicate possible liquid surfaces.
        /// </summary>
        public byte Unknown_0x02;          // Attribute mask (offset 0x02)
        
        /// <summary>
        /// Padding byte, always 0 in version 48.
        /// </summary>
        public byte Padding_0x03;          // Padding (offset 0x03)
        
        /// <summary>Surface normal X component</summary>
        public float Nx;                   // Normal X (offset 0x04)
        
        /// <summary>Surface normal Y component</summary>
        public float Ny;                   // Normal Y (offset 0x08)
        
        /// <summary>Surface normal Z component</summary>
        public float Nz;                   // Normal Z (offset 0x0C)
        
        /// <summary>Plane D value or surface height</summary>
        public float Height;               // Surface height (offset 0x10)
        
        /// <summary>First index in the MSVI chunk for this surface</summary>
        public uint MsviFirstIndex;        // MSVI first index (offset 0x14)
        
        /// <summary>MDOS chunk reference index</summary>
        public uint MdosIndex;             // MDOS reference (offset 0x18)
        
        /// <summary>
        /// 32-bit composite key that may contain encoded surface parameters.
        /// Often split into high/low 16-bit portions for more granular grouping.
        /// </summary>
        public uint PackedParams;          // 32-bit composite key (offset 0x1C)

        // Convenience accessors expected by adapters/exporters
        
        /// <summary>
        /// Alias for FlagsOrUnknown_0x00, representing the primary grouping key.
        /// This is the most reliable field for coherent object grouping in PM4 files.
        /// </summary>
        /// <remarks>
        /// Values typically indicate object types:
        /// - 0-32: Building exteriors
        /// - 33-64: Building interiors
        /// - 65-128: Terrain elements
        /// - 128+: Special objects or world elements
        /// </remarks>
        public byte SurfaceGroupKey => FlagsOrUnknown_0x00;
        
        /// <summary>
        /// Indicates whether this surface is part of an M2 model bucket.
        /// M2 buckets typically have SurfaceGroupKey = 0x00.
        /// </summary>
        public bool IsM2Bucket => SurfaceGroupKey == 0x00;
        
        /// <summary>
        /// Alias for Unknown_0x02, representing surface attributes and subdivision within object types.
        /// </summary>
        public byte SurfaceAttributeMask => Unknown_0x02;
        
        /// <summary>
        /// Indicates whether this surface might represent a liquid surface based on bit 0x80.
        /// </summary>
        public bool IsLiquidCandidate => (Unknown_0x02 & 0x80) != 0;
        
        /// <summary>
        /// The complete 32-bit composite surface key used for legacy grouping.
        /// This is less effective than using FlagsOrUnknown_0x00 and Unknown_0x02 directly.
        /// </summary>
        public uint SurfaceKey => PackedParams;

        /// <summary>
        /// High 16 bits of the PackedParams field, useful for coarse grouping.
        /// </summary>
        public ushort SurfaceKeyHigh16 => (ushort)(PackedParams >> 16);
        
        /// <summary>
        /// Low 16 bits of the PackedParams field, useful for fine-grained grouping.
        /// </summary>
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
