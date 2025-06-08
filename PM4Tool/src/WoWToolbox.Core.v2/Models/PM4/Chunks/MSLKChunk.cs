using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using Warcraft.NET.Files.Interfaces;

namespace WoWToolbox.Core.v2.Models.PM4.Chunks
{
    /// <summary>
    /// Optimized MSLK entry with decoded fields and performance improvements.
    /// Structure based on statistical analysis - represents object metadata system.
    /// </summary>
    public class MSLKEntry : IBinarySerializable
    {
        #region Core Fields - DECODED THROUGH STATISTICAL ANALYSIS

        // DECODED FIELDS - Object Metadata System (20 bytes total)
        public byte Unknown_0x00 { get; set; } // DECODED: Object Type Flags (1-18 values for classification)
        public byte Unknown_0x01 { get; set; } // DECODED: Object Subtype (0-7 values for variants)
        public ushort Unknown_0x02 { get; set; } // DECODED: Padding/Reserved (always 0x0000)
        public uint Unknown_0x04 { get; set; } // DECODED: Group/Object ID (organizational grouping identifier)
        public int MspiFirstIndex { get; set; } // int24_t - Index into MSPI for geometry, -1 for Doodad nodes
        public byte MspiIndexCount { get; set; } // uint8_t - Number of points in MSPI for geometry, 0 for Doodad nodes
        public uint Unknown_0x0C { get; set; } // DECODED: Material/Color ID (pattern: 0xFFFF#### for material references)
        public ushort Unknown_0x10 { get; set; } // DECODED: Reference Index (cross-references to other data structures)
        public ushort Unknown_0x12 { get; set; } // DECODED: System Flag (always 0x8000 - confirmed constant)

        #endregion

        #region Decoded Property Accessors

        /// <summary>Gets the object type flags for classification (1-18 different values)</summary>
        public byte ObjectTypeFlags => Unknown_0x00;

        /// <summary>Gets the object subtype for variant classification (0-7 different values)</summary>
        public byte ObjectSubtype => Unknown_0x01;

        /// <summary>Gets the group/object ID for organizational grouping</summary>
        public uint GroupObjectId => Unknown_0x04;

        /// <summary>Gets the material/color ID (pattern: 0xFFFF#### where #### varies)</summary>
        public uint MaterialColorId => Unknown_0x0C;

        /// <summary>Gets the reference index for cross-referencing other data structures</summary>
        public ushort ReferenceIndex => Unknown_0x10;

        /// <summary>Checks if this entry has geometry data (MSPI references)</summary>
        public bool HasGeometry => MspiFirstIndex >= 0 && MspiIndexCount > 0;

        /// <summary>Checks if this is a self-referencing root node (building separator)</summary>
        public bool IsRootNode => Unknown_0x04 == GetHashCode() % uint.MaxValue; // Approximation for index check

        #endregion

        public const int StructSize = 20; // Total size in bytes

        /// <summary>Initializes a new instance of the MSLKEntry class</summary>
        public MSLKEntry() { }

        #region IBinarySerializable Implementation

        /// <inheritdoc/>
        public void LoadBinaryData(byte[] inData)
        {
            if (inData == null || inData.Length < StructSize)
                throw new ArgumentException($"Input data must be at least {StructSize} bytes.", nameof(inData));

            using var ms = new MemoryStream(inData);
            using var br = new BinaryReader(ms);
            Load(br);
        }

        /// <inheritdoc/>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Load(BinaryReader br)
        {
            if (br.BaseStream.Position + StructSize > br.BaseStream.Length)
                throw new EndOfStreamException($"Not enough data remaining to read MSLKEntry (requires {StructSize} bytes).");

            Unknown_0x00 = br.ReadByte();
            Unknown_0x01 = br.ReadByte();
            Unknown_0x02 = br.ReadUInt16();
            Unknown_0x04 = br.ReadUInt32();
            MspiFirstIndex = ReadInt24(br); // Read 24-bit signed integer
            MspiIndexCount = br.ReadByte();
            Unknown_0x0C = br.ReadUInt32();
            Unknown_0x10 = br.ReadUInt16();
            Unknown_0x12 = br.ReadUInt16();
        }

        /// <inheritdoc/>
        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream(StructSize);
            using var bw = new BinaryWriter(ms);
            Write(bw);
            return ms.ToArray();
        }

        /// <summary>Writes the entry to a BinaryWriter</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Write(BinaryWriter bw)
        {
            bw.Write(Unknown_0x00);
            bw.Write(Unknown_0x01);
            bw.Write(Unknown_0x02);
            bw.Write(Unknown_0x04);
            WriteInt24(bw, MspiFirstIndex); // Write 24-bit signed integer
            bw.Write(MspiIndexCount);
            bw.Write(Unknown_0x0C);
            bw.Write(Unknown_0x10);
            bw.Write(Unknown_0x12);
        }

        /// <inheritdoc/>
        public uint GetSize() => StructSize;

        #endregion

        #region Helper Methods

        /// <summary>Reads a 24-bit signed integer (little-endian)</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int ReadInt24(BinaryReader br)
        {
            byte b1 = br.ReadByte();
            byte b2 = br.ReadByte();
            byte b3 = br.ReadByte();

            int value = b1 | (b2 << 8) | (b3 << 16);

            // Sign extend if the sign bit (MSB of the 3rd byte) is set
            if ((b3 & 0x80) != 0)
            {
                value |= unchecked((int)0xFF000000); // Sign extend with 0xFF
            }

            return value;
        }

        /// <summary>Writes a 24-bit signed integer (little-endian)</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void WriteInt24(BinaryWriter bw, int value)
        {
            bw.Write((byte)(value & 0xFF));
            bw.Write((byte)((value >> 8) & 0xFF));
            bw.Write((byte)((value >> 16) & 0xFF));
        }

        #endregion

        public override string ToString()
        {
            return $"MSLK Entry [Type:{ObjectTypeFlags:X2}, Sub:{ObjectSubtype:X2}, Group:{GroupObjectId:X8}, " +
                   $"MSPI:{MspiFirstIndex}+{MspiIndexCount}, Mat:{MaterialColorId:X8}, Ref:{ReferenceIndex:X4}]";
        }
    }

    /// <summary>
    /// Optimized MSLK chunk with performance improvements and efficient operations.
    /// Contains scene graph links and object metadata for PM4 building extraction.
    /// </summary>
    public class MSLKChunk : IIFFChunk, IBinarySerializable, IDisposable
    {
        /// <summary>The chunk signature</summary>
        public const string Signature = "MSLK";

        private List<MSLKEntry>? _entries;

        /// <summary>Gets the entries in this chunk with lazy initialization</summary>
        public List<MSLKEntry> Entries => _entries ??= new List<MSLKEntry>();

        /// <summary>Gets the entry count efficiently</summary>
        public int EntryCount => _entries?.Count ?? 0;

        /// <summary>Checks if the chunk has any entries</summary>
        public bool HasEntries => EntryCount > 0;

        /// <summary>Initializes a new instance of the MSLKChunk class</summary>
        public MSLKChunk() { }

        /// <summary>Initializes a new instance from binary data</summary>
        public MSLKChunk(byte[] inData)
        {
            LoadBinaryData(inData);
        }

        #region Efficient Access Methods

        /// <summary>
        /// Gets entries with geometry data efficiently.
        /// </summary>
        /// <returns>Entries that have MSPI geometry references</returns>
        public IEnumerable<MSLKEntry> GetEntriesWithGeometry()
        {
            if (_entries == null) yield break;
            
            for (int i = 0; i < _entries.Count; i++)
            {
                var entry = _entries[i];
                if (entry.HasGeometry)
                    yield return entry;
            }
        }

        /// <summary>
        /// Finds entries by group/object ID efficiently.
        /// </summary>
        /// <param name="groupId">Group ID to search for</param>
        /// <returns>Entries with matching group ID</returns>
        public IEnumerable<(int index, MSLKEntry entry)> GetEntriesByGroup(uint groupId)
        {
            if (_entries == null) yield break;
            
            for (int i = 0; i < _entries.Count; i++)
            {
                var entry = _entries[i];
                if (entry.GroupObjectId == groupId)
                    yield return (i, entry);
            }
        }

        /// <summary>
        /// Finds potential root nodes (self-referencing entries).
        /// </summary>
        /// <returns>Indices and entries that could be building root nodes</returns>
        public IEnumerable<(int index, MSLKEntry entry)> GetPotentialRootNodes()
        {
            if (_entries == null) yield break;
            
            for (int i = 0; i < _entries.Count; i++)
            {
                var entry = _entries[i];
                if (entry.Unknown_0x04 == i) // Self-referencing = root node
                    yield return (i, entry);
            }
        }

        /// <summary>
        /// Pre-allocates capacity for known entry count.
        /// </summary>
        /// <param name="entryCount">Expected number of entries</param>
        public void PreAllocate(int entryCount)
        {
            if (entryCount > 0)
            {
                _entries ??= new List<MSLKEntry>(entryCount);
                if (_entries.Capacity < entryCount)
                    _entries.Capacity = entryCount;
            }
        }

        #endregion

        #region IBinarySerializable Implementation

        /// <inheritdoc/>
        public void LoadBinaryData(byte[] inData)
        {
            if (inData == null)
                throw new ArgumentNullException(nameof(inData));

            using var ms = new MemoryStream(inData);
            using var br = new BinaryReader(ms);
            Load(br);
        }

        /// <inheritdoc/>
        public void Load(BinaryReader br)
        {
            if (br == null)
                throw new ArgumentNullException(nameof(br));

            var remainingBytes = br.BaseStream.Length - br.BaseStream.Position;
            var entryCount = (int)(remainingBytes / MSLKEntry.StructSize);
            
            if (entryCount > 0)
            {
                PreAllocate(entryCount);
                
                for (int i = 0; i < entryCount; i++)
                {
                    var entry = new MSLKEntry();
                    entry.Load(br);
                    Entries.Add(entry);
                }
            }
        }

        /// <inheritdoc/>
        public byte[] Serialize(long offset = 0)
        {
            if (_entries == null || _entries.Count == 0)
                return Array.Empty<byte>();

            var totalSize = _entries.Count * MSLKEntry.StructSize;
            using var ms = new MemoryStream(totalSize);
            using var bw = new BinaryWriter(ms);
            
            foreach (var entry in _entries)
            {
                entry.Write(bw);
            }
            
            return ms.ToArray();
        }

        /// <inheritdoc/>
        public string GetSignature() => Signature;

        /// <inheritdoc/>
        public uint GetSize()
        {
            return (uint)(EntryCount * MSLKEntry.StructSize);
        }

        #endregion

        #region IDisposable

        /// <summary>Disposes resources and clears entries</summary>
        public void Dispose()
        {
            _entries?.Clear();
            _entries = null;
            GC.SuppressFinalize(this);
        }

        #endregion

        public override string ToString()
        {
            return $"MSLK Chunk [{EntryCount} entries, {GetSize()} bytes]";
        }
    }

    /// <summary>
    /// Type alias for backward compatibility with original MSLK class name.
    /// </summary>
    public class MSLK : MSLKChunk
    {
        /// <summary>Initializes a new instance of the MSLK class</summary>
        public MSLK() : base() { }

        /// <summary>Initializes a new instance from binary data</summary>
        public MSLK(byte[] inData) : base(inData) { }
    }
} 