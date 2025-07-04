using System;
using System.Collections.Generic;
using System.IO;
using Warcraft.NET.Files.Interfaces;
using Warcraft.NET.Extensions;

namespace WoWToolbox.Core.v2.Foundation.PM4.Chunks
{
    /// <summary>
    /// Represents an entry in the MSLK chunk.
    /// Structure based on documentation at wowdev.wiki/PD4.md (MSLK section)
    /// Moved into MSLK.cs to resolve build issues.
    /// </summary>
    public class MSLKEntry : IBinarySerializable
    {
        // Fields based on PD4.md documentation (24 bytes total)
        // DECODED THROUGH STATISTICAL ANALYSIS - OBJECT METADATA SYSTEM
        public byte Unknown_0x00 { get; set; } // DECODED: Object Type Flags (1-18 values for classification)
        public byte Unknown_0x01 { get; set; } // DECODED: Object Subtype (0-7 values for variants)
        public ushort Unknown_0x02 { get; set; } // DECODED: Padding/Reserved (always 0x0000)
        public uint Unknown_0x04 { get; set; } // DECODED: Group/Object ID (organizational grouping identifier)
        public int MspiFirstIndex { get; set; } // int24_t - Index into MSPI for geometry, -1 for Doodad nodes.
        public byte MspiIndexCount { get; set; } // uint8_t - Number of points in MSPI for geometry, 0 for Doodad nodes.
        public uint Unknown_0x0C { get; set; } // DECODED: Material/Color ID (pattern: 0xFFFF#### for material references)
        public ushort Unknown_0x10 { get; set; } // DECODED: Reference Index (cross-references to other data structures)
        public ushort Unknown_0x12 { get; set; } // DECODED: System Flag (always 0x8000 - confirmed constant)

        // Decoded metadata accessors
        /// <summary>
        /// Gets the object type flags for classification (1-18 different values).
        /// </summary>
        public byte ObjectTypeFlags => Unknown_0x00;

        /// <summary>
        /// Gets the object subtype for variant classification (0-7 different values).
        /// </summary>
        public byte ObjectSubtype => Unknown_0x01;

        /// <summary>
        /// Gets the group/object ID for organizational grouping.
        /// </summary>
        public uint GroupObjectId => Unknown_0x04;

        /// <summary>
        /// Gets the material/color ID (pattern: 0xFFFF#### where #### varies).
        /// </summary>
        public uint MaterialColorId => Unknown_0x0C;

        /// <summary>
        /// Gets the reference index for cross-referencing other data structures.
        /// </summary>
        public ushort ReferenceIndex => Unknown_0x10;

                // Convenience helpers expected by higher-level services
        public bool IsGeometryNode => MspiIndexCount > 0;
        public byte ObjectType => Unknown_0x00;
        public ushort Unk10 => Unknown_0x10;
        public ushort Unk12 => Unknown_0x12;

        public const int StructSize = 20; // Total size in bytes (1+1+2+4+3+1+4+2+2 = 20)

        // --- Speculative Properties (Decoding Logic Unknown) ---
        
        /// <summary>
        /// SPECULATIVE: Potential scale factor. Actual decoding logic is unknown.
        /// This is a placeholder based on the simplest possible interpretation.
        /// </summary>
        public float? SpeculativeScale => null; // Placeholder - No clear candidate field identified yet.

        /// <summary>
        /// SPECULATIVE: Represents potential packed quaternion orientation data.
        /// Actual decoding logic is unknown. Likely involves Unk00, Unk01, Unk12.
        /// </summary>
        public object SpeculativeOrientationData => new { U00 = Unknown_0x00, U01 = Unknown_0x01, U12 = Unknown_0x12 }; // Placeholder

        // --- End Speculative Properties ---

        /// <summary>
        /// Initializes a new instance of the <see cref="MSLKEntry"/> class
        /// </summary>
        public MSLKEntry() { }

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

        // Added public Write method for use by MSLK chunk serializer
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
        public uint GetSize()
        {
            return StructSize;
        }

        // Helper method to read a 24-bit signed integer (assuming little-endian)
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

        // Helper method to write a 24-bit signed integer (assuming little-endian)
        private static void WriteInt24(BinaryWriter bw, int value)
        {
            bw.Write((byte)(value & 0xFF));
            bw.Write((byte)((value >> 8) & 0xFF));
            bw.Write((byte)((value >> 16) & 0xFF));
        }

        public override string ToString()
        {
            // Update ToString to reflect new fields
            return $"MSLK Entry [Unk00:0x{Unknown_0x00:X2}, Unk01:0x{Unknown_0x01:X2}, Unk02:0x{Unknown_0x02:X4}, Unk04:0x{Unknown_0x04:X8}, " +
                   $"MSPIFirst:{MspiFirstIndex}, MSPICount:{MspiIndexCount}, " +
                   $"Unk0C:0x{Unknown_0x0C:X8}, Unk10:0x{Unknown_0x10:X4}, Unk12:0x{Unknown_0x12:X4}]";
        }
    }

    /// <summary>
    /// Represents an MSLK chunk in a PM4 file, which contains links between vertices
    /// </summary>
    public class MSLK : IIFFChunk, IBinarySerializable
    {
        /// <summary>
        /// The chunk signature
        /// </summary>
        public const string Signature = "MSLK";

        /// <summary>
        /// Gets or sets the entries in this chunk
        /// </summary>
        public List<MSLKEntry> Entries { get; set; } = new List<MSLKEntry>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MSLK"/> class
        /// </summary>
        public MSLK() { }

        /// <summary>
        /// Initializes a new instance of the <see cref="MSLK"/> class
        /// </summary>
        /// <param name="inData">The binary data</param>
        public MSLK(byte[] inData)
        {
            LoadBinaryData(inData);
        }

        /// <inheritdoc/>
        public void LoadBinaryData(byte[] inData)
        {
            using (var ms = new MemoryStream(inData))
            using (var br = new BinaryReader(ms))
            {
                Load(br);
            }
        }

        /// <inheritdoc/>
        public void Load(BinaryReader br)
        {
            if (MSLKEntry.StructSize <= 0) throw new InvalidOperationException("MSLKEntry StructSize must be positive.");
            if (br.BaseStream.Length % MSLKEntry.StructSize != 0) {
                 // Optional: Log a warning here about unexpected data length
                 // Console.WriteLine($"Warning: MSLK data length {br.BaseStream.Length} is not a multiple of entry size {MSLKEntry.StructSize}");
            }

            var entryCount = br.BaseStream.Length / MSLKEntry.StructSize; // Use StructSize constant
            Entries.Clear(); // Ensure list is empty before loading
            Entries.Capacity = (int)entryCount; // Pre-allocate list capacity
            
            for (var i = 0; i < entryCount; i++)
            {
                var entry = new MSLKEntry();
                entry.Load(br); // MSLKEntry.Load reads StructSize bytes
                Entries.Add(entry);
            }
        }

        /// <inheritdoc/>
        public byte[] Serialize(long offset = 0)
        {
            using (var ms = new MemoryStream(Entries.Count * MSLKEntry.StructSize)) // Use StructSize constant
            using (var bw = new BinaryWriter(ms))
            {
                foreach (var entry in Entries)
                {
                    entry.Write(bw); // Use the public Write method from MSLKEntry
                }
                return ms.ToArray();
            }
        }

        /// <inheritdoc/>
        public string GetSignature()
        {
            return Signature;
        }

        /// <inheritdoc/>
        public uint GetSize()
        {
            return (uint)(Entries.Count * MSLKEntry.StructSize); // Use StructSize constant
        }

        public override string ToString()
        {
            return $"MSLK Chunk [{Entries.Count} entries]";
        }
    }
}
