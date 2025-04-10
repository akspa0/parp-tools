using System;
using System.Collections.Generic;
using System.IO;
using Warcraft.NET.Files.Interfaces;
using System.Numerics; // For Vector3 if used in entry helpers

namespace WoWToolbox.Core.Navigation.PM4.Chunks
{
    /// <summary>
    /// Represents a single entry in the MDSF chunk.
    /// Contains structure-related data with mixed types.
    /// </summary>
    public class MDSFEntry
    {
        public uint Value_0x00 { get; set; } // uint32_t
        public uint Value_0x04 { get; set; } // uint32_t
        public uint Value_0x08 { get; set; } // uint32_t
        public uint Value_0x0C { get; set; } // uint32_t
        public uint Value_0x10 { get; set; } // uint32_t
        public uint Value_0x14 { get; set; } // uint32_t
        public float Float_0x18 { get; set; } // float
        public float Float_0x1C { get; set; } // float
        public float Float_0x20 { get; set; } // float
        public float Float_0x24 { get; set; } // float
        public float Float_0x28 { get; set; } // float
        public float Float_0x2C { get; set; } // float
        public float Float_0x30 { get; set; } // float
        public float Float_0x34 { get; set; } // float (Duplicate field name in spec? Assuming this is 0x34)
        public uint Value_0x38 { get; set; } // uint32_t
        public uint Value_0x3C { get; set; } // uint32_t
        public uint Value_0x40 { get; set; } // uint32_t

        public const int EntrySize = 68; // 6*4 + 8*4 + 3*4 = 24 + 32 + 12 = 68

        public void Load(BinaryReader br)
        {
            Value_0x00 = br.ReadUInt32();
            Value_0x04 = br.ReadUInt32();
            Value_0x08 = br.ReadUInt32();
            Value_0x0C = br.ReadUInt32();
            Value_0x10 = br.ReadUInt32();
            Value_0x14 = br.ReadUInt32();
            Float_0x18 = br.ReadSingle();
            Float_0x1C = br.ReadSingle();
            Float_0x20 = br.ReadSingle();
            Float_0x24 = br.ReadSingle();
            Float_0x28 = br.ReadSingle();
            Float_0x2C = br.ReadSingle();
            Float_0x30 = br.ReadSingle();
            Float_0x34 = br.ReadSingle();
            Value_0x38 = br.ReadUInt32();
            Value_0x3C = br.ReadUInt32();
            Value_0x40 = br.ReadUInt32();
        }

        public void Write(BinaryWriter bw)
        {
            bw.Write(Value_0x00);
            bw.Write(Value_0x04);
            bw.Write(Value_0x08);
            bw.Write(Value_0x0C);
            bw.Write(Value_0x10);
            bw.Write(Value_0x14);
            bw.Write(Float_0x18);
            bw.Write(Float_0x1C);
            bw.Write(Float_0x20);
            bw.Write(Float_0x24);
            bw.Write(Float_0x28);
            bw.Write(Float_0x2C);
            bw.Write(Float_0x30);
            bw.Write(Float_0x34);
            bw.Write(Value_0x38);
            bw.Write(Value_0x3C);
            bw.Write(Value_0x40);
        }

        // Optional helper methods from spec example
        // public Vector3 GetPotentialPosition() => new Vector3(Float_0x18, Float_0x1C, Float_0x20);
        // public Vector3 GetPotentialRotation() => new Vector3(Float_0x24, Float_0x28, Float_0x2C);
        // public float GetPotentialScale() => Float_0x30; // Spec example seems off, maybe 0x30 or 0x34?

        public override string ToString()
        {
            // Provide a concise representation
            return $"MDSF Entry [0x00={Value_0x00:X8}, ..., Pos?=({Float_0x18}, {Float_0x1C}, {Float_0x20}), ..., 0x40={Value_0x40:X8}]";
        }
    }

    /// <summary>
    /// Represents the MDSF chunk containing structure data.
    /// Based on documentation at chunkvault/chunks/PM4/M014_MDSF.md
    /// </summary>
    public class MDSFChunk : IIFFChunk, IBinarySerializable
    {
        public const string ExpectedSignature = "MDSF";
        public string GetSignature() => ExpectedSignature;

        public List<MDSFEntry> Entries { get; private set; } = new List<MDSFEntry>();

        /// <inheritdoc/>
        public uint GetSize()
        {
            return (uint)Entries.Count * MDSFEntry.EntrySize;
        }

        /// <inheritdoc/>
        public void LoadBinaryData(byte[] chunkData)
        {
            if (chunkData == null) throw new ArgumentNullException(nameof(chunkData));

            using var ms = new MemoryStream(chunkData);
            using var br = new BinaryReader(ms);
            Load(br);
        }

        /// <inheritdoc/>
        public void Load(BinaryReader br)
        {
            long startPosition = br.BaseStream.Position;
            long endPosition = br.BaseStream.Length;
            long size = endPosition - startPosition;

             if (size < 0) throw new InvalidOperationException("Stream size is negative.");

            if (size % MDSFEntry.EntrySize != 0)
            {
                Entries.Clear();
                Console.WriteLine($"Warning: MDSF chunk size {size} is not a multiple of {MDSFEntry.EntrySize} bytes. Entry data might be corrupt.");
                size -= (size % MDSFEntry.EntrySize); // Process only complete entries
            }

            int entryCount = (int)(size / MDSFEntry.EntrySize);
            Entries = new List<MDSFEntry>(entryCount);

            for (int i = 0; i < entryCount; i++)
            {
                 if (br.BaseStream.Position + MDSFEntry.EntrySize > br.BaseStream.Length)
                {
                     Console.WriteLine($"Warning: MDSF chunk unexpected end of stream at entry {i}. Read {Entries.Count} entries out of expected {entryCount}.");
                     break;
                }
                var entry = new MDSFEntry();
                entry.Load(br);
                Entries.Add(entry);
            }
            
             long bytesRead = br.BaseStream.Position - startPosition;
            if (bytesRead != size + (size % MDSFEntry.EntrySize))
            {
                 Console.WriteLine($"Warning: MDSF chunk read {bytesRead} bytes, expected to process based on size {size}. Original size reported by header might have padding or corruption.");
            }
        }

        /// <inheritdoc/>
        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream((int)GetSize());
            using var bw = new BinaryWriter(ms);

            foreach (var entry in Entries)
            {
                entry.Write(bw);
            }

            return ms.ToArray();
        }

        public override string ToString()
        {
            return $"MDSF Chunk [{Entries.Count} Entries]";
        }
    }
} 