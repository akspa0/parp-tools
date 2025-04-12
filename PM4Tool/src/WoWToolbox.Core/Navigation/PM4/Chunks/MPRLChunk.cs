using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Warcraft.NET.Files.Interfaces;
// using WoWToolbox.Core.Vectors; // Use C3Vectori from Warcraft.NET instead
// using Warcraft.NET.Files.Structures; // Using C3Vectori from WoWToolbox.Core.Vectors
using WoWToolbox.Core.Vectors; // Keep this for C3Vectori definition
// Assuming C3Vector is accessible from here, as it's defined with MSPVChunk
using Warcraft.NET.Files.Structures; // Use C3Vector from Warcraft.NET

namespace WoWToolbox.Core.Navigation.PM4.Chunks
{
    /// <summary>
    /// Represents an entry in the MPRL chunk.
    /// Based on documentation at wowdev.wiki/PM4.md
    /// Reverting Position to C3Vector (float) based on visual results.
    /// </summary>
    public class MprlEntry
    {
        public ushort Unknown_0x00 { get; set; }          // _0x00 in doc - Meaning TBD (Wowdev says "Always 0"?)
        public short Unknown_0x02 { get; set; }           // _0x02 in doc - Meaning TBD (Wowdev says "Always -1"?)
        public ushort Unknown_0x04 { get; set; }          // _0x04 in doc - Meaning TBD.
        public ushort Unknown_0x06 { get; set; }          // _0x06 in doc - Meaning TBD.
        public C3Vector Position { get; set; }           // Offset 0x08 (12 bytes) - Vertex position (float).
        public short Unknown_0x14 { get; set; }           // _0x14 in doc - Meaning TBD.
        public ushort Unknown_0x16 { get; set; }          // _0x16 in doc - Meaning TBD.
        // Removed old UnknownFloat fields

        public const int Size = 24; // Bytes (2+2+2+2 + C3Vector(12) + 2+2)

        public void Load(BinaryReader br)
        {
            // Read fields before position
            Unknown_0x00 = br.ReadUInt16();
            Unknown_0x02 = br.ReadInt16();
            Unknown_0x04 = br.ReadUInt16();
            Unknown_0x06 = br.ReadUInt16();
            
            // Read Position as C3Vector (float, assuming X, Y, Z order in file for simplicity now)
            float fileX = br.ReadSingle();
            float fileY = br.ReadSingle(); 
            float fileZ = br.ReadSingle(); 
            
            // Assign to struct properties
            Position = new C3Vector { X = fileX, Y = fileY, Z = fileZ };

            // Read fields after position
            Unknown_0x14 = br.ReadInt16();
            Unknown_0x16 = br.ReadUInt16();
        }

        public void Write(BinaryWriter bw)
        {
            // Write fields before position
            bw.Write(Unknown_0x00);
            bw.Write(Unknown_0x02);
            bw.Write(Unknown_0x04);
            bw.Write(Unknown_0x06);

            // Write Position as C3Vector (X, Y, Z)
            bw.Write(Position.X);
            bw.Write(Position.Y); 
            bw.Write(Position.Z);

            // Write fields after position
            bw.Write(Unknown_0x14);
            bw.Write(Unknown_0x16);
        }

        public override string ToString()
        {
            // Updated ToString to reflect new fields (Position type change doesn't affect output format)
            return $"MPRL Entry [Unk00:{Unknown_0x00}, Unk02:{Unknown_0x02}, Unk04:{Unknown_0x04}, Unk06:{Unknown_0x06}, Pos:{Position}, Unk14:{Unknown_0x14}, Unk16:{Unknown_0x16}]";
        }
    }

    /// <summary>
    /// Represents the MPRL chunk containing position data referenced by MPRR.
    /// </summary>
    public class MPRLChunk : IIFFChunk, IBinarySerializable
    {
        public const string ExpectedSignature = "MPRL";
        public string GetSignature() => ExpectedSignature;

        public List<MprlEntry> Entries { get; private set; } = new List<MprlEntry>();

        /// <inheritdoc/>
        public uint GetSize()
        {
            return (uint)(Entries.Count * MprlEntry.Size);
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

            if (size % MprlEntry.Size != 0)
            {
                Entries.Clear();
                Console.WriteLine($"Warning: MPRL chunk size {size} is not a multiple of {MprlEntry.Size} bytes. Entry data might be corrupt.");
                return; // Or throw
            }

            int entryCount = (int)(size / MprlEntry.Size);
            Entries = new List<MprlEntry>(entryCount);

            for (int i = 0; i < entryCount; i++)
            {
                var entry = new MprlEntry();
                entry.Load(br);
                // Log the position components immediately after reading
                // Console.WriteLine($"DEBUG MPRL Entry {i}: Pos=({entry.Position.X}, {entry.Position.Y}, {entry.Position.Z})"); // REMOVED
                Entries.Add(entry);
            }
            
            long bytesRead = br.BaseStream.Position - startPosition;
            if (bytesRead != size)
            {
                 Console.WriteLine($"Warning: MPRL chunk read {bytesRead} bytes, expected {size} bytes.");
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
            return $"MPRL Chunk [{Entries.Count} Entries]";
        }
    }
} 