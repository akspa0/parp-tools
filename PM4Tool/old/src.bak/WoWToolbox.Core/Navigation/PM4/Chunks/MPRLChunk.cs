using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Warcraft.NET.Files.Interfaces;
using WoWToolbox.Core.Vectors; // Use C3Vector instead of C3Vectori
using Warcraft.NET.Files.Structures; // <-- Add this for C3Vector
// Assuming C3Vector is accessible from here, as it's defined with MSPVChunk

namespace WoWToolbox.Core.Navigation.PM4.Chunks
{
    /// <summary>
    /// Represents an entry in the MPRL chunk.
    /// Based on documentation at wowdev.wiki/PM4.md
    /// </summary>
    public class MprlEntry
    {
        public ushort Unknown_0x00 { get; set; }          // _0x00 in doc
        public short Unknown_0x02 { get; set; }           // _0x02 in doc
        public ushort Unknown_0x04 { get; set; }          // _0x04 in doc
        public ushort Unknown_0x06 { get; set; }          // _0x06 in doc
        public C3Vector Position { get; set; }           // Offset 0x08 (12 bytes) - Changed to C3Vector (float)
        public short Unknown_0x14 { get; set; }           // _0x14 in doc
        public ushort Unknown_0x16 { get; set; }          // _0x16 in doc
        // Removed old UnknownFloat fields

        public const int Size = 24; // Bytes (2+2+2+2 + C3Vector=12 + 2+2)

        public void Load(BinaryReader br)
        {
            Unknown_0x00 = br.ReadUInt16();
            Unknown_0x02 = br.ReadInt16();
            Unknown_0x04 = br.ReadUInt16();
            Unknown_0x06 = br.ReadUInt16();
            Position = new C3Vector // Read as floats
            {
                X = br.ReadSingle(),
                Y = br.ReadSingle(),
                Z = br.ReadSingle()
            };
            Unknown_0x14 = br.ReadInt16();
            Unknown_0x16 = br.ReadUInt16();

            // Remove debug log
            // Console.WriteLine($"DEBUG MPRL Entry: Pos=({Position.X}, {Position.Y}, {Position.Z})"); 
        }

        public void Write(BinaryWriter bw)
        {
            bw.Write(Unknown_0x00);
            bw.Write(Unknown_0x02);
            bw.Write(Unknown_0x04);
            bw.Write(Unknown_0x06);
            bw.Write(Position.X); // Write floats
            bw.Write(Position.Y);
            bw.Write(Position.Z);
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