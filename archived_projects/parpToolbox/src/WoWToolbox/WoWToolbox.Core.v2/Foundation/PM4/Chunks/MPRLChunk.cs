using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Warcraft.NET.Files.Interfaces;
using Warcraft.NET.Files.Structures;

namespace WoWToolbox.Core.v2.Foundation.PM4.Chunks
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
        public ushort Unknown_0x06 { get; set; }          // _0x06 legacy raw

        // July-2025: analysis shows this value groups portals into navigation zones.
        // Keep old name for binary load; use ZoneGroupId when writing new code.
        public ushort ZoneGroupId => Unknown_0x06;
        public C3Vector Position { get; set; }           // Offset 0x08 (12 bytes) - Vertex position (float).
        public short Unknown_0x14 { get; set; }           // _0x14 legacy raw

        // July-2025: statistical analysis suggests this signed short encodes portal floor
        // elevation relative to the map tile origin, in game units.
        public short FloorOffset => Unknown_0x14;
        public ushort Unknown_0x16 { get; set; }          // _0x16 legacy raw

        // July-2025: hypothesised attributes bit-field (see AnalyzeLinksTool for mask usage).
        public ushort AttributeFlags => Unknown_0x16;
        // Removed old UnknownFloat fields

        // -----------------------------
        // Clarified field aliases (July 2025 spec update)
        // -----------------------------

        /// <summary>
        /// When <see cref="SecondaryLinkSentinel"/> equals -1 the
        /// <see cref="SecondaryReference"/> field points to an alternate position that can
        /// satisfy missing <c>MSLK.Reference</c> values.
        /// </summary>
        public short SecondaryLinkSentinel => Unknown_0x02;

        /// <summary>
        /// Potential reference used to patch missing MSLK links when
        /// <see cref="SecondaryLinkSentinel"/> == -1.
        /// </summary>
        public ushort SecondaryReference => Unknown_0x04;

        /// <summary>Raw vertex position helper for external tools.</summary>
        public C3Vector VertexPosition => Position;

        [Obsolete("Use SecondaryLinkSentinel instead – will be removed after August 2025.")]
        public short Unk02Alias => Unknown_0x02;

        [Obsolete("Use SecondaryReference instead – will be removed after August 2025.")]
        public ushort Unk04Alias => Unknown_0x04;

        /// <summary>Fixed-size of the binary structure (in bytes).</summary>
        public const int Size = 24; // (2+2+2+2) + 12 + (2+2) = 24



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
            return $"MPRL Entry [Unk00:{Unknown_0x00}, Unk02:{Unknown_0x02}, Unk04:{Unknown_0x04}, Unk06:{Unknown_0x06}, Pos:{Position}, FloorOfs:{FloorOffset}, Attr:0x{AttributeFlags:X4}]";
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
