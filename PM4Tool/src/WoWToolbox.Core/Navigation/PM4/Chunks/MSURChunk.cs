using System;
using System.Collections.Generic;
using System.IO;
using Warcraft.NET.Files.Interfaces;

namespace WoWToolbox.Core.Navigation.PM4.Chunks
{
    /// <summary>
    /// Represents an entry in the MSUR chunk, defining a surface.
    /// Based on documentation at chunkvault/chunks/PM4/M007_MSUR.md
    /// </summary>
    public class MsurEntry
    {
        public byte FlagsOrUnknown_0x00 { get; set; }          // _0x00 - Meaning TBD. Might be flags.
        public byte IndexCount { get; set; }          // _0x01 - Number of indices in MSVI used by this surface.
        public byte Unknown_0x02 { get; set; }            // _0x02 - Meaning TBD.
        public byte Padding_0x03 { get; set; }             // _0x03 - Likely padding.
        public float UnknownFloat_0x04 { get; set; }             // _0x04 - Meaning TBD.
        public float UnknownFloat_0x08 { get; set; }             // _0x08 - Meaning TBD.
        public float UnknownFloat_0x0C { get; set; }             // _0x0C - Meaning TBD.
        public float UnknownFloat_0x10 { get; set; }             // _0x10 - Meaning TBD.
        public uint MsviFirstIndex { get; set; }      // _0x14 - Starting index in MSVI for this surface.
        public uint MdosIndex { get; set; }            // _0x18 - Index into MDOS (Destructible Object States). Note: This field is NOT directly present in PM4 docs but is implied by MDSF link.
        public uint Unknown_0x1C { get; set; }            // _0x1C - Meaning TBD.

        public const int Size = 32; // Bytes

        public void Load(BinaryReader br)
        {
            FlagsOrUnknown_0x00 = br.ReadByte();
            IndexCount = br.ReadByte();
            Unknown_0x02 = br.ReadByte();
            Padding_0x03 = br.ReadByte();
            UnknownFloat_0x04 = br.ReadSingle();
            UnknownFloat_0x08 = br.ReadSingle();
            UnknownFloat_0x0C = br.ReadSingle();
            UnknownFloat_0x10 = br.ReadSingle();
            MsviFirstIndex = br.ReadUInt32();
            MdosIndex = br.ReadUInt32(); // Read _0x18 as MdosIndex
            Unknown_0x1C = br.ReadUInt32();
        }

        public void Write(BinaryWriter bw)
        {
            bw.Write(FlagsOrUnknown_0x00);
            bw.Write(IndexCount);
            bw.Write(Unknown_0x02);
            bw.Write(Padding_0x03);
            bw.Write(UnknownFloat_0x04);
            bw.Write(UnknownFloat_0x08);
            bw.Write(UnknownFloat_0x0C);
            bw.Write(UnknownFloat_0x10);
            bw.Write(MsviFirstIndex);
            bw.Write(MdosIndex); // Write MdosIndex at _0x18
            bw.Write(Unknown_0x1C);
        }

        public override string ToString()
        {
            return $"MSUR Entry [Index: {MsviFirstIndex}, Count: {IndexCount}, Flags: {FlagsOrUnknown_0x00:X2}, MdosIndex: {MdosIndex}]"; // Updated ToString
        }
    }

    /// <summary>
    /// Represents the MSUR chunk containing surface definitions.
    /// </summary>
    public class MSURChunk : IIFFChunk, IBinarySerializable
    {
        public const string ExpectedSignature = "MSUR";
        public string GetSignature() => ExpectedSignature;

        public List<MsurEntry> Entries { get; private set; } = new List<MsurEntry>();

        /// <inheritdoc/>
        public uint GetSize()
        {
            return (uint)(Entries.Count * MsurEntry.Size);
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

            if (size % MsurEntry.Size != 0)
            {
                Entries.Clear();
                Console.WriteLine($"Warning: MSUR chunk size {size} is not a multiple of {MsurEntry.Size} bytes. Entry data might be corrupt.");
                return; // Or throw
            }

            int entryCount = (int)(size / MsurEntry.Size);
            Entries = new List<MsurEntry>(entryCount);

            for (int i = 0; i < entryCount; i++)
            {
                var entry = new MsurEntry();
                entry.Load(br);
                Entries.Add(entry);
            }
            
            long bytesRead = br.BaseStream.Position - startPosition;
            if (bytesRead != size)
            {
                 Console.WriteLine($"Warning: MSUR chunk read {bytesRead} bytes, expected {size} bytes.");
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

        /// <summary>
        /// Validates that all index ranges defined by the entries are within the bounds 
        /// of the provided MSVI index count.
        /// </summary>
        /// <param name="msviIndexCount">The total number of indices available in the corresponding MSVI chunk.</param>
        /// <returns>True if all ranges are valid, false otherwise.</returns>
        public bool ValidateIndices(int msviIndexCount)
        {
             if (msviIndexCount <= 0) return Entries.Count == 0; // No indices means no valid ranges unless empty

            for(int i = 0; i < Entries.Count; i++)
            {
                var entry = Entries[i];
                // Check if the starting index is valid
                if (entry.MsviFirstIndex >= msviIndexCount)
                {
                    Console.WriteLine($"Validation Error: MSUR entry {i} MsviFirstIndex ({entry.MsviFirstIndex}) is out of bounds for MSVI index count {msviIndexCount}.");
                    return false;
                }
                // Check if the range (start + count) exceeds the bounds
                // Need to cast MsviFirstIndex to long before adding to avoid potential uint overflow if IndexCount is large, though unlikely with byte.
                if ((long)entry.MsviFirstIndex + entry.IndexCount > msviIndexCount)
                {
                    Console.WriteLine($"Validation Error: MSUR entry {i} range ({entry.MsviFirstIndex} + {entry.IndexCount}) exceeds MSVI index count {msviIndexCount}.");
                    return false;
                }
            }
            return true;
        }

        public override string ToString()
        {
            return $"MSUR Chunk [{Entries.Count} Entries]";
        }
    }
} 