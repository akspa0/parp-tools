using System;
using System.Collections.Generic;
using System.IO;
using Warcraft.NET.Files.Interfaces;
using System.Numerics; // For Vector3 if used in entry helpers

namespace WoWToolbox.Core.Navigation.PM4.Chunks
{
    /// <summary>
    /// Represents an entry in the MDSF chunk.
    /// Structure based *strictly* on documentation at wowdev.wiki/PM4.md (MDSF section)
    /// </summary>
    public class MdsfEntry
    {
        // Reverting to 8-byte structure based on PM4 documentation
        public uint Value_0x00 { get; set; }
        public uint Value_0x04 { get; set; }

        public const int Size = 8; // Bytes (uint32 + uint32)

        public void Load(BinaryReader br)
        {
            Value_0x00 = br.ReadUInt32();
            Value_0x04 = br.ReadUInt32();
        }

        public void Write(BinaryWriter bw)
        {
            bw.Write(Value_0x00);
            bw.Write(Value_0x04);
        }

        public override string ToString()
        {
            return $"MDSF Entry [Val0: 0x{Value_0x00:X8}, Val4: 0x{Value_0x04:X8}]";
        }
    }

    /// <summary>
    /// Represents the MDSF chunk containing surface data.
    /// </summary>
    public class MDSFChunk : IIFFChunk, IBinarySerializable
    {
        public const string ExpectedSignature = "MDSF";
        public string GetSignature() => ExpectedSignature;

        public List<MdsfEntry> Entries { get; private set; } = new List<MdsfEntry>();

        /// <inheritdoc/>
        public uint GetSize()
        {
            return (uint)(Entries.Count * MdsfEntry.Size);
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
            long endPosition = br.BaseStream.Length; // Assuming the reader is positioned at the start of the chunk data
            long size = endPosition - startPosition;

            if (size < 0) throw new InvalidDataException("Stream size is negative.");

            // Use the corrected Entry Size from the documentation
            if (size % MdsfEntry.Size != 0)
            {
                // Log a warning, but continue processing based on the number of full entries
                Console.WriteLine($"Warning: MDSF chunk size {size} is not a multiple of the documented entry size {MdsfEntry.Size}. Possible padding or corruption.");
                size -= (size % MdsfEntry.Size); // Process only complete entries
            }

            int entryCount = (int)(size / MdsfEntry.Size);
            Entries = new List<MdsfEntry>(entryCount);

            for (int i = 0; i < entryCount; i++)
            {
                if (br.BaseStream.Position + MdsfEntry.Size > br.BaseStream.Length)
                {
                    Console.WriteLine($"Warning: MDSF chunk unexpected end of stream at entry {i}. Read {Entries.Count} entries out of expected {entryCount}.");
                    break;
                }
                var entry = new MdsfEntry();
                entry.Load(br);
                Entries.Add(entry);
            }

            long bytesRead = br.BaseStream.Position - startPosition;
            // Check if we read exactly the multiple of entry size we decided to process
            if (bytesRead != size && size > 0)
            {
                 Console.WriteLine($"Warning: MDSF chunk read {bytesRead} bytes, but expected to process {size} bytes based on multiples of {MdsfEntry.Size}.");
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