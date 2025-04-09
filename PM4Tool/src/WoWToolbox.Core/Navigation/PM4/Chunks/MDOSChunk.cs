using System;
using System.Collections.Generic;
using System.IO;
using Warcraft.NET.Files.Interfaces;

namespace WoWToolbox.Core.Navigation.PM4.Chunks
{
    /// <summary>
    /// Represents an entry in the MDOS chunk.
    /// Structure based on documentation at wowdev.wiki/PM4.md (MDOS section)
    /// </summary>
    public class MdosEntry
    {
        public uint Value_0x00 { get; set; }
        public uint Value_0x04 { get; set; }
        // public uint[] Values { get; private set; } = new uint[32]; // Reverted

        public const int Size = 8; // Bytes (uint32 + uint32) - Reverted to PM4 docs

        public void Load(BinaryReader br)
        {
            Value_0x00 = br.ReadUInt32();
            Value_0x04 = br.ReadUInt32();
            /* Reverted
            for (int j = 0; j < 32; j++)
            {
                Values[j] = br.ReadUInt32();
            }
            */
        }

        public void Write(BinaryWriter bw)
        {
            bw.Write(Value_0x00);
            bw.Write(Value_0x04);
            /* Reverted
            for (int j = 0; j < 32; j++)
            {
                bw.Write(Values[j]);
            }
            */
        }

        public override string ToString()
        {
            return $"MDOS Entry [Val0: 0x{Value_0x00:X8}, Val4: 0x{Value_0x04:X8}]"; // Reverted
        }
    }

    /// <summary>
    /// Represents the MDOS chunk.
    /// </summary>
    public class MDOSChunk : IIFFChunk, IBinarySerializable
    {
        public const string ExpectedSignature = "MDOS";
        public string GetSignature() => ExpectedSignature;

        public List<MdosEntry> Entries { get; private set; } = new List<MdosEntry>();

        /// <inheritdoc/>
        public uint GetSize()
        {
            return (uint)(Entries.Count * MdosEntry.Size);
        }

        /// <inheritdoc/>
        public void LoadBinaryData(byte[] chunkData)
        {
            if (chunkData == null) throw new ArgumentNullException(nameof(chunkData));
            // --- DEBUG: Log the size of the data received from the ChunkedFile loader ---
            Console.WriteLine($"DEBUG: MDOSChunk.LoadBinaryData received chunkData with Length = {chunkData.Length}");
            // --- END DEBUG ---

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
            if (size % MdosEntry.Size != 0)
            {
                // Handle potential padding or corruption? For now, log a warning.
                Console.WriteLine($"Warning: MDOS chunk size {size} is not a multiple of entry size {MdosEntry.Size}. Possible padding or corruption.");
                // Adjust size down to the nearest multiple if desired, or throw.
                size -= (size % MdosEntry.Size);
            }

            int entryCount = (int)(size / MdosEntry.Size);
            Entries = new List<MdosEntry>(entryCount);

            for (int i = 0; i < entryCount; i++)
            {
                var entry = new MdosEntry();
                entry.Load(br);
                Entries.Add(entry);
            }

            long bytesRead = br.BaseStream.Position - startPosition;
            if (bytesRead != size && size > 0) // Check if we read exactly the expected amount (ignoring padding scenario handled above)
            {
                 Console.WriteLine($"Warning: MDOS chunk read {bytesRead} bytes, expected multiple of {MdosEntry.Size} fitting within {size} bytes.");
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
            return $"MDOS Chunk [{Entries.Count} Entries]";
        }
    }
} 