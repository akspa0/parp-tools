using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Warcraft.NET.Files.Interfaces;
using WoWToolbox.Core.v2.Foundation.Helpers;

namespace WoWToolbox.Core.v2.Foundation.PM4.Chunks
{
    /// <summary>
    /// Represents an entry in the MDBH chunk, linking an index to a filename.
    /// Structure interpretation based on documentation at wowdev.wiki/PM4.md (MDBH section)
    /// </summary>
    public class MdbhEntry
    {
        public uint Index { get; set; }       // From MDBI sub-chunk concept
        public string Filename { get; set; } = string.Empty; // From MDBF sub-chunk concept
        public uint MdosIndex { get; set; }  // Placeholder mapping to MDOS

        // Size is variable due to string length

        public void Load(BinaryReader br)
        {
            Index = br.ReadUInt32();
            MdosIndex = Index;
            Filename = StringReadHelper.ReadNullTerminatedString(br, Encoding.UTF8); // Assuming UTF8
        }

        public void Write(BinaryWriter bw)
        {
            bw.Write(Index);
            StringWriteHelper.WriteNullTerminatedString(bw, Filename, Encoding.UTF8); // Assuming UTF8
        }

        public override string ToString()
        {
            return $"MDBH Entry [Index: {Index}, Filename: \"{Filename}\"]";
        }
    }

    /// <summary>
    /// Represents the MDBH chunk, containing destructible building header info.
    /// Note: Documentation mentions MDBF and MDBI as sub-chunks, but the structure description
    /// seems to imply a count followed by index/filename pairs directly within MDBH data.
    /// This implementation follows the latter interpretation.
    /// </summary>
    public class MDBHChunk : IIFFChunk, IBinarySerializable
    {
        public const string ExpectedSignature = "MDBH";
        public string GetSignature() => ExpectedSignature;

        public List<MdbhEntry> Entries { get; private set; } = new List<MdbhEntry>();
        public uint Count => (uint)Entries.Count; // Reflects the leading count field

        /// <inheritdoc/>
        public uint GetSize()
        {
            uint totalSize = 4; // For the count field
            foreach (var entry in Entries)
            {
                totalSize += 4; // For the index field
                totalSize += (uint)Encoding.UTF8.GetByteCount(entry.Filename) + 1; // For filename + null terminator
            }
            return totalSize;
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
            long streamLength = br.BaseStream.Length;

            if (streamLength - startPosition < 4)
                throw new InvalidDataException("MDBH chunk too small to contain count field.");

            uint entryCount = br.ReadUInt32();
            Entries = new List<MdbhEntry>((int)entryCount);

            if (entryCount > 0 && streamLength - br.BaseStream.Position < 4)
            {
                 Console.WriteLine($"Warning: MDBH chunk has count {entryCount} but not enough data for the first index. Stopping MDBH entry load.");
                 return;
            }

            for (int i = 0; i < entryCount; i++)
            {
                 if (br.BaseStream.Position + 4 > streamLength)
                 {
                     Console.WriteLine($"Warning: MDBH chunk unexpected end of stream while reading index for entry {i}. Processed {Entries.Count} entries.");
                     break;
                 }

                var entry = new MdbhEntry();
                try
                {
                    entry.Load(br);
                }
                catch (EndOfStreamException ex)
                {
                    throw new InvalidDataException($"MDBH chunk unexpected end of stream while reading filename for entry {i}.", ex);
                }
                catch (Exception ex)
                {
                    // Catch potential issues during string reading
                    throw new InvalidDataException($"Error reading MDBH entry {i}: {ex.Message}", ex);
                }
                Entries.Add(entry);
            }

            // Optional: Check if bytes read matches expected, though variable length makes it complex.
            // long bytesRead = br.BaseStream.Position - startPosition;
        }

        /// <inheritdoc/>
        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream((int)GetSize());
            using var bw = new BinaryWriter(ms);

            bw.Write(Count); // Write the count first
            foreach (var entry in Entries)
            {
                entry.Write(bw);
            }

            return ms.ToArray();
        }

        public override string ToString()
        {
            return $"MDBH Chunk [Count: {Count}]";
        }
    }
}
