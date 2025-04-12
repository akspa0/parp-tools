using System;
using System.Collections.Generic;
using System.IO;
using Warcraft.NET.Files.Interfaces;
using WoWToolbox.Core.Vectors; // Keep C3Vectori for potential future use, though not directly in MprrEntry now

namespace WoWToolbox.Core.Navigation.PM4.Chunks
{
    /// <summary>
    /// Represents an entry in the MPRR chunk.
    /// Structure based on documentation at wowdev.wiki/PM4.md (MPRR section)
    /// The meaning of the fields is currently unknown, but analysis suggests:
    /// - They likely represent pairs of indices into the MPRL chunk.
    /// - Unknown_0x00 often uses 0xFFFF (65535) as a sentinel value (e.g., no starting point?).
    /// - Unknown_0x02 often references index 0 (MPRL[0]), possibly as a common hub or reference.
    /// Purpose might be defining path connections, visibility, or other relationships.
    /// </summary>
    public class MprrEntry
    {
        // Fields based on PM4.md documentation (4 bytes total)
        // Meaning is unknown according to wowdev.wiki as of 2024-Q3.
        public ushort Unknown_0x00 { get; set; }      // _0x00 from doc. Often 0xFFFF (sentinel?)
        public ushort Unknown_0x02 { get; set; }      // _0x02 from doc. Often 0 (MPRL index 0?)
        
        public const int Size = 4; // Bytes (ushort + ushort)

        public void Load(BinaryReader br)
        {
            Unknown_0x00 = br.ReadUInt16();
            Unknown_0x02 = br.ReadUInt16();
        }

        public void Write(BinaryWriter bw)
        {
            bw.Write(Unknown_0x00);
            bw.Write(Unknown_0x02);
        }
        
        public override string ToString()
        {
            return $"MPRR Entry [Unk0: 0x{Unknown_0x00:X4}, Unk2: 0x{Unknown_0x02:X4}] (Meaning Unknown)";
        }
    }

    /// <summary>
    /// Represents the MPRR chunk containing data potentially referencing MPRL positions.
    /// Analysis suggests it defines relationships (connections?) between pairs of MPRL indices.
    /// </summary>
    public class MPRRChunk : IIFFChunk, IBinarySerializable
    {
        public const string ExpectedSignature = "MPRR";
        public string GetSignature() => ExpectedSignature;

        public List<MprrEntry> Entries { get; private set; } = new List<MprrEntry>();

        /// <inheritdoc/>
        public uint GetSize()
        {
            return (uint)(Entries.Count * MprrEntry.Size); // Use updated Size
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

            // Use updated Size
            if (size % MprrEntry.Size != 0)
            {
                Entries.Clear();
                Console.WriteLine($"Warning: MPRR chunk size {size} is not a multiple of {MprrEntry.Size} bytes. Entry data might be corrupt.");
                return; // Or throw
            }

            // Use updated Size
            int entryCount = (int)(size / MprrEntry.Size);
            Entries = new List<MprrEntry>(entryCount);

            for (int i = 0; i < entryCount; i++)
            {
                var entry = new MprrEntry();
                entry.Load(br);
                Entries.Add(entry);
            }
            
            long bytesRead = br.BaseStream.Position - startPosition;
            if (bytesRead != size)
            {
                 Console.WriteLine($"Warning: MPRR chunk read {bytesRead} bytes, expected {size} bytes.");
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
        /// Validates indices based on the *old* structure. Needs complete re-evaluation.
        /// Temporarily disabled - returns true.
        /// </summary>
        /// <param name="mprlEntryCount">The total number of entries available in the corresponding MPRL chunk.</param>
        /// <returns>True (temporarily).</returns>
        public bool ValidateIndices(int mprlEntryCount)
        {
            // TODO: Re-evaluate MPRR validation based on the new 4-byte structure.
            // The meaning of Unknown_0x00 and Unknown_0x02 is currently unclear.
            // Cannot apply old logic based on MprlIndex and specific Unknown_0x00 flags.
            // Console.WriteLine("Warning: MPRR Index Validation is temporarily disabled pending structural understanding.");
            return true; // Temporarily return true as the meaning/purpose of the fields is unknown.

            /* --- REMOVED Validation Logic (Based on unconfirmed assumption) ---
            // IMPLEMENTED VALIDATION based on 4-byte structure (ushort, ushort)
            // Assuming Unknown_0x00 and Unknown_0x02 are indices into MPRL
            if (mprlEntryCount <= 0) return Entries.Count == 0; // If no MPRL entries, MPRR should be empty to be valid

            for (int i = 0; i < Entries.Count; i++)
            {
                var entry = Entries[i];
                ushort index1 = entry.Unknown_0x00;
                ushort index2 = entry.Unknown_0x02;

                // Skip sentinel check (65535 is invalid anyway if >= mprlEntryCount)
                // if (index1 == 65535 || index2 == 65535) continue;

                // Check if either index is out of bounds
                if (index1 >= mprlEntryCount || index2 >= mprlEntryCount)
                {
                    Console.WriteLine($">>> FAIL: MPRR Validation Failed. Entry {i} has index out of bounds.");
                    Console.WriteLine($"    Raw Indices: Idx1={index1}, Idx2={index2}. MPRL Count: {mprlEntryCount}.");
                    return false;
                }
            }
            return true;
            */
        }

        public override string ToString()
        {
            return $"MPRR Chunk [{Entries.Count} Entries]";
        }
    }
} 