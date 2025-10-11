using System;
using System.Collections.Generic;
using System.IO;
using System.Linq; // Added for Sum()
using Warcraft.NET.Files.Interfaces;
// Removed unused WoWToolbox.Core.Vectors using

namespace WoWToolbox.Core.Navigation.PM4.Chunks
{
    // Removed MprrEntry class entirely as the fixed-pair structure was incorrect.

    /// <summary>
    /// Represents the MPRR chunk containing sequences of unsigned short values.
    /// Structure analysis (2024-Q3) indicates it's composed of variable-length sequences,
    /// each terminated by the value 0xFFFF (65535).
    /// The meaning of the values within the sequences (especially those before the terminator)
    /// and the target they index into (if any) is currently unknown. It's likely *not* MPRL indices.
    /// </summary>
    public class MPRRChunk : IIFFChunk, IBinarySerializable
    {
        public const string ExpectedSignature = "MPRR";
        public string GetSignature() => ExpectedSignature;

        /// <summary>
        /// List of sequences found in the chunk. Each inner list is a sequence of ushorts,
        /// including the terminating 0xFFFF.
        /// </summary>
        public List<List<ushort>> Sequences { get; private set; } = new List<List<ushort>>();

        /// <inheritdoc/>
        public uint GetSize()
        {
            // Calculate total size based on the number of ushorts in all sequences.
            return (uint)(Sequences.Sum(seq => seq.Count) * sizeof(ushort));
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
            Sequences.Clear();
            long startPosition = br.BaseStream.Position;
            long endPosition = br.BaseStream.Length; // Assuming Load is called with a stream containing ONLY this chunk's data.

            while (br.BaseStream.Position < endPosition)
            {
                var currentSequence = new List<ushort>();
                try
                {
                    while (true) // Loop until terminator or end of stream
                    {
                        if (br.BaseStream.Position >= endPosition)
                        {
                             // This is normal - we've reached the end of the chunk data
                             if (currentSequence.Count > 0)
                             {
                                 // Reached end of chunk while reading sequence - this is normal behavior
                             }
                             goto EndLoad; // Exit outer loop
                        }

                        ushort value = br.ReadUInt16();
                        currentSequence.Add(value);

                        if (value == 0xFFFF)
                        {
                            break; // End of this sequence
                        }
                    }
                    Sequences.Add(currentSequence);
                }
                catch (EndOfStreamException)
                {
                    // Reached end of stream - this is normal for the last sequence in a chunk
                    // Optionally add the incomplete sequence if needed: if (currentSequence.Count > 0) Sequences.Add(currentSequence);
                    break; // Exit outer loop
                }
            }

        EndLoad:
            long bytesRead = br.BaseStream.Position - startPosition;
            long expectedSize = endPosition - startPosition;
            if (bytesRead != expectedSize)
            {
                 Console.WriteLine($"Warning: MPRR chunk read {bytesRead} bytes, but expected chunk size was {expectedSize}. Data might be incomplete or reader positioning issue.");
            }
        }

        /// <inheritdoc/>
        public byte[] Serialize(long offset = 0)
        {
            // Use GetSize() which now calculates based on sequence content.
            using var ms = new MemoryStream((int)GetSize());
            using var bw = new BinaryWriter(ms);

            foreach (var sequence in Sequences)
            {
                foreach (var value in sequence)
                {
                    bw.Write(value);
                }
            }

            return ms.ToArray();
        }

        /// <summary>
        /// Validates indices based on the *old* structure assumption (pairs indexing MPRL).
        /// This logic is no longer valid due to the variable-length sequence structure
        /// and the unknown target of the indices.
        /// Method is disabled pending further understanding of MPRR data.
        /// </summary>
        /// <param name="mprlEntryCount">The total number of entries previously assumed to be available in MPRL.</param>
        /// <returns>Always true (validation disabled).</returns>
        public bool ValidateIndices(int mprlEntryCount)
        {
            /* --- VALIDATION DISABLED ---
               Reason: The previous implementation assumed MPRR contained pairs of ushorts
               indexing into the MPRL chunk. Analysis revealed MPRR consists of variable-length
               sequences terminated by 0xFFFF, and the indices likely do *not* target MPRL.
               Therefore, this validation logic is incorrect and cannot be applied.
               Re-evaluation is needed once the true meaning and target of the MPRR sequence
               values are understood.
            */
            // Console.WriteLine("Warning: MPRR Index Validation is disabled due to structural changes and unknown index targets.");
            return true;
        }

        public override string ToString()
        {
            // Updated to reflect the new sequence structure.
            return $"MPRR Chunk [{Sequences.Count} Sequences]";
        }
    }
} 