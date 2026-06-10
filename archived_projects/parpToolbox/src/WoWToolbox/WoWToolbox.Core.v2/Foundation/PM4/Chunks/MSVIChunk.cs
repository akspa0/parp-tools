using System;
using System.Collections.Generic;
using System.IO;
using Warcraft.NET.Files.Interfaces;

namespace WoWToolbox.Core.v2.Foundation.PM4.Chunks
{
    /// <summary>
    /// Represents the MSVI chunk containing vertex indices referencing the MSVT chunk.
    /// Based on documentation at chunkvault/chunks/PM4/M006_MSVI.md
    /// </summary>
    public class MSVIChunk : IIFFChunk, IBinarySerializable
    {
        public const string ExpectedSignature = "MSVI";
        public string GetSignature() => ExpectedSignature;

        public List<uint> Indices { get; private set; } = new List<uint>();

        /// <inheritdoc/>
        public uint GetSize()
        {
            return (uint)Indices.Count * sizeof(uint);
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
            long endPosition = br.BaseStream.Length; // Use length of stream passed to Load, not the underlying file
            long size = endPosition - startPosition;

            if (size < 0) throw new InvalidOperationException("Stream size is negative.");
            if (size % sizeof(uint) != 0)
            {
                Indices.Clear();
                Console.WriteLine($"Warning: MSVI chunk size {size} is not a multiple of {sizeof(uint)} bytes. Index data might be corrupt.");
                // Depending on desired strictness, could throw an exception here.
                // Read as many full indices as possible.
                size -= (size % sizeof(uint));
            }

            int indexCount = (int)(size / sizeof(uint));
            Indices = new List<uint>(indexCount);

            for (int i = 0; i < indexCount; i++)
            {
                if (br.BaseStream.Position + sizeof(uint) > br.BaseStream.Length)
                {
                     Console.WriteLine($"Warning: MSVI chunk unexpected end of stream at index {i}. Read {Indices.Count} indices out of expected {indexCount}.");
                     break; // Stop reading if not enough data
                }
                Indices.Add(br.ReadUInt32());
            }
            
            long bytesRead = br.BaseStream.Position - startPosition;
            if (bytesRead != size + (size % sizeof(uint))) // Check against original expected size if padded
            {
                 Console.WriteLine($"Warning: MSVI chunk read {bytesRead} bytes, expected to process based on size {size}. Original size reported by header might have padding or corruption.");
            }
        }

        /// <inheritdoc/>
        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream((int)GetSize());
            using var bw = new BinaryWriter(ms);

            foreach (var index in Indices)
            {
                bw.Write(index);
            }

            return ms.ToArray();
        }

        /// <summary>
        /// Validates if all indices are within the bounds of a given vertex count.
        /// </summary>
        /// <param name="vertexCount">The total number of vertices (e.g., from MSVT chunk).</param>
        /// <returns>True if all indices are valid, false otherwise.</returns>
        public bool ValidateIndices(int vertexCount)
        {
            Console.WriteLine($"--- MSVIChunk.ValidateIndices START (Vertex Count: {vertexCount}) ---");
            bool isValid = true;
            foreach (uint index in Indices)
            {
                // Console.WriteLine($"  MSVI Validating Index: {index}"); // Re-commented: Very verbose
                if (index >= vertexCount)
                {
                    Console.WriteLine($"Error: MSVIChunk.ValidateIndices - Index {index} is OUT OF BOUNDS (>= Vertex Count: {vertexCount}).");
                    isValid = false; // Keep original logic: flag invalidity but finish loop (though unlikely to hit now)
                }
            }
            Console.WriteLine($"--- MSVIChunk.ValidateIndices END (Result: {isValid}) ---");
            return isValid;
        }

        /// <summary>
        /// Retrieves a range of indices, typically corresponding to a surface definition from MSUR.
        /// </summary>
        /// <param name="firstIndex">The starting index (zero-based) within this chunk's index list.</param>
        /// <param name="count">The number of indices to retrieve.</param>
        /// <returns>A list of indices for the specified range, or an empty list if the range is invalid.</returns>
        public List<uint> GetIndicesForSurface(uint firstIndex, uint count)
        {
            if (firstIndex >= Indices.Count)
            {
                 Console.WriteLine($"Warning: MSVI GetIndicesForSurface - firstIndex ({firstIndex}) is out of bounds ({Indices.Count}).");
                 return new List<uint>();
            }

            // Ensure we don't read beyond the array bounds
            uint actualCount = Math.Min(count, (uint)(Indices.Count - firstIndex));
            if (actualCount != count)
            {
                 Console.WriteLine($"Warning: MSVI GetIndicesForSurface - requested count ({count}) exceeds available indices from firstIndex ({firstIndex}). Returning {actualCount} indices.");
            }

            return Indices.GetRange((int)firstIndex, (int)actualCount);
        }


        public override string ToString()
        {
            return $"MSVI Chunk [{Indices.Count} Indices]";
        }
    }
}
