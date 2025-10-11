using System.Collections.Generic;
using System.IO;
using Warcraft.NET.Files.Interfaces;

namespace WoWToolbox.Core.v2.Models.PM4.Chunks
{
    public class MSVIChunk : IIFFChunk, IBinarySerializable
    {
        public const string ExpectedSignature = "MSVI";
        public string GetSignature() => ExpectedSignature;
        public List<uint> Indices { get; private set; } = new List<uint>();

        public uint GetSize() => (uint)Indices.Count * sizeof(uint);

        public void LoadBinaryData(byte[] chunkData)
        {
            using var ms = new MemoryStream(chunkData);
            using var br = new BinaryReader(ms);
            Load(br);
        }

        public void Load(BinaryReader br)
        {
            long startPosition = br.BaseStream.Position;
            long endPosition = br.BaseStream.Length;
            long size = endPosition - startPosition;
            if (size < 0) throw new InvalidDataException("Stream size is negative.");
            if (size % sizeof(uint) != 0)
            {
                size -= (size % sizeof(uint));
            }
            int count = (int)(size / sizeof(uint));
            Indices = new List<uint>(count);
            for (int i = 0; i < count; i++)
                Indices.Add(br.ReadUInt32());
        }

        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms);
            foreach (var index in Indices)
                bw.Write(index);
            return ms.ToArray();
        }

        /// <summary>
        /// Validates if all indices are within the bounds of a given vertex count.
        /// </summary>
        /// <param name="vertexCount">The total number of vertices (e.g., from MSVT chunk).</param>
        /// <returns>True if all indices are valid, false otherwise.</returns>
        public bool ValidateIndices(int vertexCount)
        {
            foreach (var index in Indices)
            {
                if (index >= vertexCount)
                    return false;
            }
            return true;
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
                return new List<uint>();
            uint actualCount = System.Math.Min(count, (uint)(Indices.Count - firstIndex));
            return Indices.GetRange((int)firstIndex, (int)actualCount);
        }

        public override string ToString()
        {
            return $"MSVI Chunk [{Indices.Count} Indices]";
        }
    }
} 