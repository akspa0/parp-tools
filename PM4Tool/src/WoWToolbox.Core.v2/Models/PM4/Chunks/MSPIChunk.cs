using System.Collections.Generic;
using System.IO;
using Warcraft.NET.Files.Interfaces;

namespace WoWToolbox.Core.v2.Models.PM4.Chunks
{
    public class MSPIChunk : IIFFChunk, IBinarySerializable
    {
        public const string Signature = "MSPI";
        public List<uint> Indices { get; private set; } = new List<uint>();

        public string GetSignature() => Signature;

        public uint GetSize() => (uint)Indices.Count * 4;

        public void LoadBinaryData(byte[] inData)
        {
            using var ms = new MemoryStream(inData);
            using var br = new BinaryReader(ms);
            Read(br, (uint)inData.Length);
        }

        public void Read(BinaryReader reader, uint size)
        {
            if (size % 4 != 0)
                throw new InvalidDataException($"MSPI chunk size ({size}) must be a multiple of 4.");
            int indexCount = (int)(size / 4);
            Indices = new List<uint>(indexCount);
            for (int i = 0; i < indexCount; i++)
                Indices.Add(reader.ReadUInt32());
        }

        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms);
            Write(bw);
            return ms.ToArray();
        }

        /// <summary>
        /// Writes all indices to the provided BinaryWriter.
        /// </summary>
        public void Write(BinaryWriter writer)
        {
            foreach (var index in Indices)
                writer.Write(index);
        }

        /// <summary>
        /// Validates indices against the vertex count of the MSPV chunk.
        /// </summary>
        /// <param name="vertexCount">The number of vertices in the MSPV chunk.</param>
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
    }
} 