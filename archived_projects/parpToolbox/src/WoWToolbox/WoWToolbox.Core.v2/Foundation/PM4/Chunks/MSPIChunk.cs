using System.Collections.Generic;
using System.IO;
using Warcraft.NET.Files.Interfaces;

namespace WoWToolbox.Core.v2.Foundation.PM4.Chunks
{
    /// <summary>
    /// Represents the MSPI chunk containing indices into the MSPV chunk.
    /// </summary>
    public class MSPIChunk : IIFFChunk, IBinarySerializable
    {
        public const string Signature = "MSPI";

        public List<uint> Indices { get; private set; }

        public MSPIChunk()
        {
            Indices = new List<uint>();
        }

        public string GetSignature() => Signature;

        public void LoadBinaryData(byte[] inData)
        {
            using var ms = new MemoryStream(inData);
            using var br = new BinaryReader(ms);
            Read(br, (uint)inData.Length);
        }

        public void Read(BinaryReader reader, uint size)
        {
            if (size % 4 != 0)
            {
                throw new InvalidDataException($"MSPI chunk size ({size}) must be a multiple of 4.");
            }

            int indexCount = (int)(size / 4);
            Indices = new List<uint>(indexCount);

            for (int i = 0; i < indexCount; i++)
            {
                Indices.Add(reader.ReadUInt32());
            }
        }

        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms);
            Write(bw);
            return ms.ToArray();
        }

        public void Write(BinaryWriter writer)
        {
            foreach (uint index in Indices)
            {
                writer.Write(index);
            }
        }

        public uint GetSize()
        {
            return (uint)Indices.Count * 4;
        }

        /// <summary>
        /// Validates indices against the vertex count of the MSPV chunk.
        /// </summary>
        /// <param name="vertexCount">The number of vertices in the MSPV chunk.</param>
        /// <returns>True if all indices are valid, false otherwise.</returns>
        public bool ValidateIndices(int vertexCount)
        {
            foreach (uint index in Indices)
            {
                if (index >= vertexCount)
                {
                    return false;
                }
            }
            return true;
        }
    }
}
