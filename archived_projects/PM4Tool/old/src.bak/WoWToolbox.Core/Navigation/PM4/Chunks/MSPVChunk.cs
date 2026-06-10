using System.Collections.Generic;
using System.IO;
using Warcraft.NET.Files.Interfaces;
using WoWToolbox.Core.Vectors;

namespace WoWToolbox.Core.Navigation.PM4.Chunks
{
    /// <summary>
    /// Represents the MSPV chunk containing MSP vertices.
    /// Uses WoWToolbox.Core.Vectors.C3Vectori based on documentation.
    /// </summary>
    public class MSPVChunk : IIFFChunk, IBinarySerializable
    {
        public const string Signature = "MSPV";

        public List<C3Vectori> Vertices { get; private set; }

        public MSPVChunk()
        {
            Vertices = new List<C3Vectori>();
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
            const int vertexSize = 12; // 3 * int32, Size of WoWToolbox.Core.Vectors.C3Vectori
            if (size % vertexSize != 0)
            {
                throw new InvalidDataException($"MSPV chunk size ({size}) must be a multiple of {vertexSize}.");
            }

            int vertexCount = (int)(size / vertexSize);
            Vertices = new List<C3Vectori>(vertexCount);

            for (int i = 0; i < vertexCount; i++)
            {
                C3Vectori vertex = new C3Vectori
                {
                    X = reader.ReadInt32(),
                    Y = reader.ReadInt32(),
                    Z = reader.ReadInt32()
                };
                Vertices.Add(vertex);
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
            foreach (var vertex in Vertices)
            {
                writer.Write(vertex.X);
                writer.Write(vertex.Y);
                writer.Write(vertex.Z);
            }
        }

        public uint GetSize()
        {
            const int vertexSize = 12; // Size of WoWToolbox.Core.Vectors.C3Vectori
            return (uint)Vertices.Count * vertexSize;
        }
    }
} 