using System.Collections.Generic;
using System.IO;
using Warcraft.NET.Files.Interfaces;

namespace WoWToolbox.Core.v2.Models.PM4.Chunks
{
    public struct MspvEntry
    {
        public float X { get; set; }
        public float Y { get; set; }
        public float Z { get; set; }

        public const int StructSize = 12;
    }

    public class MSPVChunk : IIFFChunk, IBinarySerializable
    {
        public const string ExpectedSignature = "MSPV";
        public string GetSignature() => ExpectedSignature;
        public List<C3Vector> Vertices { get; private set; } = new List<C3Vector>();

        public uint GetSize() => (uint)(Vertices.Count * 12);

        public void LoadBinaryData(byte[] chunkData)
        {
            using var ms = new MemoryStream(chunkData);
            using var br = new BinaryReader(ms);
            Load(br);
        }

        public void Load(BinaryReader br)
        {
            long startPosition = br.BaseStream.Position;
            long size = br.BaseStream.Length - startPosition;
            int vertexSize = 12;
            if (size < 0) throw new InvalidDataException("Stream size is negative.");
            if (size % vertexSize != 0)
            {
                size -= (size % vertexSize);
            }
            int vertexCount = (int)(size / vertexSize);
            Vertices = new List<C3Vector>(vertexCount);
            for (int i = 0; i < vertexCount; i++)
            {
                float x = br.ReadSingle();
                float y = br.ReadSingle();
                float z = br.ReadSingle();
                Vertices.Add(new C3Vector { X = x, Y = y, Z = z });
            }
        }

        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms);
            foreach (var v in Vertices)
            {
                bw.Write(v.X);
                bw.Write(v.Y);
                bw.Write(v.Z);
            }
            return ms.ToArray();
        }
    }
} 