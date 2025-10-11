using System.Collections.Generic;
using System.IO;
using Warcraft.NET.Files.Interfaces;

namespace WoWToolbox.Core.v2.Models.PM4.Chunks
{
    public class MSRNChunk : IIFFChunk, IBinarySerializable
    {
        public const string ExpectedSignature = "MSRN";
        public string GetSignature() => ExpectedSignature;
        public List<C3Vectori> Normals { get; private set; } = new List<C3Vectori>();

        public uint GetSize() => (uint)Normals.Count * C3Vectori.Size;

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
            if (size % C3Vectori.Size != 0)
            {
                size -= (size % C3Vectori.Size);
            }
            int count = (int)(size / C3Vectori.Size);
            Normals = new List<C3Vectori>(count);
            for (int i = 0; i < count; i++)
            {
                var normal = new C3Vectori
                {
                    X = br.ReadInt32(),
                    Y = br.ReadInt32(),
                    Z = br.ReadInt32()
                };
                Normals.Add(normal);
            }
        }

        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms);
            foreach (var n in Normals)
            {
                bw.Write(n.X);
                bw.Write(n.Y);
                bw.Write(n.Z);
            }
            return ms.ToArray();
        }
    }
} 