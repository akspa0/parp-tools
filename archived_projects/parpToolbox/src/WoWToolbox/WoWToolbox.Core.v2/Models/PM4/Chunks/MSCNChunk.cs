using System.Collections.Generic;
using System.IO;
using System.Numerics;
using Warcraft.NET.Files.Interfaces;

namespace WoWToolbox.Core.v2.Models.PM4.Chunks
{
    public class MSCNChunk : IIFFChunk, IBinarySerializable
    {
        public const string Signature = "MSCN";
        public List<Vector3> ExteriorVertices { get; private set; } = new List<Vector3>();

        public string GetSignature() => Signature;

        public uint GetSize() => (uint)(ExteriorVertices.Count * 12);

        public void LoadBinaryData(byte[] inData)
        {
            using var ms = new MemoryStream(inData);
            using var br = new BinaryReader(ms);
            Read(br, (uint)inData.Length);
        }

        public void Read(BinaryReader reader, uint size)
        {
            const int vectorSize = 12;
            if (size % vectorSize != 0)
                throw new InvalidDataException($"MSCN chunk size ({size}) must be a multiple of {vectorSize}.");
            int vectorCount = (int)(size / vectorSize);
            ExteriorVertices = new List<Vector3>(vectorCount);
            for (int i = 0; i < vectorCount; i++)
            {
                float x = reader.ReadSingle();
                float y = reader.ReadSingle();
                float z = reader.ReadSingle();
                ExteriorVertices.Add(new Vector3(x, y, z));
            }
        }

        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms);
            Write(bw);
            return ms.ToArray();
        }

        /// <summary>
        /// Writes all exterior vertices to the provided BinaryWriter.
        /// </summary>
        public void Write(BinaryWriter writer)
        {
            foreach (var vertex in ExteriorVertices)
            {
                writer.Write(vertex.X);
                writer.Write(vertex.Y);
                writer.Write(vertex.Z);
            }
        }

        public override string ToString()
        {
            return $"MSCN Chunk [{ExteriorVertices.Count} Exterior Vertices (Vector3)]";
        }

        /// <summary>
        /// Converts an MSCN vertex from file coordinates (X, Y, Z) to canonical world coordinates (Y, -X, Z).
        /// </summary>
        public static Vector3 ToCanonicalWorldCoordinates(Vector3 vertex)
        {
            return new Vector3(vertex.Y, -vertex.X, vertex.Z);
        }
    }
} 