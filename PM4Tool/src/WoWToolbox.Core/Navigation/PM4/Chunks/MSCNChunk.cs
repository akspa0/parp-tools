using System;
using System.Collections.Generic;
using System.IO;
using Warcraft.NET.Files.Interfaces;
using System.Numerics;

namespace WoWToolbox.Core.Navigation.PM4.Chunks
{
    /// <summary>
    /// Represents the MSCN chunk, containing exterior vertices (previously misinterpreted as normals).
    /// These vertices define the exterior boundary of the object in the PM4 file.
    /// </summary>
    public class MSCNChunk : IIFFChunk, IBinarySerializable
    {
        public const string Signature = "MSCN";

        /// <summary>
        /// The exterior vertices for the object, as defined by the MSCN chunk.
        /// </summary>
        public List<Vector3> ExteriorVertices { get; private set; }

        public MSCNChunk()
        {
            ExteriorVertices = new List<Vector3>();
        }

        public string GetSignature() => Signature;

        public void LoadBinaryData(byte[] inData)
        {
            using var ms = new MemoryStream(inData);
            using var br = new BinaryReader(ms);
            Read(br, (uint)inData.Length);
        }

        // Read as Vector3 (float)
        public void Read(BinaryReader reader, uint size)
        {
            const int vectorSize = sizeof(float) * 3; // 12 bytes
            if (size % vectorSize != 0)
            {
                throw new InvalidDataException($"MSCN chunk size ({size}) must be a multiple of {vectorSize}.");
            }

            int vectorCount = (int)(size / vectorSize);
            ExteriorVertices = new List<Vector3>(vectorCount);

            for (int i = 0; i < vectorCount; i++)
            {
                Vector3 vertex = new Vector3
                {
                    X = reader.ReadSingle(),
                    Y = reader.ReadSingle(),
                    Z = reader.ReadSingle()
                };
                ExteriorVertices.Add(vertex);
            }
        }

        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms);
            Write(bw);
            return ms.ToArray();
        }

        // Write as Vector3 (float)
        public void Write(BinaryWriter writer)
        {
            foreach (var vertex in ExteriorVertices)
            {
                writer.Write(vertex.X);
                writer.Write(vertex.Y);
                writer.Write(vertex.Z);
            }
        }

        public uint GetSize()
        {
            const int vectorSize = sizeof(float) * 3;
            return (uint)ExteriorVertices.Count * vectorSize;
        }

        public override string ToString()
        {
            return $"MSCN Chunk [{ExteriorVertices.Count} Exterior Vertices (Vector3)]";
        }
    }
} 