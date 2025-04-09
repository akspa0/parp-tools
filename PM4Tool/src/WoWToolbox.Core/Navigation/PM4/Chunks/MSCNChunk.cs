using System;
using System.Collections.Generic;
using System.IO;
using Warcraft.NET.Files.Interfaces;
using System.Numerics;
using WoWToolbox.Core.Vectors; // Keep for C3Vectori if used elsewhere, remove if not

namespace WoWToolbox.Core.Navigation.PM4.Chunks
{
    /// <summary>
    /// Represents the MSCN chunk, containing normal vectors (read as floats).
    /// Documentation notes 'n != normals', but treating as floats for export.
    /// </summary>
    public class MSCNChunk : IIFFChunk, IBinarySerializable
    {
        public const string Signature = "MSCN";

        // Changed from C3Vectori to Vector3 (float-based)
        public List<Vector3> Vectors { get; private set; }

        public MSCNChunk()
        {
            // Initialize with Vector3 list
            Vectors = new List<Vector3>();
        }

        public string GetSignature() => Signature;

        public void LoadBinaryData(byte[] inData)
        {
            using var ms = new MemoryStream(inData);
            using var br = new BinaryReader(ms);
            Read(br, (uint)inData.Length);
        }

        // Updated to read floats (Vector3)
        public void Read(BinaryReader reader, uint size)
        {
            const int vectorSize = sizeof(float) * 3; // Size of Vector3 (3 * float)
            if (size % vectorSize != 0)
            {
                throw new InvalidDataException($"MSCN chunk size ({size}) must be a multiple of {vectorSize}.");
            }

            int vectorCount = (int)(size / vectorSize);
            Vectors = new List<Vector3>(vectorCount);

            for (int i = 0; i < vectorCount; i++)
            {
                // Read X, Y, Z as floats
                Vector3 vector = new Vector3
                {
                    X = reader.ReadSingle(),
                    Y = reader.ReadSingle(),
                    Z = reader.ReadSingle()
                };
                Vectors.Add(vector);
            }
        }

        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms);
            Write(bw);
            return ms.ToArray();
        }

        // Updated to write floats (Vector3)
        public void Write(BinaryWriter writer)
        {
            foreach (var vector in Vectors)
            {
                writer.Write(vector.X);
                writer.Write(vector.Y);
                writer.Write(vector.Z);
            }
        }

        // Updated to use Vector3 size
        public uint GetSize()
        {
            const int vectorSize = sizeof(float) * 3;
            return (uint)Vectors.Count * vectorSize;
        }

        public override string ToString()
        {
            // Update type in string
            return $"MSCN Chunk [{Vectors.Count} Vectors (Vector3)]";
        }
    }
} 