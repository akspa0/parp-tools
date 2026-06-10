using System;
using System.Collections.Generic;
using System.IO;
using Warcraft.NET.Files.Interfaces;
using WoWToolbox.Core.Vectors; // Ensure this using is present

namespace WoWToolbox.Core.Navigation.PM4.Chunks
{
    /// <summary>
    /// Represents the MSCN chunk, believed to contain normal vectors or similar.
    /// Based on documentation structure C3Vectori mscn[];
    /// Documentation explicitly states 'n != normals'.
    /// </summary>
    public class MSCNChunk : IIFFChunk, IBinarySerializable
    {
        public const string Signature = "MSCN";

        // Renamed from Normals based on documentation note 'n != normals'
        public List<C3Vectori> Vectors { get; private set; }

        public MSCNChunk()
        {
            Vectors = new List<C3Vectori>();
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
            const int vectorSize = 12; // Size of C3Vectori (3 * int)
            if (size % vectorSize != 0)
            {
                throw new InvalidDataException($"MSCN chunk size ({size}) must be a multiple of {vectorSize}.");
            }

            int vectorCount = (int)(size / vectorSize);
            Vectors = new List<C3Vectori>(vectorCount);

            for (int i = 0; i < vectorCount; i++)
            {
                C3Vectori vector = new C3Vectori
                {
                    X = reader.ReadInt32(),
                    Y = reader.ReadInt32(),
                    Z = reader.ReadInt32()
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

        public void Write(BinaryWriter writer)
        {
            foreach (var vector in Vectors)
            {
                writer.Write(vector.X);
                writer.Write(vector.Y);
                writer.Write(vector.Z);
            }
        }

        public uint GetSize()
        {
            const int vectorSize = 12;
            return (uint)Vectors.Count * vectorSize;
        }

        public override string ToString()
        {
            return $"MSCN Chunk [{Vectors.Count} Vectors]";
        }
    }
} 