using System.Collections.Generic;
using System.IO;
using Warcraft.NET.Files.Interfaces;

namespace WoWToolbox.Core.v2.Models.PM4.Chunks
{
    public struct MsvtEntry
    {
        public float Y { get; set; }
        public float X { get; set; }
        public float Z { get; set; }

        public const int StructSize = 12;

        public override string ToString() => $"(X:{X:F3}, Y:{Y:F3}, Z:{Z:F3})";

        // Coordinate transformation constants (from original)
        private const float CoordinateOffset = 17066.666f;
        private const float HeightScaleFactor = 36.0f;

        /// <summary>
        /// Converts the internal file coordinates (YXZ floats) to world coordinates (XYZ floats).
        /// </summary>
        public System.Numerics.Vector3 ToWorldCoordinates()
        {
            float worldX = Y;  // Use Y property (first float read) for OBJ X
            float worldY = Z;  // Use Z property (third float read) for OBJ Y (Up)
            float worldZ = -X; // Use X property (second float read), negated, for OBJ Z (Depth)
            return new System.Numerics.Vector3(worldX, worldY, worldZ);
        }

        /// <summary>
        /// Creates an MsvtEntry from standard world coordinates (XYZ floats).
        /// </summary>
        public static MsvtEntry FromWorldCoordinates(System.Numerics.Vector3 worldPos)
        {
            // Inverse calculation (see original for details)
            return new MsvtEntry
            {
                Y = worldPos.X,
                X = -worldPos.Z,
                Z = worldPos.Y
            };
        }
    }

    public class MSVTChunk : IIFFChunk, IBinarySerializable
    {
        public const string ExpectedSignature = "MSVT";
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
            // MSVT records have evolved; many files use a 24-byte layout (XYZ floats + 3 unknown floats or padding).
            // Detect stride automatically: prefer 24 if evenly divisible, else 12 for old docs.
            int vertexSize = (size % 24 == 0) ? 24 : 12;
            if (size < 0) throw new InvalidDataException("Stream size is negative.");
            if (size % vertexSize != 0)
            {
                size -= (size % vertexSize);
            }
            int vertexCount = (int)(size / vertexSize);
            Vertices = new List<C3Vector>(vertexCount);
            for (int i = 0; i < vertexCount; i++)
            {
                float y = br.ReadSingle();
                float x = br.ReadSingle();
                float z = br.ReadSingle();
                // If stride is 24, skip the remaining 12 bytes (3 floats) which are currently undocumented
                if (vertexSize == 24)
                {
                    br.ReadSingle(); // unkA
                    br.ReadSingle(); // unkB
                    br.ReadSingle(); // unkC
                }
                Vertices.Add(new C3Vector { X = x, Y = y, Z = z });
            }
        }

        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms);
            foreach (var v in Vertices)
            {
                bw.Write(v.Y); // Y first
                bw.Write(v.X); // X second
                bw.Write(v.Z); // Z third
            }
            return ms.ToArray();
        }

        public override string ToString()
        {
            return $"MSVT Chunk [{Vertices.Count} Vertices] (YXZ order, 12-byte float struct)";
        }

        /// <summary>
        /// Converts a C3Vector (v2) to an MsvtEntry (original struct, YXZ order).
        /// </summary>
        public static MsvtEntry ToMsvtEntry(C3Vector v)
        {
            return new MsvtEntry { X = v.X, Y = v.Y, Z = v.Z };
        }
    }
} 