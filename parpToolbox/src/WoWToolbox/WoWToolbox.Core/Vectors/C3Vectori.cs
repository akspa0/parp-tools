using System.IO;
using System.Numerics;
using Warcraft.NET.Files.Interfaces;

namespace WoWToolbox.Core.Vectors
{
    /// <summary>
    /// A 3D vector with integer components.
    /// Represents (X, Y, Z) but is read/written as (X, Y, Z) directly unlike MsvtVertex.
    /// </summary>
    public struct C3Vectori : IBinarySerializable
    {
        public int X { get; set; }
        public int Y { get; set; }
        public int Z { get; set; } // Reverted back to int

        public const int Size = 12; // 3 * sizeof(int)

        /// <inheritdoc/>
        public uint GetSize() => (uint)Size;

        public void Load(BinaryReader br)
        {
            X = br.ReadInt32();
            Y = br.ReadInt32();
            Z = br.ReadInt32(); // Read as int
        }

        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream(Size);
            using var bw = new BinaryWriter(ms);
            Write(bw);
            return ms.ToArray();
        }

        public void Write(BinaryWriter bw)
        {
            bw.Write(X);
            bw.Write(Y);
            bw.Write(Z); // Write as int
        }

        public override string ToString() => $"({X}, {Y}, {Z})";

        // Conversion to System.Numerics.Vector3 might need adjustment later
        // public Vector3 ToVector3() => new Vector3((float)X, (float)Y, Z);
    }
} 