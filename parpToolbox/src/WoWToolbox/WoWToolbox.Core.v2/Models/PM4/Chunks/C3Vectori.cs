using System.IO;
using Warcraft.NET.Files.Interfaces;

namespace WoWToolbox.Core.v2.Models.PM4.Chunks
{
    /// <summary>
    /// A 3D vector with integer components.
    /// </summary>
    public struct C3Vectori : IBinarySerializable
    {
        public int X { get; set; }
        public int Y { get; set; }
        public int Z { get; set; }

        public const int Size = 12; // 3 * sizeof(int)

        public uint GetSize() => (uint)Size;

        public void Load(BinaryReader br)
        {
            X = br.ReadInt32();
            Y = br.ReadInt32();
            Z = br.ReadInt32();
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
            bw.Write(Z);
        }

        public override string ToString() => $"({X}, {Y}, {Z})";
    }
} 