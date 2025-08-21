using System.Collections.Generic;
using System.IO;
using Warcraft.NET.Files.Interfaces;

namespace WoWToolbox.Core.v2.Models.PM4.Chunks
{
    public struct MprlEntry
    {
        public ushort Unknown_0x00 { get; set; }
        public short Unknown_0x02 { get; set; }
        public ushort Unknown_0x04 { get; set; }
        public ushort Unknown_0x06 { get; set; }
        public float PositionX { get; set; }
        public float PositionY { get; set; }
        public float PositionZ { get; set; }
        public short Unknown_0x14 { get; set; }
        public ushort Unknown_0x16 { get; set; }

        // Add Position property for transform logic
        public C3Vector Position => new C3Vector(PositionX, PositionY, PositionZ);

        public const int StructSize = 24;

        public void Load(BinaryReader br)
        {
            Unknown_0x00 = br.ReadUInt16();
            Unknown_0x02 = br.ReadInt16();
            Unknown_0x04 = br.ReadUInt16();
            Unknown_0x06 = br.ReadUInt16();
            PositionX = br.ReadSingle();
            PositionY = br.ReadSingle();
            PositionZ = br.ReadSingle();
            Unknown_0x14 = br.ReadInt16();
            Unknown_0x16 = br.ReadUInt16();
        }

        public void Write(BinaryWriter bw)
        {
            bw.Write(Unknown_0x00);
            bw.Write(Unknown_0x02);
            bw.Write(Unknown_0x04);
            bw.Write(Unknown_0x06);
            bw.Write(PositionX);
            bw.Write(PositionY);
            bw.Write(PositionZ);
            bw.Write(Unknown_0x14);
            bw.Write(Unknown_0x16);
        }

        public override string ToString()
        {
            return $"MPRL Entry [Unk00:{Unknown_0x00}, Unk02:{Unknown_0x02}, Unk04:{Unknown_0x04}, Unk06:{Unknown_0x06}, Pos:({PositionX},{PositionY},{PositionZ}), Unk14:{Unknown_0x14}, Unk16:{Unknown_0x16}]";
        }
    }

    public class MPRLChunk : IIFFChunk, IBinarySerializable
    {
        public const string Signature = "MPRL";
        public string GetSignature() => Signature;
        public List<MprlEntry> Entries { get; private set; } = new();

        public uint GetSize() => (uint)(Entries.Count * MprlEntry.StructSize);

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
            if (size % MprlEntry.StructSize != 0)
                throw new InvalidDataException($"MPRL chunk size {size} is not a multiple of {MprlEntry.StructSize} bytes.");
            int entryCount = (int)(size / MprlEntry.StructSize);
            Entries = new List<MprlEntry>(entryCount);
            for (int i = 0; i < entryCount; i++)
            {
                var entry = new MprlEntry();
                entry.Load(br);
                Entries.Add(entry);
            }
        }

        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream((int)GetSize());
            using var bw = new BinaryWriter(ms);
            foreach (var entry in Entries)
            {
                entry.Write(bw);
            }
            return ms.ToArray();
        }

        public override string ToString()
        {
            return $"MPRL Chunk [{Entries.Count} Entries]";
        }
    }
} 