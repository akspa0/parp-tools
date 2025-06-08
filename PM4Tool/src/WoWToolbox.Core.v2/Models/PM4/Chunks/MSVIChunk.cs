using System.Collections.Generic;
using System.IO;
using Warcraft.NET.Files.Interfaces;

namespace WoWToolbox.Core.v2.Models.PM4.Chunks
{
    public class MSVIChunk : IIFFChunk, IBinarySerializable
    {
        public const string ExpectedSignature = "MSVI";
        public string GetSignature() => ExpectedSignature;
        public List<uint> Indices { get; private set; } = new List<uint>();

        public uint GetSize() => (uint)Indices.Count * sizeof(uint);

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
            if (size % sizeof(uint) != 0)
            {
                size -= (size % sizeof(uint));
            }
            int count = (int)(size / sizeof(uint));
            Indices = new List<uint>(count);
            for (int i = 0; i < count; i++)
                Indices.Add(br.ReadUInt32());
        }

        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms);
            foreach (var index in Indices)
                bw.Write(index);
            return ms.ToArray();
        }
    }
} 