using System.Collections.Generic;
using System.IO;
using Warcraft.NET.Files.Interfaces;

namespace WoWToolbox.Core.v2.Models.PM4.Chunks
{
    public class MPRRChunk : IIFFChunk, IBinarySerializable
    {
        public const string Signature = "MPRR";
        public string GetSignature() => Signature;
        public List<List<ushort>> Sequences { get; private set; } = new();

        public uint GetSize()
        {
            uint total = 0;
            foreach (var seq in Sequences)
                total += (uint)seq.Count * sizeof(ushort);
            return total;
        }

        public void LoadBinaryData(byte[] chunkData)
        {
            using var ms = new MemoryStream(chunkData);
            using var br = new BinaryReader(ms);
            Load(br);
        }

        public void Load(BinaryReader br)
        {
            Sequences.Clear();
            long startPosition = br.BaseStream.Position;
            long endPosition = br.BaseStream.Length;
            while (br.BaseStream.Position < endPosition)
            {
                var currentSequence = new List<ushort>();
                while (br.BaseStream.Position < endPosition)
                {
                    var value = br.ReadUInt16();
                    currentSequence.Add(value);
                    if (value == 0xFFFF)
                        break;
                }
                if (currentSequence.Count > 0)
                    Sequences.Add(currentSequence);
            }
        }

        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream((int)GetSize());
            using var bw = new BinaryWriter(ms);
            foreach (var seq in Sequences)
                foreach (var value in seq)
                    bw.Write(value);
            return ms.ToArray();
        }

        public override string ToString()
        {
            return $"MPRR Chunk [{Sequences.Count} Sequences]";
        }
    }
} 