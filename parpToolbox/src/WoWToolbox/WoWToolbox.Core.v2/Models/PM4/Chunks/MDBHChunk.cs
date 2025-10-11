using System.Collections.Generic;
using System.IO;
using System.Text;
using Warcraft.NET.Files.Interfaces;

namespace WoWToolbox.Core.v2.Models.PM4.Chunks
{
    public struct MdbhEntry
    {
        public uint Index { get; set; }
        public string Filename { get; set; }

        public void Load(BinaryReader br)
        {
            Index = br.ReadUInt32();
            Filename = MDBHChunk.ReadNullTerminatedString(br);
        }

        public void Write(BinaryWriter bw)
        {
            bw.Write(Index);
            MDBHChunk.WriteNullTerminatedString(bw, Filename);
        }

        public override string ToString()
        {
            return $"MDBH Entry [Index: {Index}, Filename: \"{Filename}\"]";
        }
    }

    public class MDBHChunk : IIFFChunk, IBinarySerializable
    {
        public const string Signature = "MDBH";
        public string GetSignature() => Signature;
        public List<MdbhEntry> Entries { get; private set; } = new();
        public uint Count => (uint)Entries.Count;

        public uint GetSize()
        {
            uint totalSize = 4; // For the count field
            foreach (var entry in Entries)
            {
                totalSize += 4; // For the index field
                totalSize += (uint)System.Text.Encoding.UTF8.GetByteCount(entry.Filename) + 1; // For filename + null terminator
            }
            return totalSize;
        }

        public void LoadBinaryData(byte[] chunkData)
        {
            using var ms = new MemoryStream(chunkData);
            using var br = new BinaryReader(ms);
            Load(br);
        }

        public void Load(BinaryReader br)
        {
            long startPosition = br.BaseStream.Position;
            long streamLength = br.BaseStream.Length;
            if (streamLength - startPosition < 4)
                throw new InvalidDataException("MDBH chunk too small to contain count field.");
            uint entryCount = br.ReadUInt32();
            Entries = new List<MdbhEntry>((int)entryCount);
            for (int i = 0; i < entryCount; i++)
            {
                if (br.BaseStream.Position + 4 > streamLength)
                    break;
                var entry = new MdbhEntry();
                entry.Load(br);
                Entries.Add(entry);
            }
        }

        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream((int)GetSize());
            using var bw = new BinaryWriter(ms);
            bw.Write(Count);
            foreach (var entry in Entries)
            {
                entry.Write(bw);
            }
            return ms.ToArray();
        }

        public override string ToString()
        {
            return $"MDBH Chunk [Count: {Count}]";
        }

        public static string ReadNullTerminatedString(BinaryReader reader)
        {
            var bytes = new List<byte>();
            byte b;
            while ((b = reader.ReadByte()) != 0)
            {
                bytes.Add(b);
            }
            return System.Text.Encoding.UTF8.GetString(bytes.ToArray());
        }

        public static void WriteNullTerminatedString(BinaryWriter writer, string value)
        {
            var bytes = System.Text.Encoding.UTF8.GetBytes(value);
            writer.Write(bytes);
            writer.Write((byte)0); // Null terminator
        }
    }
} 