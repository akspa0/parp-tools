using System.Collections.Generic;
using System.IO;
using System.Text;

namespace WoWToolbox.Core.v2.Models.PM4.Chunks
{
    public struct MdbhEntry
    {
        public uint Index { get; set; }
        public string Filename { get; set; }
    }

    public class MDBHChunk
    {
        public List<MdbhEntry> Entries { get; private set; } = new();

        public void Read(BinaryReader br, long size)
        {
            if (size < sizeof(uint)) return;

            var numEntries = br.ReadUInt32();
            Entries = new List<MdbhEntry>((int)numEntries);

            for (var i = 0; i < numEntries; i++)
            {
                if (br.BaseStream.Position + sizeof(uint) > br.BaseStream.Length)
                    break;
                
                var index = br.ReadUInt32();
                var filename = ReadNullTerminatedString(br);
                
                Entries.Add(new MdbhEntry { Index = index, Filename = filename });
            }
        }

        private static string ReadNullTerminatedString(BinaryReader reader)
        {
            var bytes = new List<byte>();
            byte b;
            while ((b = reader.ReadByte()) != 0)
            {
                bytes.Add(b);
            }
            return Encoding.UTF8.GetString(bytes.ToArray());
        }
    }
} 