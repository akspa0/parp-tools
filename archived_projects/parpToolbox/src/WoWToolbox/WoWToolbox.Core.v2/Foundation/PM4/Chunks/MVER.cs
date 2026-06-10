using System.IO;
using Warcraft.NET.Files.Interfaces;

namespace WoWToolbox.Core.v2.Foundation.PM4.Chunks
{
    /// <summary>
    /// Represents an MVER chunk in a PM4 file, which stores the version number of the file.
    /// This class is ported from the legacy Core.Navigation project to keep all PM4 chunks
    /// within the Core.v2.Foundation namespace tree.
    /// </summary>
    public class MVER : IIFFChunk, IBinarySerializable
    {
        public const string Signature = "MVER";

        /// <summary>
        /// Version number present in the chunk (typically 1).
        /// </summary>
        public uint Version { get; set; }

        public MVER() { }
        public MVER(uint version) => Version = version;
        public MVER(byte[] data) => LoadBinaryData(data);

        public string GetSignature() => Signature;

        public uint GetSize() => 4; // single uint32

        public void LoadBinaryData(byte[] inData)
        {
            using var ms = new MemoryStream(inData);
            using var br = new BinaryReader(ms);
            Load(br);
        }

        public void Load(BinaryReader br)
        {
            Version = br.ReadUInt32();
        }

        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms);
            bw.Write(Version);
            return ms.ToArray();
        }

        public override string ToString() => $"MVER [Version: {Version}]";
    }
}
