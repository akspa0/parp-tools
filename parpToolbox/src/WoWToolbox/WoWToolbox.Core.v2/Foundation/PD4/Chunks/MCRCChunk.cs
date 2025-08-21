using System.IO;
using Warcraft.NET.Files.Interfaces;

namespace WoWToolbox.Core.v2.Foundation.PD4.Chunks
{
    /// <summary>
    /// PD4-specific MCRC chunk. According to every sample seen (version 48) this holds a single uint32 that is always zero.
    /// It appears to be a checksum placeholder or legacy field.
    /// </summary>
    public class MCRCChunk : IIFFChunk
    {
        public const string CHUNK_SIGNATURE = "MCRC"; // forward order (will be byte-swapped by the reader)

        /// <summary>
        /// Always zero in every real PD4 so far but kept as uint for completeness.
        /// </summary>
        public uint Unknown0x00 { get; private set; }

        public string GetSignature() => CHUNK_SIGNATURE;

        public void LoadBinaryData(byte[] data)
        {
            using var ms = new MemoryStream(data);
            using var br = new BinaryReader(ms);
            Unknown0x00 = br.ReadUInt32();
        }
    }
}
