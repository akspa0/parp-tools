using System.IO;
using Warcraft.NET.Files;
using Warcraft.NET.Files.Interfaces;
using Warcraft.NET;

namespace WoWToolbox.Core.Navigation.PD4.Chunks
{
    /// <summary>
    /// MCRC Chunk specific to PD4 files.
    /// Contains a single uint32 value, always 0 in observed versions.
    /// </summary>
    // [ChunkSignature(CHUNK_SIGNATURE)] // Removed - Attribute not found in Warcraft.NET and base loader doesn't seem to use it.
    public class MCRCChunk : IIFFChunk
    {
        public const string CHUNK_SIGNATURE = "MCRC";

        /// <summary>
        /// Always 0 in version 48.
        /// </summary>
        public uint Unknown0x00 { get; private set; }

        public string GetSignature()
        {
            return CHUNK_SIGNATURE;
        }

        public void LoadBinaryData(byte[] data)
        {
            using (var stream = new MemoryStream(data))
            using (var reader = new BinaryReader(stream))
            {
                Unknown0x00 = reader.ReadUInt32();
                // Add checks for stream position/remaining bytes if needed
            }
        }
    }
} 