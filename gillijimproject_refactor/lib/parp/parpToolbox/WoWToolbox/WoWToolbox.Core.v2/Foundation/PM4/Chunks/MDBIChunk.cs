using System;
using System.IO;
using Warcraft.NET.Files.Interfaces;

namespace WoWToolbox.Core.v2.Foundation.PM4.Chunks
{
    /// <summary>
    /// MDBI – Blend/Index (?) data. Structure unknown; keeping raw as with MDBF.
    /// </summary>
    public class MDBIChunk : IIFFChunk, IBinarySerializable
    {
        public const string ExpectedSignature = "MDBI";
        public string GetSignature() => ExpectedSignature;

        public byte[] Data { get; private set; } = Array.Empty<byte>();

        public uint GetSize() => (uint)(Data?.Length ?? 0);

        public void LoadBinaryData(byte[] chunkData)
        {
            if (chunkData == null) throw new ArgumentNullException(nameof(chunkData));
            Data = chunkData;
        }

        public void Load(BinaryReader br)
        {
            if (br == null) throw new ArgumentNullException(nameof(br));
            long size = br.BaseStream.Length - br.BaseStream.Position;
            if (size < 0) throw new InvalidDataException("Negative size while reading MDBI.");
            Data = br.ReadBytes((int)size);
        }

        public byte[] Serialize(long offset = 0) => Data ?? Array.Empty<byte>();
    }
}
