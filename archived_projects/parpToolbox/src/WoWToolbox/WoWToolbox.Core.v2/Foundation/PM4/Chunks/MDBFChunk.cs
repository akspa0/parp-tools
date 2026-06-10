using System;
using System.IO;
using Warcraft.NET.Files.Interfaces;

namespace WoWToolbox.Core.v2.Foundation.PM4.Chunks
{
    /// <summary>
    /// MDBF – Blend/Flag data (structure currently unknown). Implemented as raw blob so tooling can at least
    /// register its presence and size.  This brings Core.v2 chunk coverage to 100 % parity with legacy Core.
    /// </summary>
    public class MDBFChunk : IIFFChunk, IBinarySerializable
    {
        public const string ExpectedSignature = "MDBF";
        public string GetSignature() => ExpectedSignature;

        /// <summary>Raw uninterpreted payload.</summary>
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
            if (size < 0) throw new InvalidDataException("Negative size while reading MDBF.");
            Data = br.ReadBytes((int)size);
        }

        public byte[] Serialize(long offset = 0)
        {
            return Data ?? Array.Empty<byte>();
        }
    }
}
