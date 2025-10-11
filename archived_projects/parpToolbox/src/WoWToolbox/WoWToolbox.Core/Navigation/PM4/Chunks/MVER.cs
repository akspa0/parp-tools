using System;
using System.IO;
using Warcraft.NET.Files.Interfaces;

namespace WoWToolbox.Core.Navigation.PM4.Chunks
{
    /// <summary>
    /// Represents an MVER chunk in a PM4 file, which contains version information
    /// </summary>
    public class MVER : IIFFChunk, IBinarySerializable
    {
        /// <summary>
        /// The chunk signature
        /// </summary>
        public const string Signature = "MVER";

        /// <summary>
        /// Gets or sets the version number
        /// </summary>
        public uint Version { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="MVER"/> class
        /// </summary>
        public MVER() { }

        /// <summary>
        /// Initializes a new instance of the <see cref="MVER"/> class
        /// </summary>
        /// <param name="inData">The binary data</param>
        public MVER(byte[] inData)
        {
            LoadBinaryData(inData);
        }

        /// <inheritdoc/>
        public void LoadBinaryData(byte[] inData)
        {
            using (var ms = new MemoryStream(inData))
            using (var br = new BinaryReader(ms))
            {
                Load(br);
            }
        }

        /// <inheritdoc/>
        public void Load(BinaryReader br)
        {
            Version = br.ReadUInt32();
        }

        /// <inheritdoc/>
        public byte[] Serialize(long offset = 0)
        {
            using (var ms = new MemoryStream())
            using (var bw = new BinaryWriter(ms))
            {
                bw.Write(Version);
                return ms.ToArray();
            }
        }

        /// <inheritdoc/>
        public string GetSignature()
        {
            return Signature;
        }

        /// <inheritdoc/>
        public uint GetSize()
        {
            return 4; // Version is a uint (4 bytes)
        }

        public override string ToString()
        {
            return $"MVER [Version: {Version}]";
        }
    }
} 