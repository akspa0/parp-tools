using System;
using System.IO;
using NewPM4Reader.Interfaces;

namespace NewPM4Reader.PM4.Chunks
{
    /// <summary>
    /// Represents the MVER (Version) chunk in a PM4 file.
    /// </summary>
    public class MVER : IPM4Chunk
    {
        /// <summary>
        /// Gets the signature of the chunk.
        /// </summary>
        public string Signature => "MVER";

        /// <summary>
        /// Gets or sets the version.
        /// </summary>
        public uint Version { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="MVER"/> class with the default version.
        /// </summary>
        public MVER()
        {
            Version = 48; // Default version
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="MVER"/> class with a specific version.
        /// </summary>
        /// <param name="version">The version number.</param>
        public MVER(uint version)
        {
            Version = version;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="MVER"/> class from binary data.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        public MVER(BinaryReader reader)
        {
            ReadBinary(reader);
        }

        /// <summary>
        /// Reads the chunk data from a binary reader.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        public void ReadBinary(BinaryReader reader)
        {
            Version = reader.ReadUInt32();
        }

        /// <summary>
        /// Writes the chunk data to a binary writer.
        /// </summary>
        /// <param name="writer">The binary writer to write to.</param>
        public void WriteBinary(BinaryWriter writer)
        {
            writer.Write(Version);
        }
    }
} 