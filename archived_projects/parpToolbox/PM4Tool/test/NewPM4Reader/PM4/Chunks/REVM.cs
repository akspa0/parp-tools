using System.IO;
using NewPM4Reader.Interfaces;

namespace NewPM4Reader.PM4.Chunks
{
    /// <summary>
    /// Represents the REVM (MVER reversed) version chunk in a PM4 file.
    /// </summary>
    public class REVM : IPM4Chunk
    {
        /// <summary>
        /// Gets the signature of the chunk.
        /// </summary>
        public string Signature => "REVM";

        /// <summary>
        /// Gets or sets the version number.
        /// </summary>
        public uint Version { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="REVM"/> class.
        /// </summary>
        public REVM()
        {
            Version = 0x3010; // Default version seen in files (12304 decimal)
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="REVM"/> class from binary data.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        public REVM(BinaryReader reader)
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
        
        /// <summary>
        /// Returns a human-readable representation of the chunk data.
        /// </summary>
        /// <returns>A string representing the chunk content.</returns>
        public override string ToString()
        {
            return $"Version: 0x{Version:X} ({Version})";
        }
    }
} 