using System;
using System.IO;
using NewPM4Reader.Interfaces;

namespace NewPM4Reader.PM4
{
    /// <summary>
    /// Represents an unknown chunk in a PM4 file, storing raw data.
    /// </summary>
    public class UnknownChunk : IPM4Chunk
    {
        /// <summary>
        /// Gets the signature of the chunk.
        /// </summary>
        public string Signature { get; }

        /// <summary>
        /// Gets the raw data of the chunk.
        /// </summary>
        public byte[] Data { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="UnknownChunk"/> class.
        /// </summary>
        /// <param name="signature">The chunk signature.</param>
        /// <param name="data">The raw chunk data.</param>
        public UnknownChunk(string signature, byte[] data)
        {
            Signature = signature;
            Data = data ?? Array.Empty<byte>();
        }

        /// <summary>
        /// Reads the chunk data from a binary reader. 
        /// This is a no-op for UnknownChunk as it is initialized with data.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        public void ReadBinary(BinaryReader reader)
        {
            // No-op - the data is provided in the constructor
        }

        /// <summary>
        /// Writes the chunk data to a binary writer.
        /// </summary>
        /// <param name="writer">The binary writer to write to.</param>
        public void WriteBinary(BinaryWriter writer)
        {
            writer.Write(Data);
        }
    }
} 