using System.IO;

namespace NewPM4Reader.Interfaces
{
    /// <summary>
    /// Interface for PM4 file chunks.
    /// </summary>
    public interface IPM4Chunk
    {
        /// <summary>
        /// Gets the signature of the chunk (4 characters).
        /// </summary>
        string Signature { get; }

        /// <summary>
        /// Reads the chunk data from a binary reader.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        void ReadBinary(BinaryReader reader);

        /// <summary>
        /// Writes the chunk data to a binary writer.
        /// </summary>
        /// <param name="writer">The binary writer to write to.</param>
        void WriteBinary(BinaryWriter writer);
    }
} 