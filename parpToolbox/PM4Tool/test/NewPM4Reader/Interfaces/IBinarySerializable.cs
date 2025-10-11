using System.IO;

namespace NewPM4Reader.Interfaces
{
    /// <summary>
    /// Interface for objects that can be serialized to and from binary data.
    /// </summary>
    public interface IBinarySerializable
    {
        /// <summary>
        /// Reads the object's data from a binary reader.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        void ReadBinary(BinaryReader reader);

        /// <summary>
        /// Writes the object's data to a binary writer.
        /// </summary>
        /// <param name="writer">The binary writer to write to.</param>
        void WriteBinary(BinaryWriter writer);
    }
} 