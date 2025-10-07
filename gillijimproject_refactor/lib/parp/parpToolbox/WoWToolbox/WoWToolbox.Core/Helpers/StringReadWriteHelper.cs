using System.IO;
using System.Text;
using System.Collections.Generic;

namespace WoWToolbox.Core.Helpers
{
    /// <summary>
    /// Helper methods for reading specific string formats from BinaryReader.
    /// </summary>
    public static class StringReadHelper
    {
        /// <summary>
        /// Reads a null-terminated string from the current position of the BinaryReader.
        /// </summary>
        /// <param name="reader">The BinaryReader to read from.</param>
        /// <param name="encoding">The encoding to use for the string.</param>
        /// <returns>The string read from the stream.</returns>
        /// <exception cref="EndOfStreamException">Thrown if the end of the stream is reached before finding a null terminator.</exception>
        public static string ReadNullTerminatedString(BinaryReader reader, Encoding encoding)
        {
            List<byte> byteList = new List<byte>();
            byte currentByte;
            while (true)
            {
                try
                {
                    currentByte = reader.ReadByte();
                }
                catch (EndOfStreamException ex)
                {
                    // Re-throw with more context maybe, or handle based on requirements
                    throw new EndOfStreamException("Stream ended before null terminator found for string.", ex);
                }

                if (currentByte == 0)
                {
                    break;
                }
                byteList.Add(currentByte);
            }
            return encoding.GetString(byteList.ToArray());
        }
    }

    /// <summary>
    /// Helper methods for writing specific string formats using BinaryWriter.
    /// </summary>
    public static class StringWriteHelper
    {
        /// <summary>
        /// Writes a string followed by a null terminator to the current position of the BinaryWriter.
        /// </summary>
        /// <param name="writer">The BinaryWriter to write to.</param>
        /// <param name="value">The string to write.</param>
        /// <param name="encoding">The encoding to use for the string.</param>
        public static void WriteNullTerminatedString(BinaryWriter writer, string value, Encoding encoding)
        {
            if (value != null)
            {
                byte[] bytes = encoding.GetBytes(value);
                writer.Write(bytes);
            }
            writer.Write((byte)0); // Write the null terminator
        }
    }
} 