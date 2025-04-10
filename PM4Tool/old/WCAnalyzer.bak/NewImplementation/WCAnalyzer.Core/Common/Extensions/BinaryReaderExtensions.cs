using System;
using System.IO;
using System.Text;

namespace WCAnalyzer.Core.Common.Extensions
{
    /// <summary>
    /// Extension methods for BinaryReader to facilitate binary file parsing
    /// </summary>
    public static class BinaryReaderExtensions
    {
        /// <summary>
        /// Reads a C-style null-terminated string
        /// </summary>
        /// <param name="reader">The binary reader</param>
        /// <returns>The string read from the stream</returns>
        public static string ReadCString(this BinaryReader reader)
        {
            var bytes = new System.Collections.Generic.List<byte>();
            byte b;
            
            while ((b = reader.ReadByte()) != 0)
            {
                bytes.Add(b);
            }
            
            return Encoding.ASCII.GetString(bytes.ToArray());
        }
        
        /// <summary>
        /// Reads a fixed-length string and trims any null terminators
        /// </summary>
        /// <param name="reader">The binary reader</param>
        /// <param name="length">The length of the string to read</param>
        /// <returns>The string read from the stream</returns>
        public static string ReadFixedString(this BinaryReader reader, int length)
        {
            byte[] bytes = reader.ReadBytes(length);
            int nullTerminatorIndex = Array.IndexOf(bytes, (byte)0);
            
            if (nullTerminatorIndex >= 0)
            {
                return Encoding.ASCII.GetString(bytes, 0, nullTerminatorIndex);
            }
            
            return Encoding.ASCII.GetString(bytes);
        }
        
        /// <summary>
        /// Reads a 24-bit signed integer
        /// </summary>
        /// <param name="reader">The binary reader</param>
        /// <returns>The 24-bit signed integer read from the stream</returns>
        public static int ReadInt24(this BinaryReader reader)
        {
            byte[] bytes = reader.ReadBytes(3);
            
            // Check if the highest bit is set (negative number)
            if ((bytes[2] & 0x80) != 0)
            {
                // Fill with 0xFF for the 4th byte to maintain sign extension
                return (int)(0xFF000000 | (bytes[2] << 16) | (bytes[1] << 8) | bytes[0]);
            }
            
            return (int)((bytes[2] << 16) | (bytes[1] << 8) | bytes[0]);
        }
        
        /// <summary>
        /// Reads a 24-bit unsigned integer
        /// </summary>
        /// <param name="reader">The binary reader</param>
        /// <returns>The 24-bit unsigned integer read from the stream</returns>
        public static uint ReadUInt24(this BinaryReader reader)
        {
            byte[] bytes = reader.ReadBytes(3);
            return (uint)((bytes[2] << 16) | (bytes[1] << 8) | bytes[0]);
        }
        
        /// <summary>
        /// Reads a 4-character chunk signature as a string
        /// </summary>
        /// <param name="reader">The binary reader</param>
        /// <returns>The 4-character signature</returns>
        public static string ReadChunkSignature(this BinaryReader reader)
        {
            return Encoding.ASCII.GetString(reader.ReadBytes(4));
        }
    }
} 