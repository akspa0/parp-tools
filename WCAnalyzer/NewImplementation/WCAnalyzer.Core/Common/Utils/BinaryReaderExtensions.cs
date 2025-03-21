using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;

namespace WCAnalyzer.Core.Common.Utils
{
    /// <summary>
    /// Extension methods for BinaryReader
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
            var bytes = new List<byte>();
            byte b;
            
            while ((b = reader.ReadByte()) != 0)
            {
                bytes.Add(b);
            }
            
            return Encoding.UTF8.GetString(bytes.ToArray());
        }
        
        /// <summary>
        /// Reads a C-style null-terminated string with ASCII encoding
        /// </summary>
        /// <param name="reader">The binary reader</param>
        /// <returns>The string read from the stream</returns>
        public static string ReadCStringAscii(this BinaryReader reader)
        {
            var bytes = new List<byte>();
            byte b;
            
            while ((b = reader.ReadByte()) != 0)
            {
                bytes.Add(b);
            }
            
            return Encoding.ASCII.GetString(bytes.ToArray());
        }
        
        /// <summary>
        /// Reads a 4-character chunk signature as a string
        /// </summary>
        /// <param name="reader">The binary reader</param>
        /// <returns>The 4-character signature</returns>
        public static string ReadChunkSignature(this BinaryReader reader)
        {
            var chars = reader.ReadChars(4);
            return new string(chars);
        }
        
        /// <summary>
        /// Reads a Vector2
        /// </summary>
        /// <param name="reader">The binary reader</param>
        /// <returns>The vector read from the stream</returns>
        public static Vector2 ReadVector2(this BinaryReader reader)
        {
            float x = reader.ReadSingle();
            float y = reader.ReadSingle();
            return new Vector2(x, y);
        }
        
        /// <summary>
        /// Reads a Vector3
        /// </summary>
        /// <param name="reader">The binary reader</param>
        /// <returns>The vector read from the stream</returns>
        public static Vector3 ReadVector3(this BinaryReader reader)
        {
            float x = reader.ReadSingle();
            float y = reader.ReadSingle();
            float z = reader.ReadSingle();
            return new Vector3(x, y, z);
        }
        
        /// <summary>
        /// Reads a Vector4
        /// </summary>
        /// <param name="reader">The binary reader</param>
        /// <returns>The vector read from the stream</returns>
        public static Vector4 ReadVector4(this BinaryReader reader)
        {
            float x = reader.ReadSingle();
            float y = reader.ReadSingle();
            float z = reader.ReadSingle();
            float w = reader.ReadSingle();
            return new Vector4(x, y, z, w);
        }
        
        /// <summary>
        /// Reads a fixed-length string with padding
        /// </summary>
        /// <param name="reader">The binary reader</param>
        /// <param name="length">String length including potential padding</param>
        /// <returns>The string without null padding</returns>
        public static string ReadFixedString(this BinaryReader reader, int length)
        {
            byte[] bytes = reader.ReadBytes(length);
            
            // Find first zero terminator if any
            int terminatorIndex = Array.IndexOf(bytes, (byte)0);
            
            if (terminatorIndex >= 0)
            {
                return Encoding.UTF8.GetString(bytes, 0, terminatorIndex);
            }
            
            return Encoding.UTF8.GetString(bytes);
        }
        
        /// <summary>
        /// Aligns the reader to the specified byte boundary
        /// </summary>
        /// <param name="reader">The binary reader</param>
        /// <param name="alignment">The alignment in bytes</param>
        public static void Align(this BinaryReader reader, int alignment)
        {
            long position = reader.BaseStream.Position;
            long mod = position % alignment;
            
            if (mod != 0)
            {
                reader.BaseStream.Position += alignment - mod;
            }
        }
        
        /// <summary>
        /// Skips a specified number of bytes
        /// </summary>
        /// <param name="reader">The binary reader</param>
        /// <param name="count">Number of bytes to skip</param>
        public static void Skip(this BinaryReader reader, int count)
        {
            reader.BaseStream.Seek(count, SeekOrigin.Current);
        }
    }
} 