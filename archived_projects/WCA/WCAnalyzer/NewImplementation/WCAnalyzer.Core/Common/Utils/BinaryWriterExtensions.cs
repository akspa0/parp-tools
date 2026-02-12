using System;
using System.IO;
using System.Numerics;
using System.Text;

namespace WCAnalyzer.Core.Common.Utils
{
    /// <summary>
    /// Extension methods for BinaryWriter
    /// </summary>
    public static class BinaryWriterExtensions
    {
        /// <summary>
        /// Writes a C-style null-terminated string
        /// </summary>
        /// <param name="writer">The binary writer</param>
        /// <param name="value">The string to write</param>
        public static void WriteCString(this BinaryWriter writer, string value)
        {
            if (value != null)
            {
                byte[] bytes = Encoding.UTF8.GetBytes(value);
                writer.Write(bytes);
            }
            
            writer.Write((byte)0); // Null terminator
        }
        
        /// <summary>
        /// Writes a C-style null-terminated ASCII string
        /// </summary>
        /// <param name="writer">The binary writer</param>
        /// <param name="value">The string to write</param>
        public static void WriteCStringAscii(this BinaryWriter writer, string value)
        {
            if (value != null)
            {
                byte[] bytes = Encoding.ASCII.GetBytes(value);
                writer.Write(bytes);
            }
            
            writer.Write((byte)0); // Null terminator
        }
        
        /// <summary>
        /// Writes a 4-character chunk signature
        /// </summary>
        /// <param name="writer">The binary writer</param>
        /// <param name="signature">The signature to write</param>
        public static void WriteChunkSignature(this BinaryWriter writer, string signature)
        {
            if (signature.Length != 4)
            {
                throw new ArgumentException("Chunk signature must be exactly 4 characters", nameof(signature));
            }
            
            writer.Write(signature.ToCharArray());
        }
        
        /// <summary>
        /// Writes a Vector2
        /// </summary>
        /// <param name="writer">The binary writer</param>
        /// <param name="vector">The vector to write</param>
        public static void Write(this BinaryWriter writer, Vector2 vector)
        {
            writer.Write(vector.X);
            writer.Write(vector.Y);
        }
        
        /// <summary>
        /// Writes a Vector3
        /// </summary>
        /// <param name="writer">The binary writer</param>
        /// <param name="vector">The vector to write</param>
        public static void Write(this BinaryWriter writer, Vector3 vector)
        {
            writer.Write(vector.X);
            writer.Write(vector.Y);
            writer.Write(vector.Z);
        }
        
        /// <summary>
        /// Writes a Vector4
        /// </summary>
        /// <param name="writer">The binary writer</param>
        /// <param name="vector">The vector to write</param>
        public static void Write(this BinaryWriter writer, Vector4 vector)
        {
            writer.Write(vector.X);
            writer.Write(vector.Y);
            writer.Write(vector.Z);
            writer.Write(vector.W);
        }
        
        /// <summary>
        /// Writes a fixed-length string with padding
        /// </summary>
        /// <param name="writer">The binary writer</param>
        /// <param name="value">The string to write</param>
        /// <param name="length">Total length including padding</param>
        public static void WriteFixedString(this BinaryWriter writer, string value, int length)
        {
            if (string.IsNullOrEmpty(value))
            {
                writer.Write(new byte[length]);
                return;
            }
            
            byte[] bytes = new byte[length];
            byte[] stringBytes = Encoding.UTF8.GetBytes(value);
            
            int copyLength = Math.Min(stringBytes.Length, length);
            Buffer.BlockCopy(stringBytes, 0, bytes, 0, copyLength);
            
            writer.Write(bytes);
        }
        
        /// <summary>
        /// Aligns the writer to the specified byte boundary
        /// </summary>
        /// <param name="writer">The binary writer</param>
        /// <param name="alignment">The alignment in bytes</param>
        public static void Align(this BinaryWriter writer, int alignment)
        {
            long position = writer.BaseStream.Position;
            long mod = position % alignment;
            
            if (mod != 0)
            {
                int padding = alignment - (int)mod;
                writer.Write(new byte[padding]);
            }
        }
        
        /// <summary>
        /// Writes padding bytes
        /// </summary>
        /// <param name="writer">The binary writer</param>
        /// <param name="count">Number of padding bytes</param>
        public static void WritePadding(this BinaryWriter writer, int count)
        {
            writer.Write(new byte[count]);
        }
    }
} 