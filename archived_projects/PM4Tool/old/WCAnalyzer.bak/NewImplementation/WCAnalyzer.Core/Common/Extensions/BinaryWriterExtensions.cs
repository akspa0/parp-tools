using System;
using System.IO;
using System.Text;

namespace WCAnalyzer.Core.Common.Extensions
{
    /// <summary>
    /// Extension methods for BinaryWriter to facilitate binary file writing
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
            if (value == null)
            {
                writer.Write((byte)0);
                return;
            }
            
            byte[] bytes = Encoding.ASCII.GetBytes(value);
            writer.Write(bytes);
            writer.Write((byte)0); // null terminator
        }
        
        /// <summary>
        /// Writes a fixed-length string, padding with nulls if necessary
        /// </summary>
        /// <param name="writer">The binary writer</param>
        /// <param name="value">The string to write</param>
        /// <param name="length">The length of the string to write</param>
        public static void WriteFixedString(this BinaryWriter writer, string value, int length)
        {
            if (value == null)
            {
                for (int i = 0; i < length; i++)
                {
                    writer.Write((byte)0);
                }
                return;
            }
            
            byte[] bytes = Encoding.ASCII.GetBytes(value);
            int bytesToWrite = Math.Min(bytes.Length, length);
            
            writer.Write(bytes, 0, bytesToWrite);
            
            // Pad with nulls if necessary
            for (int i = bytesToWrite; i < length; i++)
            {
                writer.Write((byte)0);
            }
        }
        
        /// <summary>
        /// Writes a 24-bit signed integer
        /// </summary>
        /// <param name="writer">The binary writer</param>
        /// <param name="value">The value to write</param>
        public static void WriteInt24(this BinaryWriter writer, int value)
        {
            // Validate range
            if (value < -8388608 || value > 8388607)
            {
                throw new ArgumentOutOfRangeException(nameof(value), "Value must be between -8388608 and 8388607 for a 24-bit signed integer");
            }
            
            writer.Write((byte)(value & 0xFF));
            writer.Write((byte)((value >> 8) & 0xFF));
            writer.Write((byte)((value >> 16) & 0xFF));
        }
        
        /// <summary>
        /// Writes a 24-bit unsigned integer
        /// </summary>
        /// <param name="writer">The binary writer</param>
        /// <param name="value">The value to write</param>
        public static void WriteUInt24(this BinaryWriter writer, uint value)
        {
            // Validate range
            if (value > 0xFFFFFF)
            {
                throw new ArgumentOutOfRangeException(nameof(value), "Value must be between 0 and 16777215 for a 24-bit unsigned integer");
            }
            
            writer.Write((byte)(value & 0xFF));
            writer.Write((byte)((value >> 8) & 0xFF));
            writer.Write((byte)((value >> 16) & 0xFF));
        }
        
        /// <summary>
        /// Writes a 4-character chunk signature
        /// </summary>
        /// <param name="writer">The binary writer</param>
        /// <param name="signature">The 4-character signature to write</param>
        public static void WriteChunkSignature(this BinaryWriter writer, string signature)
        {
            if (signature == null)
            {
                throw new ArgumentNullException(nameof(signature));
            }
            
            if (signature.Length != 4)
            {
                throw new ArgumentException("Chunk signature must be exactly 4 characters", nameof(signature));
            }
            
            byte[] bytes = Encoding.ASCII.GetBytes(signature);
            writer.Write(bytes);
        }
    }
} 