using System;
using System.IO;
using System.Text;

namespace WCAnalyzer.Core.Utilities
{
    /// <summary>
    /// Provides utility methods for safe type conversions.
    /// </summary>
    public static class SafeConversions
    {
        /// <summary>
        /// Safely converts a uint to int with overflow checking.
        /// </summary>
        /// <param name="value">The uint value to convert.</param>
        /// <returns>The converted int value.</returns>
        /// <exception cref="InvalidDataException">Thrown when the uint value exceeds int.MaxValue.</exception>
        public static int UIntToInt(uint value)
        {
            if (value > int.MaxValue)
            {
                throw new InvalidDataException($"Value {value} exceeds maximum allowed size (int.MaxValue)");
            }
            
            return (int)value;
        }
        
        /// <summary>
        /// Safely converts a uint to int, returning a default value if conversion is not possible.
        /// </summary>
        /// <param name="value">The uint value to convert.</param>
        /// <param name="defaultValue">The default value to return if conversion fails.</param>
        /// <returns>The converted int value or the default value if conversion fails.</returns>
        public static int UIntToIntOrDefault(uint value, int defaultValue = 0)
        {
            return value <= int.MaxValue ? (int)value : defaultValue;
        }
        
        /// <summary>
        /// Safely reads an array of bytes from a binary reader, handling uint to int conversion.
        /// </summary>
        /// <param name="reader">The binary reader.</param>
        /// <param name="count">The number of bytes to read as a uint.</param>
        /// <returns>The read bytes.</returns>
        /// <exception cref="InvalidDataException">Thrown when the count exceeds int.MaxValue.</exception>
        public static byte[] ReadBytes(BinaryReader reader, uint count)
        {
            int safeCount = UIntToInt(count);
            return reader.ReadBytes(safeCount);
        }
        
        /// <summary>
        /// Safely accesses an array using an uint index.
        /// </summary>
        /// <typeparam name="T">The type of elements in the array.</typeparam>
        /// <param name="array">The array to access.</param>
        /// <param name="index">The uint index.</param>
        /// <returns>The element at the specified index.</returns>
        /// <exception cref="InvalidDataException">Thrown when the index exceeds int.MaxValue.</exception>
        /// <exception cref="IndexOutOfRangeException">Thrown when the index is out of range.</exception>
        public static T GetArrayElementAt<T>(T[] array, uint index)
        {
            int safeIndex = UIntToInt(index);
            return array[safeIndex];
        }
        
        /// <summary>
        /// Safely gets the element at the specified index in the array, returning a default value if the index is out of range.
        /// </summary>
        /// <typeparam name="T">The type of elements in the array.</typeparam>
        /// <param name="array">The array to access.</param>
        /// <param name="index">The uint index.</param>
        /// <param name="defaultValue">The default value to return if the index is out of range.</param>
        /// <returns>The element at the specified index, or the default value if the index is out of range.</returns>
        public static T GetArrayElementAtOrDefault<T>(T[] array, uint index, T defaultValue = default)
        {
            if (array == null)
                return defaultValue;
                
            int safeIndex;
            try
            {
                safeIndex = UIntToInt(index);
            }
            catch (InvalidDataException)
            {
                return defaultValue;
            }
            
            if (safeIndex < 0 || safeIndex >= array.Length)
                return defaultValue;
                
            return array[safeIndex];
        }
        
        /// <summary>
        /// Safely gets a range of elements from the array.
        /// </summary>
        /// <typeparam name="T">The type of elements in the array.</typeparam>
        /// <param name="array">The array to access.</param>
        /// <param name="startIndex">The uint start index.</param>
        /// <param name="count">The uint count of elements to get.</param>
        /// <returns>A new array containing the specified range of elements.</returns>
        /// <exception cref="InvalidDataException">Thrown when the index or count exceeds int.MaxValue.</exception>
        public static T[] GetArrayRange<T>(T[] array, uint startIndex, uint count)
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array));
                
            int safeStartIndex = UIntToInt(startIndex);
            int safeCount = UIntToInt(count);
            
            if (safeStartIndex < 0 || safeStartIndex + safeCount > array.Length)
                throw new ArgumentOutOfRangeException(nameof(startIndex), "The specified range is outside the bounds of the array.");
                
            T[] result = new T[safeCount];
            Array.Copy(array, safeStartIndex, result, 0, safeCount);
            return result;
        }
        
        /// <summary>
        /// Reads a signed 24-bit integer (int24_t) from a binary reader.
        /// </summary>
        /// <param name="reader">The binary reader.</param>
        /// <returns>The 24-bit signed integer as an int.</returns>
        public static int ReadInt24(BinaryReader reader)
        {
            // Read 3 bytes
            byte b0 = reader.ReadByte();
            byte b1 = reader.ReadByte();
            byte b2 = reader.ReadByte();
            
            // Construct a 32-bit integer with sign extension for the 24-bit value
            int value = (b2 << 16) | (b1 << 8) | b0;
            
            // Check if the high bit is set (sign bit for 24-bit integer)
            if ((value & 0x800000) != 0)
            {
                // Extend the sign bit to the full 32-bit int
                value |= unchecked((int)0xFF000000);
            }
            
            return value;
        }
        
        /// <summary>
        /// Reads a null-terminated string from a binary reader with a maximum length.
        /// </summary>
        /// <param name="reader">The binary reader.</param>
        /// <param name="maxLength">The maximum length of the string in bytes.</param>
        /// <returns>The string.</returns>
        public static string ReadNullTerminatedString(BinaryReader reader, int maxLength)
        {
            byte[] bytes = reader.ReadBytes(maxLength);
            int endPos = 0;
            
            while (endPos < bytes.Length && bytes[endPos] != 0)
            {
                endPos++;
            }
            
            return Encoding.ASCII.GetString(bytes, 0, endPos);
        }
    }
} 