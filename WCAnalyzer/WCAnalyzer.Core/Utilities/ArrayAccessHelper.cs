using System;
using System.IO;

namespace WCAnalyzer.Core.Utilities
{
    /// <summary>
    /// Provides helper methods for safely accessing arrays with uint indices.
    /// This class adds safety without modifying the chunk definitions themselves.
    /// </summary>
    [Obsolete("Use SafeConversions class instead. This class will be removed in a future version.")]
    public static class ArrayAccessHelper
    {
        /// <summary>
        /// Safely converts a uint to int with overflow checking.
        /// </summary>
        /// <param name="value">The uint value to convert.</param>
        /// <returns>The converted int value.</returns>
        /// <exception cref="InvalidDataException">Thrown when the uint value exceeds int.MaxValue.</exception>
        [Obsolete("Use SafeConversions.UIntToInt instead. This method will be removed in a future version.")]
        public static int SafeUIntToInt(uint value)
        {
            return SafeConversions.UIntToInt(value);
        }
        
        /// <summary>
        /// Safely gets the element at the specified index in the array.
        /// </summary>
        /// <typeparam name="T">The type of elements in the array.</typeparam>
        /// <param name="array">The array to access.</param>
        /// <param name="index">The uint index.</param>
        /// <returns>The element at the specified index.</returns>
        /// <exception cref="InvalidDataException">Thrown when the index exceeds int.MaxValue.</exception>
        /// <exception cref="IndexOutOfRangeException">Thrown when the index is out of range of the array.</exception>
        [Obsolete("Use SafeConversions.GetArrayElementAt instead. This method will be removed in a future version.")]
        public static T GetAt<T>(T[] array, uint index)
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array));
                
            return array[SafeConversions.UIntToInt(index)];
        }
        
        /// <summary>
        /// Safely gets the element at the specified index in the array, returning a default value if the index is out of range.
        /// </summary>
        /// <typeparam name="T">The type of elements in the array.</typeparam>
        /// <param name="array">The array to access.</param>
        /// <param name="index">The uint index.</param>
        /// <param name="defaultValue">The default value to return if the index is out of range.</param>
        /// <returns>The element at the specified index, or the default value if the index is out of range.</returns>
        /// <exception cref="InvalidDataException">Thrown when the index exceeds int.MaxValue.</exception>
        [Obsolete("Use SafeConversions.GetArrayElementAtOrDefault instead. This method will be removed in a future version.")]
        public static T GetAtOrDefault<T>(T[] array, uint index, T defaultValue = default)
        {
            return SafeConversions.GetArrayElementAtOrDefault(array, index, defaultValue);
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
        [Obsolete("Use SafeConversions.GetArrayRange instead. This method will be removed in a future version.")]
        public static T[] GetRange<T>(T[] array, uint startIndex, uint count)
        {
            return SafeConversions.GetArrayRange(array, startIndex, count);
        }
    }
} 