using Warcraft.NET.Files.Interfaces;

namespace WoWToolbox.Core.Legacy.Interfaces
{
    /// <summary>
    /// Interface for chunk version converters
    /// </summary>
    /// <typeparam name="T">The type of chunk this converter handles</typeparam>
    public interface IVersionConverter<T> where T : IIFFChunk
    {
        /// <summary>
        /// Checks if this converter can handle conversion between the specified versions
        /// </summary>
        /// <param name="fromVersion">Source version</param>
        /// <param name="toVersion">Target version</param>
        /// <returns>True if conversion is supported, false otherwise</returns>
        bool CanConvert(int fromVersion, int toVersion);

        /// <summary>
        /// Converts a legacy chunk to its modern equivalent
        /// </summary>
        /// <param name="source">The legacy chunk to convert</param>
        /// <returns>The converted modern chunk</returns>
        /// <exception cref="InvalidOperationException">Thrown when conversion is not possible</exception>
        T Convert(ILegacyChunk source);
    }
} 