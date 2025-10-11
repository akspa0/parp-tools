using Warcraft.NET.Files.Interfaces;

namespace WoWToolbox.Core.Legacy.Interfaces
{
    /// <summary>
    /// Interface for legacy chunk implementations that can be converted to modern formats
    /// </summary>
    public interface ILegacyChunk : IIFFChunk
    {
        /// <summary>
        /// Gets the version of the legacy chunk format
        /// </summary>
        int Version { get; }

        /// <summary>
        /// Checks if this chunk can be converted to the modern format
        /// </summary>
        /// <returns>True if conversion is possible, false otherwise</returns>
        bool CanConvertToModern();

        /// <summary>
        /// Converts this legacy chunk to its modern equivalent
        /// </summary>
        /// <returns>The modern chunk implementation</returns>
        /// <exception cref="InvalidOperationException">Thrown when conversion is not possible</exception>
        IIFFChunk ConvertToModern();
    }
} 