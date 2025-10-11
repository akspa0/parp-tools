using System;
using Warcraft.NET.Files.Interfaces;
using WoWToolbox.Core.Legacy.Interfaces;

namespace WoWToolbox.Core.Legacy.Converters
{
    /// <summary>
    /// Base class for chunk version converters
    /// </summary>
    /// <typeparam name="T">The type of chunk this converter handles</typeparam>
    public abstract class ChunkConverterBase<T> : IVersionConverter<T> where T : IIFFChunk
    {
        /// <summary>
        /// The source version this converter handles
        /// </summary>
        protected abstract int FromVersion { get; }

        /// <summary>
        /// The target version this converter produces
        /// </summary>
        protected abstract int ToVersion { get; }

        /// <inheritdoc/>
        public virtual bool CanConvert(int fromVersion, int toVersion)
        {
            return fromVersion == FromVersion && toVersion == ToVersion;
        }

        /// <inheritdoc/>
        public T Convert(ILegacyChunk source)
        {
            if (!CanConvert(source.Version, ToVersion))
            {
                throw new InvalidOperationException($"Cannot convert from version {source.Version} to {ToVersion}");
            }

            return ConvertInternal(source);
        }

        /// <summary>
        /// Internal conversion implementation to be provided by derived classes
        /// </summary>
        /// <param name="source">The source chunk to convert</param>
        /// <returns>The converted chunk</returns>
        protected abstract T ConvertInternal(ILegacyChunk source);
    }
} 