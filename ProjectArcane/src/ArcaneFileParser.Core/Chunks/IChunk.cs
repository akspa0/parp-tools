using System;

namespace ArcaneFileParser.Core.Chunks;

/// <summary>
/// Represents a chunk in a WoW file format.
/// </summary>
public interface IChunk
{
    /// <summary>
    /// Gets the signature of the chunk.
    /// </summary>
    uint Signature { get; }

    /// <summary>
    /// Gets the size of the chunk in bytes.
    /// </summary>
    uint Size { get; }

    /// <summary>
    /// Gets the version of the chunk, if applicable.
    /// </summary>
    uint Version { get; }

    /// <summary>
    /// Gets whether this chunk has a version field.
    /// </summary>
    bool HasVersion { get; }

    /// <summary>
    /// Gets whether this chunk was successfully parsed.
    /// </summary>
    bool IsValid { get; }
} 