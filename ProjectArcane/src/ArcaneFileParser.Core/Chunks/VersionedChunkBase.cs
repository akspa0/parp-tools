using System;
using System.IO;

namespace ArcaneFileParser.Core.Chunks;

/// <summary>
/// Base class for chunks that include a version field.
/// </summary>
public abstract class VersionedChunkBase : ChunkBase
{
    /// <inheritdoc />
    public override bool HasVersion => true;

    /// <summary>
    /// The expected version for this chunk type.
    /// </summary>
    protected abstract uint ExpectedVersion { get; }

    protected VersionedChunkBase(BinaryReader reader) : base(reader)
    {
        if (!ValidateVersion(ExpectedVersion))
        {
            EnsureAtEnd(reader);
        }
    }

    /// <summary>
    /// Creates a string representation of the versioned chunk for debugging.
    /// </summary>
    public override string ToString() =>
        $"{GetType().Name} [Signature: {Signature:X8}, Size: {Size}, Version: {Version}, Valid: {IsValid}]";
} 