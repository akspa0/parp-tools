using System;
using System.IO;

namespace ArcaneFileParser.Core.Chunks;

/// <summary>
/// Base class for all chunks in WoW file formats.
/// </summary>
public abstract class ChunkBase : IChunk
{
    /// <inheritdoc />
    public uint Signature { get; }

    /// <inheritdoc />
    public uint Size { get; }

    /// <inheritdoc />
    public uint Version { get; }

    /// <inheritdoc />
    public virtual bool HasVersion => false;

    /// <inheritdoc />
    public bool IsValid { get; protected set; }

    /// <summary>
    /// Gets the position where the chunk data starts (after signature, size, and version if present).
    /// </summary>
    protected long DataStartPosition { get; }

    /// <summary>
    /// Gets the position where the chunk should end.
    /// </summary>
    protected long ExpectedEndPosition { get; }

    protected ChunkBase(BinaryReader reader, bool readHeader = true)
    {
        if (readHeader)
        {
            Signature = reader.ReadUInt32();
            Size = reader.ReadUInt32();
            
            if (HasVersion)
            {
                Version = reader.ReadUInt32();
                DataStartPosition = reader.BaseStream.Position;
                ExpectedEndPosition = DataStartPosition + Size - 4; // Account for version field
            }
            else
            {
                DataStartPosition = reader.BaseStream.Position;
                ExpectedEndPosition = DataStartPosition + Size;
            }
        }
        else
        {
            DataStartPosition = reader.BaseStream.Position;
            ExpectedEndPosition = DataStartPosition;
        }

        IsValid = true;
    }

    /// <summary>
    /// Ensures the reader is at the expected end position of the chunk.
    /// </summary>
    protected void EnsureAtEnd(BinaryReader reader)
    {
        if (reader.BaseStream.Position != ExpectedEndPosition)
        {
            reader.BaseStream.Seek(ExpectedEndPosition, SeekOrigin.Begin);
        }
    }

    /// <summary>
    /// Validates that the chunk has the expected signature.
    /// </summary>
    protected bool ValidateSignature(uint expectedSignature)
    {
        if (Signature != expectedSignature)
        {
            IsValid = false;
            return false;
        }
        return true;
    }

    /// <summary>
    /// Validates that the chunk has the expected version.
    /// </summary>
    protected bool ValidateVersion(uint expectedVersion)
    {
        if (!HasVersion || Version != expectedVersion)
        {
            IsValid = false;
            return false;
        }
        return true;
    }

    /// <summary>
    /// Creates a string representation of the chunk for debugging.
    /// </summary>
    public override string ToString() =>
        $"{GetType().Name} [Signature: {Signature:X8}, Size: {Size}, Valid: {IsValid}]" +
        (HasVersion ? $", Version: {Version}" : string.Empty);
} 