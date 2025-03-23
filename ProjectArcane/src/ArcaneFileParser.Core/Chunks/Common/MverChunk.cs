using System.IO;

namespace ArcaneFileParser.Core.Chunks.Common;

/// <summary>
/// Common version chunk used across multiple WoW file formats.
/// </summary>
public class MverChunk : ChunkBase
{
    /// <summary>
    /// The expected signature for MVER chunks.
    /// </summary>
    public const uint ExpectedSignature = 0x5245564D; // "MVER"

    /// <summary>
    /// Creates a new instance of the MVER chunk.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    public MverChunk(BinaryReader reader) : base(reader)
    {
        if (!ValidateSignature(ExpectedSignature))
        {
            return;
        }

        // MVER chunks are always 4 bytes containing the version
        if (Size != 4)
        {
            IsValid = false;
            EnsureAtEnd(reader);
            return;
        }

        Version = reader.ReadUInt32();
        EnsureAtEnd(reader);
    }

    /// <summary>
    /// Creates a string representation of the chunk for debugging.
    /// </summary>
    public override string ToString() =>
        $"MVER [Version: {Version}, Valid: {IsValid}]";
} 