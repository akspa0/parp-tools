using System;
using System.Text;

namespace ArcaneFileParser.Core.Chunks;

/// <summary>
/// Utility class for handling chunk signatures.
/// </summary>
public static class ChunkSignature
{
    /// <summary>
    /// Creates a chunk signature from a four-character code.
    /// </summary>
    public static uint FromString(string fourCC)
    {
        if (string.IsNullOrEmpty(fourCC))
            throw new ArgumentException("FourCC cannot be null or empty", nameof(fourCC));

        if (fourCC.Length != 4)
            throw new ArgumentException("FourCC must be exactly 4 characters", nameof(fourCC));

        return (uint)(
            (fourCC[3] << 24) |
            (fourCC[2] << 16) |
            (fourCC[1] << 8) |
            fourCC[0]
        );
    }

    /// <summary>
    /// Converts a chunk signature to its four-character code representation.
    /// </summary>
    public static string ToString(uint signature)
    {
        return new string(new[]
        {
            (char)(signature & 0xFF),
            (char)((signature >> 8) & 0xFF),
            (char)((signature >> 16) & 0xFF),
            (char)((signature >> 24) & 0xFF)
        });
    }

    /// <summary>
    /// Checks if a signature matches a four-character code.
    /// </summary>
    public static bool Matches(uint signature, string fourCC)
    {
        return signature == FromString(fourCC);
    }

    /// <summary>
    /// Creates a formatted string representation of a signature for debugging.
    /// </summary>
    public static string Format(uint signature)
    {
        return $"{ToString(signature)} (0x{signature:X8})";
    }
} 