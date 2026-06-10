namespace ParpToolbox.Formats.P4.Chunks.Common;

using System;
using System.IO;

/// <summary>
/// Helper for reading FourCC identifiers from on-disk PM4 / PD4 chunks.
/// Chunk IDs are stored little-endian (bytes reversed) compared to the
/// canonical strings used in documentation (e.g. bytes "REVM" ⇒ "MVER").
/// </summary>
public static class FourCc
{
    /// <summary>
    /// Reads four bytes from the given <see cref="BinaryReader"/> and returns the canonical
    /// FourCC string (byte-reversed).
    /// </summary>
    public static string Read(BinaryReader br)
    {
        Span<byte> bytes = stackalloc byte[4];
        int read = br.Read(bytes);
        if (read != 4)
            throw new EndOfStreamException("Unexpected EOF while reading FourCC");
        Span<char> chars = stackalloc char[4];
        // Reverse byte order
        chars[0] = (char)bytes[3];
        chars[1] = (char)bytes[2];
        chars[2] = (char)bytes[1];
        chars[3] = (char)bytes[0];
        return new string(chars);
    }
}
