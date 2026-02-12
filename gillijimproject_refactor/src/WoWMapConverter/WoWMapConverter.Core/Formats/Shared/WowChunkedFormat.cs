namespace WoWMapConverter.Core.Formats.Shared;

/// <summary>
/// Base constants for WoW chunked formats.
/// </summary>
public abstract class WowChunkedFormat
{
    /// <summary>
    /// Number of bytes for letters (FourCC) + size in a chunk header.
    /// </summary>
    public const int ChunkLettersAndSize = 8;

    /// <summary>
    /// Alpha MCNK terrain header size.
    /// </summary>
    public const int McnkTerrainHeaderSize = 128;

    /// <summary>
    /// LK MCNK terrain header size.
    /// </summary>
    public const int McnkLkHeaderSize = 0x80;

    /// <summary>
    /// Read bytes from a file stream.
    /// </summary>
    public static byte[] ReadBytes(FileStream fs, int offset, int length)
    {
        byte[] buffer = new byte[length];
        fs.Seek(offset, SeekOrigin.Begin);
        int total = 0;
        while (total < length)
        {
            int read = fs.Read(buffer, total, length - total);
            if (read == 0) throw new EndOfStreamException();
            total += read;
        }
        return buffer;
    }

    /// <summary>
    /// Reverse a FourCC string (files store reversed).
    /// </summary>
    protected static string ReverseFourCC(string s)
    {
        if (s is null || s.Length != 4)
            throw new ArgumentException("FourCC must be 4 chars", nameof(s));
        return new string(new[] { s[3], s[2], s[1], s[0] });
    }
}
