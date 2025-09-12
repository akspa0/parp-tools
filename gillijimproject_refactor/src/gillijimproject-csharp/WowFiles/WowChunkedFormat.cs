namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] Base constants for WoW chunked formats (from WowChunkedFormat.h)
/// </summary>
public abstract class WowChunkedFormat
{
    /// <summary>
    /// [PORT] Number of bytes for letters (FourCC) + size in a chunk header.
    /// </summary>
    public const int ChunkLettersAndSize = 8;

    /// <summary>
    /// [PORT] Alpha MCNK terrain header size.
    /// </summary>
    public const int McnkTerrainHeaderSize = 128;
    
    /// <summary>
    /// [PORT] Read bytes from a file stream
    /// </summary>
    /// <param name="fs">File stream to read from</param>
    /// <param name="offset">Position to start reading</param>
    /// <param name="length">Number of bytes to read</param>
    /// <returns>Byte array containing the data</returns>
    public static byte[] ReadBytes(System.IO.FileStream fs, int offset, int length)
    {
        byte[] buffer = new byte[length];
        fs.Seek(offset, System.IO.SeekOrigin.Begin);
        int total = 0;
        while (total < length)
        {
            int read = fs.Read(buffer, total, length - total);
            if (read == 0) throw new System.IO.EndOfStreamException();
            total += read;
        }
        return buffer;
    }
}
