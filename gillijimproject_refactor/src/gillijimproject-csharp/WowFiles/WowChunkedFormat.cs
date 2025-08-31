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
}
