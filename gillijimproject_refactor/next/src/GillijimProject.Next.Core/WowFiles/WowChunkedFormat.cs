namespace GillijimProject.Next.Core.WowFiles;

/// <summary>
/// Base helpers and constants for chunked file formats (WDT/ADT/WDL).
/// FourCCs are forward in memory and reversed on disk by serializers.
/// </summary>
public abstract class WowChunkedFormat
{
    /// <summary>
    /// Size of a chunk header (FourCC + uint32 size).
    /// </summary>
    public const int ChunkHeaderSize = 8;

    /// <summary>
    /// Computes next offset from a chunk start, including optional pad byte for odd sizes.
    /// </summary>
    public static long NextOffset(long start, int dataSize)
    {
        int pad = (dataSize & 1) == 1 ? 1 : 0;
        return start + ChunkHeaderSize + (long)dataSize + pad;
    }
}
