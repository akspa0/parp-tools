namespace WoWMapConverter.Core.Formats.Shared;

/// <summary>
/// Map header chunk containing offsets to other chunks.
/// </summary>
public class Mhdr : Chunk
{
    // Offset constants for each chunk type in the MHDR data
    public const int McinOffset = 4;
    public const int MtexOffset = 8;
    public const int MmdxOffset = 12;
    public const int MmidOffset = 16;
    public const int MwmoOffset = 20;
    public const int MwidOffset = 24;
    public const int MddfOffset = 28;
    public const int ModfOffset = 32;
    public const int MfboOffset = 36;
    public const int Mh2oOffset = 40;
    public const int MtxfOffset = 44;

    public Mhdr() : base("MHDR", 0, Array.Empty<byte>()) { }
    public Mhdr(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Mhdr(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Mhdr(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }

    /// <summary>
    /// Gets the flags from the header.
    /// </summary>
    public int GetFlags()
    {
        return Data.Length >= 4 ? BitConverter.ToInt32(Data, 0) : 0;
    }

    /// <summary>
    /// Gets an offset value from the header data at the specified position.
    /// </summary>
    public new int GetOffset(int position)
    {
        return Data.Length >= position + 4 ? BitConverter.ToInt32(Data, position) : 0;
    }
}
