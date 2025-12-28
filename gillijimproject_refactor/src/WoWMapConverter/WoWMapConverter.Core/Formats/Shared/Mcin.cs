namespace WoWMapConverter.Core.Formats.Shared;

/// <summary>
/// MCIN chunk containing offsets to all 256 MCNK chunks.
/// </summary>
public class Mcin : Chunk
{
    private const int EntrySize = 16; // 4 bytes offset + 4 bytes size + 8 bytes unused

    public Mcin(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Mcin(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Mcin(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }

    /// <summary>
    /// Get all 256 MCNK offsets from the MCIN data.
    /// </summary>
    public List<int> GetMcnkOffsets()
    {
        var offsets = new List<int>(256);
        for (int i = 0; i < 256; i++)
        {
            int dataOffset = i * EntrySize;
            int val = dataOffset + 4 <= Data.Length ? BitConverter.ToInt32(Data, dataOffset) : 0;
            offsets.Add(val);
        }
        return offsets;
    }

    /// <summary>
    /// Get MCNK sizes from the MCIN data.
    /// </summary>
    public List<int> GetMcnkSizes()
    {
        var sizes = new List<int>(256);
        for (int i = 0; i < 256; i++)
        {
            int dataOffset = i * EntrySize + 4; // Size is at offset 4 within entry
            int val = dataOffset + 4 <= Data.Length ? BitConverter.ToInt32(Data, dataOffset) : 0;
            sizes.Add(val);
        }
        return sizes;
    }
}
