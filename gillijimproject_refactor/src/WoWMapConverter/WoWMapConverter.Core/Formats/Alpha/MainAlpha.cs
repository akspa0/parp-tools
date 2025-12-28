using WoWMapConverter.Core.Formats.Shared;

namespace WoWMapConverter.Core.Formats.Alpha;

/// <summary>
/// Alpha WDT MAIN chunk - 64x64 grid of tile entries (16 bytes each).
/// </summary>
public class MainAlpha : Chunk
{
    private const int CellSize = 16;
    private const int GridSize = 4096; // 64x64

    public MainAlpha() : base("MAIN", 0, Array.Empty<byte>()) { }
    public MainAlpha(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public MainAlpha(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public MainAlpha(string letters, int givenSize, byte[] data) : base("MAIN", givenSize, data) { }

    /// <summary>
    /// Returns the 4096 MHDR offsets from the Alpha MAIN grid.
    /// Each cell is 16 bytes; the first 4 bytes store an MHDR file offset.
    /// </summary>
    public List<int> GetMhdrOffsets()
    {
        var result = new List<int>(GridSize);
        int offset = 0;
        for (int i = 0; i < GridSize; i++)
        {
            int val = offset + 4 <= Data.Length ? BitConverter.ToInt32(Data, offset) : 0;
            result.Add(val);
            offset += CellSize;
        }
        return result;
    }

    /// <summary>
    /// Get tile indices that have ADT data (non-zero offset).
    /// </summary>
    public List<int> GetExistingTileIndices()
    {
        var offsets = GetMhdrOffsets();
        var existing = new List<int>();
        for (int i = 0; i < offsets.Count; i++)
        {
            if (offsets[i] != 0) existing.Add(i);
        }
        return existing;
    }

    /// <summary>
    /// Convert Alpha MAIN to LK MAIN (32768 bytes = 4096 * 8 bytes per cell).
    /// </summary>
    public Chunk ToLkMain()
    {
        const int lkSize = 32768;
        var lkData = new byte[lkSize];
        int j = 0;
        for (int i = 0; i < 65536; i += CellSize)
        {
            if (i + 4 <= Data.Length && BitConverter.ToInt32(Data, i) != 0)
            {
                lkData[j] = 1; // Flag: tile exists
            }
            j += 8;
            if (j >= lkSize) break;
        }
        return new Chunk("MAIN", lkSize, lkData);
    }
}
