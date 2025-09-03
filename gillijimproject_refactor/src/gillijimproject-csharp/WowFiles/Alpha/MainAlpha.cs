using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using GillijimProject.WowFiles.LichKing;
using GillijimProject.WowFiles;

namespace GillijimProject.WowFiles.Alpha;

/// <summary>
/// [PORT] C# port of MainAlpha (see lib/gillijimproject/wowfiles/alpha/MainAlpha.{h,cpp})
/// Parses the Alpha MAIN chunk (grid of 4096 entries, 16 bytes each) and converts to LK MAIN.
/// </summary>
public class MainAlpha : Chunk
{
    public MainAlpha() : base("MAIN", 0, Array.Empty<byte>()) { }

    public MainAlpha(FileStream wdtAlphaFile, int offsetInFile) : base(wdtAlphaFile, offsetInFile) { }

    public MainAlpha(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }

    public MainAlpha(string letters, int givenSize, byte[] data) : base("MAIN", givenSize, data) { }

    /// <summary>
    /// [PORT] Returns the 4096 MHDR offsets from the Alpha MAIN grid.
    /// Each cell is 16 bytes; the first 4 bytes store an MHDR file offset.
    /// </summary>
    public List<int> GetMhdrOffsets()
    {
        var result = new List<int>(capacity: 4096);
        const int cellSize = 16;
        int currentMainOffset = 0;
        for (int i = 0; i < 4096; i++)
        {
            int val = currentMainOffset + 4 <= Data.Length ? BitConverter.ToInt32(Data, currentMainOffset) : 0;
            result.Add(val);
            currentMainOffset += cellSize;
        }
        return result;
    }

    /// <summary>
    /// [PORT] Convert Alpha MAIN to LK MAIN (32768 bytes).
    /// For each 16-byte Alpha cell, set one flag byte in the LK 8-byte stride when offset != 0.
    /// </summary>
    public LichKing.Main ToMain()
    {
        var mhdrOffsets = GetMhdrOffsets();
        var lkOffsets = new LichKing.MhdrOffset[mhdrOffsets.Count];
        for (int i = 0; i < lkOffsets.Length; i++)
        {
            lkOffsets[i] = new LichKing.MhdrOffset
            {
                Flags = mhdrOffsets[i] != 0 ? 1 : 0,
                Offset = 0,
                Size = 0,
                Unknown = 0
            };
        }
        return new LichKing.Main(lkOffsets);
    }

    public Dictionary<int, string> GetAdtFileNames()
    {
        // This functionality is not available in Alpha, return empty dictionary
        return new Dictionary<int, string>();
    }
}
