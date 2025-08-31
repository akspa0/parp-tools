using System;
using System.Collections.Generic;
using System.IO;
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
    public Chunk ToMain()
    {
        const int lkSize = 32768;
        var mainLkData = new byte[lkSize];
        int j = 0;
        for (int i = 0; i < 65536; i += 16)
        {
            if (i + 4 <= Data.Length && BitConverter.ToInt32(Data, i) != 0)
            {
                mainLkData[j] = 1;
            }
            j += 8;
            if (j >= lkSize) break;
        }
        return new Chunk("MAIN", lkSize, mainLkData);
    }
}
