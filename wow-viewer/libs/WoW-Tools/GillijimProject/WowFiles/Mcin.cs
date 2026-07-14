using System;
using System.IO;
using System.Collections.Generic;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port skeleton of Mcin (see lib/gillijimproject/wowfiles/Mcin.h)
/// </summary>
public class Mcin : Chunk
{
    public Mcin(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Mcin(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Mcin(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }

    public List<int> GetMcnkOffsets()
    {
        var mcnkOffsets = new List<int>(capacity: 256);
        const int otherMcinDataSize = 16;
        int currentMcinOffset = 0;
        for (int mcnkNumber = 0; mcnkNumber < 256; ++mcnkNumber)
        {
            int val = currentMcinOffset + 4 <= Data.Length ? BitConverter.ToInt32(Data, currentMcinOffset) : 0;
            mcnkOffsets.Add(val);
            currentMcinOffset += otherMcinDataSize;
        }
        return mcnkOffsets;
    }
}
