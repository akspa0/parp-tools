using System;using System.IO;using GillijimProject.WowFiles;

namespace GillijimProject.WowFiles.Alpha;

/// <summary>
/// [PORT] C# port skeleton of McnkAlpha (see lib/gillijimproject/wowfiles/alpha/McnkAlpha.h)
/// </summary>
public class McnkAlpha : Chunk
{
    public McnkAlpha(FileStream adtFile, int offsetInFile) : base(adtFile, offsetInFile) { }
    public McnkAlpha(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public McnkAlpha(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }
}
