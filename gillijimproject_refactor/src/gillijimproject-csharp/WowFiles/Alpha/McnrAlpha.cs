using System;using System.IO;using GillijimProject.WowFiles;

namespace GillijimProject.WowFiles.Alpha;

/// <summary>
/// [PORT] C# port skeleton of McnrAlpha (see lib/gillijimproject/wowfiles/alpha/McnrAlpha.h)
/// </summary>
public class McnrAlpha : Chunk
{
    public McnrAlpha(FileStream adtFile, int offsetInFile) : base(adtFile, offsetInFile) { }
    public McnrAlpha(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public McnrAlpha(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }
}
