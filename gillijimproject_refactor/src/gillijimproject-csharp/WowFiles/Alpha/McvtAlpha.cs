using System;using System.IO;using GillijimProject.WowFiles;

namespace GillijimProject.WowFiles.Alpha;

/// <summary>
/// [PORT] C# port skeleton of McvtAlpha (see lib/gillijimproject/wowfiles/alpha/McvtAlpha.h)
/// </summary>
public class McvtAlpha : Chunk
{
    public McvtAlpha(FileStream adtFile, int offsetInFile) : base(adtFile, offsetInFile) { }
    public McvtAlpha(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public McvtAlpha(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }
}
