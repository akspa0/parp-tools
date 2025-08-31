using System;using System.IO;using GillijimProject.WowFiles;

namespace GillijimProject.WowFiles.LichKing;

/// <summary>
/// [PORT] C# port skeleton of McnkLk (see lib/gillijimproject/wowfiles/lichking/McnkLk.h)
/// </summary>
public class McnkLk : Chunk
{
    public McnkLk(FileStream adtFile, int offsetInFile) : base(adtFile, offsetInFile) { }
    public McnkLk(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public McnkLk(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }
}
