using System;using System.IO;using GillijimProject.WowFiles;

namespace GillijimProject.WowFiles.LichKing;

/// <summary>
/// [PORT] C# port skeleton of McnrLk (see lib/gillijimproject/wowfiles/lichking/McnrLk.h)
/// </summary>
public class McnrLk : Chunk
{
    public McnrLk(FileStream adtFile, int offsetInFile) : base(adtFile, offsetInFile) { }
    public McnrLk(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public McnrLk(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }
}
