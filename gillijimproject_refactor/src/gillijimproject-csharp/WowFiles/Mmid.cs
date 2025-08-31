using System;using System.IO;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port skeleton of Mmid (see lib/gillijimproject/wowfiles/Mmid.h)
/// </summary>
public class Mmid : Chunk
{
    public Mmid(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Mmid(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Mmid(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }
}
