using System;using System.IO;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port skeleton of Wdl (see lib/gillijimproject/wowfiles/Wdl.h)
/// </summary>
public class Wdl : Chunk
{
    public Wdl(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Wdl(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Wdl(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }
}
