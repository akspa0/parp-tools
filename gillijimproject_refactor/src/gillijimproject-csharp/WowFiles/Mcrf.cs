using System;using System.IO;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port skeleton of Mcrf (see lib/gillijimproject/wowfiles/Mcrf.h)
/// </summary>
public class Mcrf : Chunk
{
    public Mcrf(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Mcrf(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Mcrf(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }
}
