using System;using System.IO;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port skeleton of Modf (see lib/gillijimproject/wowfiles/Modf.h)
/// </summary>
public class Modf : Chunk
{
    public Modf(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Modf(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Modf(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }
}
