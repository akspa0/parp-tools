using System;using System.IO;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port skeleton of Mcal (see lib/gillijimproject/wowfiles/Mcal.h)
/// </summary>
public class Mcal : Chunk
{
    public Mcal(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Mcal(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Mcal(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }
}
