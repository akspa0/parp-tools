using System;using System.IO;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port skeleton of Mcin (see lib/gillijimproject/wowfiles/Mcin.h)
/// </summary>
public class Mcin : Chunk
{
    public Mcin(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Mcin(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Mcin(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }
}
