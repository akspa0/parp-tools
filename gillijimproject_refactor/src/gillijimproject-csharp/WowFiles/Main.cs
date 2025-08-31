using System;using System.IO;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port skeleton of Main (see lib/gillijimproject/wowfiles/Main.h)
/// </summary>
public class Main : Chunk
{
    public Main(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Main(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Main(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }
}
