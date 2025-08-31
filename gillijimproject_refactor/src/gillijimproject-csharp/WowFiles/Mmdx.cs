using System;using System.IO;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port skeleton of Mmdx (see lib/gillijimproject/wowfiles/Mmdx.h)
/// </summary>
public class Mmdx : Chunk
{
    public Mmdx(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Mmdx(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Mmdx(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }
}
