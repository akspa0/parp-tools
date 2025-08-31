using System;using System.IO;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port skeleton of Mh2o (see lib/gillijimproject/wowfiles/Mh2o.h)
/// </summary>
public class Mh2o : Chunk
{
    public Mh2o(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Mh2o(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Mh2o(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }
}
