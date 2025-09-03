using System;

namespace GillijimProject.WowFiles;

public class Mcly : Chunk
{
    public Mcly() : base("MCLY", 0, Array.Empty<byte>()) { }

    public Mcly(string letters, int givenSize, byte[] chunkData)
        : base(letters, givenSize, chunkData)
    {
    }

    public Mcly(byte[] adtFile, int offsetInFile)
        : base(adtFile, offsetInFile)
    {
    }
}
