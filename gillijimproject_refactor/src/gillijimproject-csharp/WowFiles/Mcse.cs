using System;

namespace GillijimProject.WowFiles;

public class Mcse : Chunk
{
    public Mcse() : base("MCSE", 0, Array.Empty<byte>()) { }

    public Mcse(string letters, int givenSize, byte[] chunkData)
        : base(letters, givenSize, chunkData)
    {
    }

    public Mcse(byte[] adtFile, int offsetInFile)
        : base(adtFile, offsetInFile)
    {
    }
}
