using System;

namespace GillijimProject.WowFiles;

public class Mcvt : Chunk
{
    public Mcvt() : base("MCVT", 0, Array.Empty<byte>()) { }

    public Mcvt(string letters, int givenSize, byte[] chunkData)
        : base(letters, givenSize, chunkData)
    {
    }

    public Mcvt(byte[] adtFile, int offsetInFile)
        : base(adtFile, offsetInFile)
    {
    }
}
