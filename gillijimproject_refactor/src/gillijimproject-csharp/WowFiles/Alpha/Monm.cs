using System;
using System.Collections.Generic;
using System.IO;
using GillijimProject.Utilities;
using GillijimProject.WowFiles;

namespace GillijimProject.WowFiles.Alpha;

/// <summary>
/// [PORT] C# port of Monm (see lib/gillijimproject/wowfiles/alpha/Monm.{h,cpp})
/// Holds MONM file list and can convert to LK MWMO chunk.
/// </summary>
public class Monm : Chunk
{
    public Monm() : base("MONM", 0, Array.Empty<byte>()) { }

    public Monm(FileStream wdtAlphaFile, int offsetInFile) : base(wdtAlphaFile, offsetInFile) { }

    public Monm(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }

    public Monm(string letters, int givenSize, byte[] data) : base("MONM", givenSize, data) { }

    public List<string> GetFilesNames()
    {
        return Utilities.GetFileNames(Data);
    }

    public Chunk ToMwmo()
    {
        return new Chunk("MWMO", GivenSize, Data);
    }
}
