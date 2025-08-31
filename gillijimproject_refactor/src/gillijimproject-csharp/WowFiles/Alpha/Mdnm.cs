using System;
using System.Collections.Generic;
using System.IO;
using GillijimProject.Utilities;
using GillijimProject.WowFiles;

namespace GillijimProject.WowFiles.Alpha;

/// <summary>
/// [PORT] C# port of Mdnm (see lib/gillijimproject/wowfiles/alpha/Mdnm.{h,cpp})
/// Holds MDNM file list.
/// </summary>
public class Mdnm : Chunk
{
    public Mdnm() : base("MDNM", 0, Array.Empty<byte>()) { }

    public Mdnm(FileStream wdtAlphaFile, int offsetInFile) : base(wdtAlphaFile, offsetInFile) { }

    public Mdnm(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }

    public Mdnm(string letters, int givenSize, byte[] data) : base("MDNM", givenSize, data) { }

    public List<string> GetFilesNames()
    {
        return Utilities.GetFileNames(Data);
    }
}
