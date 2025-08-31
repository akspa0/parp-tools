using System;
using System.IO;
using GillijimProject.WowFiles;

namespace GillijimProject.WowFiles.Alpha;

/// <summary>
/// [PORT] C# port of MphdAlpha (see lib/gillijimproject/wowfiles/alpha/MphdAlpha.{h,cpp})
/// </summary>
public class MphdAlpha : Chunk
{
    public MphdAlpha() : base("MPHD", 0, Array.Empty<byte>()) { }

    public MphdAlpha(FileStream wdtAlphaFile, int offsetInFile) : base(wdtAlphaFile, offsetInFile) { }

    public MphdAlpha(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }

    public MphdAlpha(string letters, int givenSize, byte[] data) : base("MPHD", givenSize, data) { }

    /// <summary>
    /// [PORT] Returns true if WMO-based (int at offset 8 == 2).
    /// </summary>
    public bool IsWmoBased()
    {
        if (Data.Length < 12) return false;
        int v = BitConverter.ToInt32(Data, 8);
        return v == 2;
    }

    /// <summary>
    /// [PORT] Convert to LK MPHD: 32 bytes, set first flag byte to 1 when WMO-based.
    /// </summary>
    public Mphd ToMphd()
    {
        var data = new byte[32];
        if (IsWmoBased()) data[0] = 1;
        return new Mphd("MPHD", data.Length, data);
    }
}
