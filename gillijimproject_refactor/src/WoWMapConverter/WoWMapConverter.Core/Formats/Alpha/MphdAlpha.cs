using WoWMapConverter.Core.Formats.Shared;

namespace WoWMapConverter.Core.Formats.Alpha;

/// <summary>
/// Alpha WDT MPHD chunk - contains flags and offsets to MDNM/MONM.
/// </summary>
public class MphdAlpha : Chunk
{
    public MphdAlpha() : base("MPHD", 0, Array.Empty<byte>()) { }
    public MphdAlpha(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public MphdAlpha(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public MphdAlpha(string letters, int givenSize, byte[] data) : base("MPHD", givenSize, data) { }

    /// <summary>
    /// Returns true if WMO-based map (int at offset 8 == 2).
    /// </summary>
    public bool IsWmoBased()
    {
        if (Data.Length < 12) return false;
        return BitConverter.ToInt32(Data, 8) == 2;
    }

    /// <summary>
    /// Get MDNM offset from MPHD data.
    /// </summary>
    public int GetMdnmOffset() => Data.Length >= 8 ? BitConverter.ToInt32(Data, 4) : 0;

    /// <summary>
    /// Get MONM offset from MPHD data.
    /// </summary>
    public int GetMonmOffset() => Data.Length >= 16 ? BitConverter.ToInt32(Data, 12) : 0;

    /// <summary>
    /// Convert to LK MPHD: 32 bytes, set first flag byte to 1 when WMO-based.
    /// </summary>
    public Chunk ToLkMphd()
    {
        var data = new byte[32];
        if (IsWmoBased()) data[0] = 1;
        return new Chunk("MPHD", data.Length, data);
    }
}
