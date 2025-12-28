using WoWMapConverter.Core.Formats.Shared;
using WoWMapConverter.Core.Utilities;

namespace WoWMapConverter.Core.Formats.Alpha;

/// <summary>
/// Alpha WDT MONM chunk - WMO name table.
/// </summary>
public class Monm : Chunk
{
    public Monm() : base("MONM", 0, Array.Empty<byte>()) { }
    public Monm(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Monm(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Monm(string letters, int givenSize, byte[] data) : base("MONM", givenSize, data) { }

    /// <summary>
    /// Parse the NUL-separated filename table.
    /// </summary>
    public List<string> GetFileNames() => FileUtils.GetFileNames(Data);

    /// <summary>
    /// Convert to LK MWMO chunk.
    /// </summary>
    public Chunk ToMwmo() => new Chunk("MWMO", GivenSize, Data);
}
