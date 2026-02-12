using WoWMapConverter.Core.Formats.Shared;
using WoWMapConverter.Core.Utilities;

namespace WoWMapConverter.Core.Formats.Alpha;

/// <summary>
/// Alpha WDT MDNM chunk - M2/MDX model name table.
/// </summary>
public class Mdnm : Chunk
{
    public Mdnm() : base("MDNM", 0, Array.Empty<byte>()) { }
    public Mdnm(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Mdnm(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Mdnm(string letters, int givenSize, byte[] data) : base("MDNM", givenSize, data) { }

    /// <summary>
    /// Parse the NUL-separated filename table.
    /// </summary>
    public List<string> GetFileNames() => FileUtils.GetFileNames(Data);
}
