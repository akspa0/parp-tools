using WoWMapConverter.Core.Formats.Shared;
using WoWMapConverter.Core.Utilities;

namespace WoWMapConverter.Core.Formats.Alpha;

/// <summary>
/// Alpha WDT parser - parses the monolithic Alpha WDT format.
/// </summary>
public class WdtAlpha : WowChunkedFormat
{
    private readonly string _wdtPath;
    private readonly Chunk _mver;
    private readonly MphdAlpha _mphd;
    private readonly MainAlpha _main;
    private readonly Mdnm _mdnm;
    private readonly Monm _monm;
    private readonly Chunk _modf;

    public WdtAlpha(string wdtPath)
    {
        _wdtPath = wdtPath;
        using var fs = File.OpenRead(wdtPath);

        int offset = 0;

        // MVER
        _mver = new Chunk(fs, offset);
        offset = NextOffset(offset, _mver);

        // MPHD
        int mphdDataStart = offset + ChunkLettersAndSize;
        _mphd = new MphdAlpha(fs, offset);
        offset = NextOffset(offset, _mphd);

        // MAIN
        _main = new MainAlpha(fs, offset);
        offset = NextOffset(offset, _main);

        // MDNM and MONM offsets come from MPHD data
        int mdnmOffset = FileUtils.GetIntFromFile(fs, mphdDataStart + 4);
        _mdnm = new Mdnm(fs, mdnmOffset);

        int monmOffset = FileUtils.GetIntFromFile(fs, mphdDataStart + 12);
        _monm = new Monm(fs, monmOffset);
        offset = NextOffset(monmOffset, _monm);

        // Optional MODF after MONM when WMO-based
        if (_mphd.IsWmoBased())
        {
            _modf = new Chunk(fs, offset);
        }
        else
        {
            _modf = new Chunk("MODF", 0, Array.Empty<byte>());
        }
    }

    public string WdtPath => _wdtPath;
    public string MapName => Path.GetFileNameWithoutExtension(_wdtPath);
    public bool IsWmoBased => _mphd.IsWmoBased();

    /// <summary>
    /// Get tile indices that have ADT data.
    /// </summary>
    public List<int> GetExistingTileIndices() => _main.GetExistingTileIndices();

    /// <summary>
    /// Get all 4096 ADT offsets from MAIN chunk.
    /// </summary>
    public List<int> GetAdtOffsets() => _main.GetMhdrOffsets();

    /// <summary>
    /// Get M2/MDX model names from MDNM.
    /// </summary>
    public List<string> GetMdnmNames() => _mdnm.GetFileNames();

    /// <summary>
    /// Get WMO names from MONM.
    /// </summary>
    public List<string> GetMonmNames() => _monm.GetFileNames();

    /// <summary>
    /// Get tile coordinates from index.
    /// </summary>
    public static (int x, int y) IndexToCoords(int index) => (index % 64, index / 64);

    /// <summary>
    /// Get index from tile coordinates.
    /// </summary>
    public static int CoordsToIndex(int x, int y) => y * 64 + x;

    private static int NextOffset(int start, Chunk c)
    {
        int pad = (c.GivenSize & 1) == 1 ? 1 : 0;
        return start + ChunkLettersAndSize + c.GivenSize + pad;
    }
}
